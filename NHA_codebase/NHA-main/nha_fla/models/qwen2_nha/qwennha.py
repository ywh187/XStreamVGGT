# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .configuration_qwen2_nha import Qwen2NHAConfig
from nha_fla.ops.nha import fused_recurrent_nha, chunk_nha

from nha_fla.models.nha_cache import NHACache

from transformers.models.qwen2.modeling_qwen2 import (
    repeat_kv,
    apply_rotary_pos_emb,
    Qwen2RotaryEmbedding,
)

from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)

class Qwen2NativeHybridAttention(nn.Module):
    def __init__(
        self, 
        config: Qwen2NHAConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        
        self.num_slots = config.num_slots
        self.window_size = config.window_size
        self.gate_logit_normalizer = 8

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        self.g_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.num_slots, bias=False)

        if config.block_size is not None and config.transformer_idx is not None:
            if self.layer_idx % config.block_size == config.transformer_idx:
                self.window_size = 2048
            else:
                self.window_size = config.window_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[NHACache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        last_state = None
        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            last_state = past_key_value[self.layer_idx]

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        g = g.view(bsz, q_len, -1, self.num_slots).transpose(1, 2)

        q = query_states
        k = repeat_kv(key_states, self.num_key_value_groups)
        v = repeat_kv(value_states, self.num_key_value_groups)
        g = repeat_kv(g, self.num_key_value_groups)

        # norm
        # q = F.softmax(q, dim=-1)
        # k = F.softmax(k, dim=-1)

        g = F.logsigmoid(g) / self.gate_logit_normalizer # (b, h, n, m)
        s = (1 - g.exp()).to(g.dtype)

        # dealing with left-padding
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                s = s.mul_(attention_mask[:, :, :, -s.shape[2]:])
                v = v.mul_(attention_mask[:, :, :, -v.shape[2]:])
            else:
                s = s.mul_(attention_mask[:, None, -s.shape[2]:, None])
                v = v.mul_(attention_mask[:, None, -v.shape[2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        q, k, v, s, g = (x.transpose(1, 2).contiguous() for x in (q, k, v, s, g))

        if past_key_value is not None:
            # prepare gsa input
            last_k, last_v, last_g = past_key_value.get_pop_kvf(self.layer_idx, self.window_size)
            
            if last_k is None:
                b, _, h, d_k, d_v = *q.shape, v.shape[-1]
                last_k, last_v, last_g = torch.zeros((b, 1, h, d_k), dtype=k.dtype, device=k.device), \
                                            torch.zeros((b, 1, h, d_v), dtype=v.dtype, device=v.device), \
                                            torch.zeros((b, 1, h, self.num_slots), dtype=g.dtype, device=g.device)
                last_s = last_g
            
            else:
                last_k = rearrange(last_k, '... (h d) -> ... h d', h=self.num_heads)
                last_v = rearrange(last_v, '... (h d) -> ... h d', h=self.num_heads)
                last_g = rearrange(last_g, '... (h d) -> ... h d', h=self.num_heads)
                last_s = (1 - last_g.exp()).to(last_g.dtype)
            # update swa cache
            k_cached, v_cached, g_cached = past_key_value.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1), g.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=0,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            cache_has_content = past_key_value.get_seq_length(self.layer_idx) > 0
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', h=self.num_heads)
                v = rearrange(v, '... (h d) -> ... h d', h=self.num_heads)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(v, position_ids)
            sq, sk = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        else:
            if q.shape[1] == 1:
                # inferece handle swa position
                window_len = k.shape[1]
                swa_pos = position_ids - torch.arange(window_len - 1, -1, -1, device=position_ids.device).unsqueeze(0)
                cos, sin = self.rotary_emb(v.transpose(1, 2), swa_pos)
                sq, sk = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
                sq = sq[:, -1:, ...]
            else:
                cos, sin = position_embeddings
                sq, sk = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        input_dtype = sq.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            sq = sq.to(target_dtype)
            sk = sk.to(target_dtype)
            v = v.to(target_dtype)

        if self.training or q.shape[1] > 1:
            if self.window_size <= 64:
                rotary_q, rotary_k = sq, sk

                prefix_k = torch.zeros(k.size(0), self.window_size, k.size(2), k.size(3), dtype=k.dtype, device=k.device)
                prefix_v = torch.zeros(v.size(0), self.window_size, v.size(2), v.size(3), dtype=v.dtype, device=v.device)
                prefix_s = torch.zeros(s.size(0), self.window_size, s.size(2), s.size(3), dtype=s.dtype, device=s.device)
                prefix_g = torch.zeros(g.size(0), self.window_size, g.size(2), g.size(3), dtype=g.dtype, device=g.device)

                shift_k = torch.cat([prefix_k, k], dim=1)
                shift_v = torch.cat([prefix_v, v], dim=1)
                shift_s = torch.cat([prefix_s, s], dim=1)
                shift_g = torch.cat([prefix_g, g], dim=1)

                rotary_k = torch.cat([prefix_k, rotary_k], dim=1)

                o, recurrent_state = chunk_nha(
                    q=q,
                    k=shift_k,
                    v=shift_v,
                    rotary_q=rotary_q,
                    rotary_k=rotary_k,
                    window_size=self.window_size,
                    s=shift_s,
                    g=shift_g,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=None,
                    head_first=False,
                    rotary=None,
                )
            else:
                rotary_q, rotary_k = sq, sk
                prefix_k = torch.zeros(k.size(0), self.num_slots, k.size(2), k.size(3), dtype=k.dtype, device=k.device)
                prefix_v = torch.zeros(v.size(0), self.num_slots, v.size(2), v.size(3), dtype=v.dtype, device=v.device)
                shift_q = torch.cat([prefix_k, rotary_q], dim=1)
                shift_k = torch.cat([prefix_k, rotary_k], dim=1)
                shift_v = torch.cat([prefix_v, v], dim=1)

                sliding_window = self.naive_swa(sq, sk, self.window_size)
                sliding_window_prob = sliding_window.softmax(-1)
                o = torch.einsum('bthw,bwhd->bthd',
                                sliding_window_prob,
                                v)
        else:
            if self.window_size <= 64:
                sliding_window = self.naive_swa(sq, sk, self.window_size)
                shift_k, shift_v, shift_g, shift_s = last_k, last_v, last_g, last_s

                o, sliding_window_prob, recurrent_state = fused_recurrent_nha(
                    q=q,
                    k=shift_k,
                    v=shift_v,
                    sliding_window=sliding_window,
                    s=shift_s,
                    g=shift_g,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    # scale=scale,
                    head_first=False
                )
                o += torch.einsum('bthw,bwhd->bthd',
                                    sliding_window_prob.to(v.dtype),
                                    v)
            else:
                prefix_k = torch.zeros(k.size(0), self.num_slots, k.size(2), k.size(3), dtype=k.dtype, device=k.device)
                prefix_v = torch.zeros(v.size(0), self.num_slots, v.size(2), v.size(3), dtype=v.dtype, device=v.device)
                shift_k = torch.cat([prefix_k, sk], dim=1)
                shift_v = torch.cat([prefix_v, v], dim=1)
                sliding_window = self.naive_swa(sq, sk, self.window_size)
                sliding_window_prob = sliding_window.softmax(-1)
                o = torch.einsum('bthw,bwhd->bthd',
                                sliding_window_prob,
                                v)

        if past_key_value is not None:
            past_key_value.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        o = rearrange(o, 'b n h d -> b n (h d)')
        o = self.o_proj(o)

        return o, None

    def naive_swa(self, q: torch.Tensor, k: torch.Tensor, W: int):
        seq_len = q.shape[1]
        i = torch.arange(seq_len, device=q.device).view(-1, 1)  # (T, 1)
        j = torch.arange(seq_len, device=q.device).view(1, -1)  # (1, T)

        left_bound = torch.clamp(i - W + 1, min=0)          # (T, 1)
        valid_mask = (j >= left_bound) & (j <= i)                    # (T, T)

        scale_factor = 1 / (q.shape[-1] ** 0.5)
        qk = torch.einsum('bthd,bnhd->bhtn', q, k) * scale_factor
        qk = qk.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), -1e7)

        return qk.transpose(1, 2)
    
    def reference_swa(self, q: torch.Tensor, k: torch.Tensor, W: int):
        q_len = q.shape[1]
        k_len = k.shape[1]
        i = torch.arange(q_len, device=q.device).view(-1, 1)  # (T, 1)
        j = torch.arange(k_len, device=q.device).view(1, -1)  # (1, T)
        
        right_bound = i + W - 1                               # (T, 1)
        valid_mask = (j >= i) & (j <= right_bound)                    # (T, T)

        scale_factor = q.shape[-1] ** -0.5
        origin_dtype = q.dtype
        qk = (torch.einsum('bthd,bnhd->bhtn', q.to(torch.float32), k.to(torch.float32)) * scale_factor).to(origin_dtype)

        qk = qk.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        return qk.transpose(1, 2)