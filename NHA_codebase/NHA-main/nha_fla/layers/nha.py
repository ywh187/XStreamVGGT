# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import time
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nha_fla.models.nha.configuration_nha import NHAConfig
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.activations import swish
from fla.modules.feature_map import (ReLUFeatureMap, SwishFeatureMap,
                                     T2RFeatureMap)
from fla.modules.layernorm import rms_norm_linear
from nha_fla.ops.nha import chunk_nha, fused_recurrent_nha
from fla.modules import RotaryEmbedding

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.nha_cache import NHACache
from math import ceil
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None


class NativeHybridAttention(nn.Module):

    def __init__(
        self,
        config: NHAConfig,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_slots: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 8,
        feature_map: str = 'swish',
        use_output_gate: bool = False,
        use_norm: bool = True,
        layer_idx: Optional[int] = None,
        scale: Optional[float] = 1.,
        **kwargs
    ) -> NativeHybridAttention:
        super().__init__()
        self.config = config
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.scale = scale

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots

        self.window_size = config.window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.register_module('feature_map', None)
        if feature_map == 'swish':
            self.feature_map = SwishFeatureMap()
        elif feature_map == 'relu':
            self.feature_map = ReLUFeatureMap()
        elif feature_map == 't2r':
            self.feature_map = T2RFeatureMap(self.head_k_dim, self.head_k_dim)
        else:
            raise NotImplementedError(f"Feature map `{feature_map}` is not supported now.")

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False)
        self.f_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        self.g_norm = RMSNorm(self.hidden_size, elementwise_affine, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=self.rope_theta)

        self.learnable_init_slots = False
        if self.learnable_init_slots:
            # TODO: parameter effient, maybe only add learnable hidden state and generate k, v, f
            self.init_slot_k = nn.Parameter(torch.empty(self.window_size, self.num_heads, self.head_k_dim))
            self.init_slot_v = nn.Parameter(torch.empty(self.window_size, self.num_heads, self.head_v_dim))
            self.init_slot_f = nn.Parameter(torch.empty(self.window_size, self.num_heads, self.num_slots))
            nn.init.constant_(self.init_slot_k, 0)
            nn.init.constant_(self.init_slot_v, 0)
            nn.init.constant_(self.init_slot_f, 0)

        if config.block_size is not None and config.transformer_idx is not None:
            if self.layer_idx % config.block_size == config.transformer_idx:
                self.window_size = 2048
            else:
                self.window_size = config.window_size
        

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[NHACache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[NHACache]]:

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache,
                                            seq_idx=position_ids)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        f = self.f_proj(hidden_states)

        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_k_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        f = rearrange(f, 'b t (h m) -> b t h m', m=self.num_slots)

        # if self.feature_map is not None:
        #     q, k = map(lambda x: self.feature_map(x), (q, k))
        v = F.silu(v)

        f = F.logsigmoid(f) / self.gate_logit_normalizer
        s = (1 - f.exp()).to(f.dtype)
        
        seqlen_offset, max_seqlen = 0, q.shape[1]

        # dealing with left-padding
        if attention_mask is not None:
            s = s.mul_(attention_mask[:, -s.shape[1]:, None, None])
            v = v.mul_(attention_mask[:, -v.shape[1]:, None, None])

        if past_key_values is not None:
            # prepare gsa input
            last_k, last_v, last_f = past_key_values.get_pop_kvf(self.layer_idx, self.window_size)
            
            if last_k is None:
                if self.learnable_init_slots and seqlen_offset[0] != 0:
                    last_k = self.init_slot_k[None, seqlen_offset:seqlen_offset+q.shape[1], ...]
                    last_v = self.init_slot_v[None, seqlen_offset:seqlen_offset+q.shape[1], ...]
                    last_f = self.init_slot_f[None, seqlen_offset:seqlen_offset+q.shape[1], ...]
                    last_s = (1 - last_f.exp()).to(last_f.dtype)
                else:
                    b, _, h, d_k, d_v = *q.shape, v.shape[-1]
                    last_k, last_v, last_f = torch.zeros((b, 1, h, d_k), dtype=k.dtype, device=k.device), \
                                                torch.zeros((b, 1, h, d_v), dtype=v.dtype, device=v.device), \
                                                torch.zeros((b, 1, h, self.num_slots), dtype=f.dtype, device=f.device)
                    last_s = last_f
            
            else:
                last_k = rearrange(last_k, '... (h d) -> ... h d', h=self.num_heads)
                last_v = rearrange(last_v, '... (h d) -> ... h d', h=self.num_heads)
                last_f = rearrange(last_f, '... (h d) -> ... h d', h=self.num_heads)
                last_s = (1 - last_f.exp()).to(last_f.dtype)

            # update swa cache
            k_cached, v_cached, f_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1), f.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=0,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

        cu_seqlens = kwargs.get('cu_seqlens', None)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            pad_len = max(max_seqlen, 2048)
            offset_int = seqlen_offset
            q_offset = seqlen_offset
            pad_q, pad_k = torch.zeros((q.shape[0], pad_len, q.shape[2], q.shape[3]), dtype=q.dtype, device=q.device), torch.zeros((k.shape[0], pad_len, k.shape[2], k.shape[3]), dtype=k.dtype, device=k.device)
            pad_q[:, q_offset:q_offset+1, :, :] = q
            pad_k[:, offset_int-k.shape[1]+1:offset_int+1, :, :] = k
            pad_q, pad_k = self.rotary(pad_q.clone(), pad_k.clone(), seqlen_offset=0, max_seqlen=8192, cu_seqlens=cu_seqlens)
            rotary_q = pad_q[:, q_offset:q_offset+1, :, :].clone()
            rotary_k = pad_k[:, offset_int-k.shape[1]+1:offset_int+1, :, :].clone()
            if self.window_size <= 64:
                sliding_window = self.naive_swa(rotary_q, rotary_k, self.window_size)
                shift_k, shift_v, shift_f, shift_s = last_k, last_v, last_f, last_s
                
                o, sliding_window_prob, recurrent_state = fused_recurrent_nha(
                    q=q,
                    k=shift_k,
                    v=shift_v,
                    sliding_window=sliding_window,
                    s=shift_s,
                    g=shift_f,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=None,
                    cu_seqlens=cu_seqlens,
                    head_first=False,
                )
                o += torch.einsum('bthw,bwhd->bthd',
                                    sliding_window_prob.to(v.dtype),
                                    v)
            else:
                prefix_k = torch.zeros(k.size(0), self.num_slots, k.size(2), k.size(3), dtype=k.dtype, device=k.device)
                prefix_v = torch.zeros(v.size(0), self.num_slots, v.size(2), v.size(3), dtype=v.dtype, device=v.device)
                shift_k = torch.cat([prefix_k, rotary_k], dim=1)
                shift_v = torch.cat([prefix_v, v], dim=1)
                sliding_window = self.naive_swa(rotary_q, shift_k, self.window_size)
                sliding_window_prob = sliding_window.softmax(-1)
                o = torch.einsum('bthw,bwhd->bthd',
                                sliding_window_prob,
                                shift_v)
        elif mode == 'chunk':
            if self.window_size <= 64:

                rotary_q, rotary_k = self.rotary(q, k, seqlen_offset=0, max_seqlen=8192, cu_seqlens=cu_seqlens)

                prefix_k = torch.zeros(k.size(0), self.window_size, k.size(2), k.size(3), dtype=k.dtype, device=k.device)
                prefix_v = torch.zeros(v.size(0), self.window_size, v.size(2), v.size(3), dtype=v.dtype, device=v.device)
                prefix_s = torch.zeros(s.size(0), self.window_size, s.size(2), s.size(3), dtype=s.dtype, device=s.device)
                prefix_f = torch.zeros(f.size(0), self.window_size, f.size(2), f.size(3), dtype=f.dtype, device=f.device)

                shift_k = torch.cat([prefix_k, k], dim=1)
                shift_v = torch.cat([prefix_v, v], dim=1)
                shift_s = torch.cat([prefix_s, s], dim=1)
                shift_f = torch.cat([prefix_f, f], dim=1)

                rotary_k = torch.cat([prefix_k, rotary_k], dim=1)

                o, recurrent_state = chunk_nha(
                    q=q,
                    k=shift_k,
                    v=shift_v,
                    rotary_q=rotary_q,
                    rotary_k=rotary_k,
                    window_size=self.window_size,
                    s=shift_s,
                    g=shift_f,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=None,
                    cu_seqlens=cu_seqlens,
                    head_first=False,
                    rotary=self.rotary,
                )
            else:
                rotary_q, rotary_k = self.rotary(q, k, seqlen_offset=0, max_seqlen=8192, cu_seqlens=cu_seqlens)
                prefix_k = torch.zeros(k.size(0), self.num_slots, k.size(2), k.size(3), dtype=k.dtype, device=k.device)
                prefix_v = torch.zeros(v.size(0), self.num_slots, v.size(2), v.size(3), dtype=v.dtype, device=v.device)
                shift_q = torch.cat([prefix_k, rotary_q], dim=1)
                shift_k = torch.cat([prefix_k, rotary_k], dim=1)
                shift_v = torch.cat([prefix_v, v], dim=1)

                sliding_window = self.naive_swa(shift_q, shift_k, self.window_size)
                sliding_window_prob = sliding_window.softmax(-1)
                o = torch.einsum('bthw,bwhd->bthd',
                                sliding_window_prob,
                                shift_v)
                o = o[:, self.num_slots:, :, :]
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = rms_norm_linear(F.silu(o), self.g_norm.weight, self.g_norm.bias, self.o_proj.weight, self.o_proj.bias)

        return o, None, past_key_values

    def naive_swa(self, q: torch.Tensor, k: torch.Tensor, W: int):
        seq_len = q.shape[1]
        i = torch.arange(seq_len, device=q.device).view(-1, 1)  # (T, 1)
        j = torch.arange(seq_len, device=q.device).view(1, -1)  # (1, T)
        
        left_bound = torch.clamp(i - W + 1, min=0)          # (T, 1)
        
        valid_mask = (j >= left_bound) & (j <= i)                    # (T, T)

        scale_factor = 1 / (q.shape[-1] ** 0.5)
        qk = torch.einsum('bthd,bnhd->bhtn', q, k) * scale_factor

        qk = qk.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        return qk.transpose(1, 2)