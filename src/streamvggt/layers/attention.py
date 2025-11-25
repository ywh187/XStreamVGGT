import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

from einops import rearrange

XFORMERS_AVAILABLE = False
if os.environ.get("USE_XFORMERS", "1") == "1":
    try:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (not used)")
    except ImportError:
        warnings.warn("xFormers is not available")


# from .nha_cache import NHACache, StreamNHACache
# from nha_fla.ops.nha import chunk_nha, fused_recurrent_nha


# class NativeHybridAttention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = True,
#         proj_bias: bool = True,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         norm_layer: nn.Module = nn.LayerNorm,
#         qk_norm: bool = False,
#         # NHA specific args
#         window_size: int = 708,
#         num_slots: int = 4,
#         rope=None,
#         layer_idx: Optional[int] = None,
#         fused_attn: bool = True,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim**-0.5
#         self.layer_idx = layer_idx

#         # NHA specific params
#         self.num_slots = num_slots
#         self.window_size = window_size

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.g_proj = nn.Linear(dim, self.num_heads * self.num_slots, bias=False)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.rope = rope

#     def forward(self, 
#         x: torch.Tensor, 
#         pos=None, 
#         attn_mask=None, 
#         past_key_value: Optional[NHACache] = None, 
#         use_cache: bool = False,
#         **kwargs,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, NHACache]]:
#         import pdb; pdb.set_trace()

#         if use_cache and isinstance(past_key_value, StreamNHACache):
#             return self.stream_forward(x, pos, past_key_value)

#         B, N, C = x.shape
        
#         # QKV projection
#         qkv = self.qkv(x)
#         q, k, v = qkv.chunk(3, dim=-1)
#         g = self.g_proj(x)

#         # Reshape to multi-head format: [B, H, N, D]
#         q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
#         k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
#         v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
#         g = rearrange(g, 'b n (h m) -> b h n m', h=self.num_heads)

#         # Gate logic
#         gate_logit_normalizer = 8
#         g = F.logsigmoid(g) / gate_logit_normalizer  # [B, H, N, M]
#         s = 1 - torch.exp(g).to(g.dtype)  # [B, H, N, M]
        
#         # 移除错误的mask操作 - mask应该在attention score上应用,不是在s或v上 
#         # mask = attn_mask[-s.shape[2]:, -s.shape[2]:]
#         # s = s.mul_(mask.unsqueeze(0).unsqueeze(-1))  # 删除这行
#         # v = v.mul_(mask.unsqueeze(0).unsqueeze(-1))  # 删除这行

#         # Get recurrent state from cache if available
#         recurrent_state = None
#         if past_key_value is not None and len(past_key_value) > self.layer_idx:
#             recurrent_state = past_key_value[self.layer_idx].get('recurrent_state')

#         # Apply RoPE BEFORE transpose (RoPE expects [B, H, N, D])
#         if self.rope is not None:
#             if use_cache and q.shape[2] == 1:
#                 # Inference: apply RoPE on the sliding window part
#                 window_len = k.shape[2]
#                 swa_pos = pos - torch.arange(window_len - 1, -1, -1, device=pos.device).unsqueeze(0)
#                 sq, sk = self.rope(q, pos), self.rope(k, swa_pos)
#             else:
#                 # Training: apply RoPE to the whole sequence
#                 sq, sk = self.rope(q, pos), self.rope(k, pos) # torch.Size([1, 16, 7080, 64])
#         else:
#             sq, sk = q, k

#         # NOW transpose for NHA ops: [B, H, N, D] -> [B, N, H, D]
#         q, k, v, s, g = (x.transpose(1, 2).contiguous() for x in (q, k, v, s, g))
#         sq, sk = sq.transpose(1, 2).contiguous(), sk.transpose(1, 2).contiguous()

#         # Handle cache logic for inference
#         if use_cache:
#             # Update attention state (k, v, g) in the cache
#             past_key_value.update(
#                 attn_state=(k.flatten(-2, -1), v.flatten(-2, -1), g.flatten(-2, -1)),
#                 layer_idx=self.layer_idx,
#                 offset=0,
#                 cache_kwargs=dict(window_size=self.window_size)
#             )
#             # Get sliding window k,v from cache
#             cache_has_content = past_key_value.get_seq_length(self.layer_idx) > 0
#             if cache_has_content:
#                 k_cached, v_cached, _ = past_key_value[self.layer_idx]['attn_state']
#                 k = rearrange(k_cached, '... (h d) -> ... h d', h=self.num_heads)
#                 v = rearrange(v_cached, '... (h d) -> ... h d', h=self.num_heads)
#                 # Re-apply rope on cached keys (需要转回[B, H, N, D])
#                 k_for_rope = k.transpose(1, 2)
#                 sk_cached = self.rope(k_for_rope, pos)
#                 sk = sk_cached.transpose(1, 2)

#         # Main NHA logic
#         if self.training or not use_cache or q.shape[1] > 1:
#             # Training or chunked-inference path
#             rotary_q, rotary_k = sq, sk

#             # Add dummy prefixes for chunk_nha compatibility
#             prefix_k = torch.zeros(k.size(0), self.window_size, k.size(2), k.size(3), dtype=k.dtype, device=k.device)
#             prefix_v = torch.zeros(v.size(0), self.window_size, v.size(2), v.size(3), dtype=v.dtype, device=v.device)
#             prefix_s = torch.zeros(s.size(0), self.window_size, s.size(2), s.size(3), dtype=s.dtype, device=s.device)
#             prefix_g = torch.zeros(g.size(0), self.window_size, g.size(2), g.size(3), dtype=g.dtype, device=g.device)

#             shift_k, shift_v = torch.cat([prefix_k, k], dim=1), torch.cat([prefix_v, v], dim=1)
#             shift_s, shift_g = torch.cat([prefix_s, s], dim=1), torch.cat([prefix_g, g], dim=1)
#             rotary_k = torch.cat([prefix_k, rotary_k], dim=1)

#             o, recurrent_state = chunk_nha(
#                 q=q,
#                 k=shift_k, v=shift_v, s=shift_s, g=shift_g,
#                 rotary_q=rotary_q, rotary_k=rotary_k,
#                 window_size=self.window_size,
#                 initial_state=recurrent_state,
#                 output_final_state=use_cache,
#                 scale=None,
#                 head_first=False,
#                 rotary=None,
#             )
#         else:
#             # Single-token autoregressive inference path
#             sliding_window = self.naive_swa(sq, sk, self.window_size)
            
#             # Get the oldest k,v,g from cache for recurrent update
#             last_k, last_v, last_g = past_key_value.get_pop_kvf(self.layer_idx, self.window_size)
#             if last_k is None:
#                 b, _, h, d_k, d_v = *q.shape, v.shape[-1]
#                 last_k = torch.zeros((b, 1, h, d_k), dtype=k.dtype, device=k.device)
#                 last_v = torch.zeros((b, 1, h, d_v), dtype=v.dtype, device=v.device)
#                 last_g = torch.zeros((b, 1, h, self.num_slots), dtype=g.dtype, device=g.device)
#             else:
#                 last_k = rearrange(last_k, '... (h d) -> ... h d', h=self.num_heads)
#                 last_v = rearrange(last_v, '... (h d) -> ... h d', h=self.num_heads)
#                 last_g = rearrange(last_g, '... (h d) -> ... h d', h=self.num_heads)
#             last_s = (1 - last_g.exp()).to(last_g.dtype)

#             o, sliding_window_prob, recurrent_state = fused_recurrent_nha(
#                 q=q, k=last_k, v=last_v, s=last_s, g=last_g,
#                 sliding_window=sliding_window,
#                 initial_state=recurrent_state,
#                 output_final_state=use_cache,
#             )
#             # Add the local attention part
#             o += torch.einsum('bthw,bwhd->bthd', sliding_window_prob.to(v.dtype), v)

#         # Update recurrent state in cache
#         if use_cache:
#             past_key_value.update(
#                 recurrent_state=recurrent_state,
#                 layer_idx=self.layer_idx,
#                 offset=q.shape[1]
#             )

#         x = rearrange(o, 'b n h d -> b n (h d)')
#         x = self.proj(x)
#         x = self.proj_drop(x)
        
#         if use_cache:
#             return x, past_key_value
#         return x

#     def stream_forward(self, x: torch.Tensor, pos, past_key_value: StreamNHACache):
#         B, N, C = x.shape
#         qkv = self.qkv(x)
#         q, k, v = qkv.chunk(3, dim=-1)
#         g = self.g_proj(x)

#         # Reshape to [B, H, N, D]
#         q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
#         k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
#         v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
#         g = rearrange(g, 'b n (h m) -> b h n m', h=self.num_heads)

#         gate_logit_normalizer = 8
#         g = F.logsigmoid(g) / gate_logit_normalizer
#         s = 1 - torch.exp(g).to(g.dtype)

#         # Apply RoPE before transpose
#         if self.rope is not None:
#             sq, sk = self.rope(q, pos), self.rope(k, pos)
#         else:
#             sq, sk = q, k

#         # Transpose to [B, N, H, D]
#         q, k, v, s, g = (x.transpose(1, 2).contiguous() for x in (q, k, v, s, g))
#         sq, sk = sq.transpose(1, 2).contiguous(), sk.transpose(1, 2).contiguous()

#         recurrent_state = None
#         if len(past_key_value) > self.layer_idx:
#             recurrent_state = past_key_value[self.layer_idx].get('recurrent_state')

#         if past_key_value.cache_status[self.layer_idx] == 'EMPTY':
#             # First frame
#             o, recurrent_state = chunk_nha(
#                 q=q, k=k, v=v, s=s, g=g, rotary_q=sq, rotary_k=sk,
#                 window_size=N, initial_state=None, output_final_state=True
#             )
#             past_key_value.update(
#                 recurrent_state=recurrent_state,
#                 attn_state=(k, v, g),
#                 layer_idx=self.layer_idx,
#                 offset=N
#             )
#         elif past_key_value.cache_status[self.layer_idx] == 'STATIC_FILLED':
#             # Second frame
#             static_k, static_v, _ = past_key_value.get_kv('static', self.layer_idx)
            
#             # Attention with static tokens
#             o_static = self.naive_attention(sq, static_k, static_v)

#             # Self-attention for new tokens
#             o_new, recurrent_state = chunk_nha(
#                 q=q, k=k, v=v, s=s, g=g, rotary_q=sq, rotary_k=sk,
#                 window_size=N, initial_state=recurrent_state, output_final_state=True
#             )
#             o = o_static + o_new
#             past_key_value.update(
#                 recurrent_state=recurrent_state,
#                 attn_state=(k, v, g),
#                 layer_idx=self.layer_idx,
#                 offset=N
#             )
#         elif past_key_value.cache_status[self.layer_idx] == 'SLOT_FILLED':
#             # Streaming case
#             static_k, static_v, _ = past_key_value.get_kv('static', self.layer_idx)
#             slot_k, slot_v, slot_g = past_key_value.get_kv('slot', self.layer_idx)
            
#             # Attention with static tokens
#             o_static = self.naive_attention(sq, static_k, static_v)

#             # Recalculate s for slot
#             slot_s = 1 - torch.exp(slot_g).to(slot_g.dtype)

#             combined_k = torch.cat([slot_k, k], dim=1)
#             combined_v = torch.cat([slot_v, v], dim=1)
#             combined_s = torch.cat([slot_s, s], dim=1)
#             combined_g = torch.cat([slot_g, g], dim=1)
            
#             # RoPE for combined sequence
#             slot_pos = torch.arange(slot_k.shape[1], device=q.device).unsqueeze(0)
#             new_pos = torch.arange(k.shape[1], device=q.device).unsqueeze(0) + slot_k.shape[1]
            
#             # Apply RoPE (need to transpose back)
#             slot_k_for_rope = slot_k.transpose(1, 2)
#             slot_sk_rope = self.rope(slot_k_for_rope, slot_pos).transpose(1, 2)
            
#             combined_sk = torch.cat([slot_sk_rope, sk], dim=1)

#             o_stream, recurrent_state = chunk_nha(
#                 q=q, k=combined_k, v=combined_v, s=combined_s, g=combined_g,
#                 rotary_q=sq, rotary_k=combined_sk,
#                 window_size=N * 2, initial_state=recurrent_state, output_final_state=True
#             )

#             o = o_static + o_stream

#             # Update slot
#             updated_k = combined_k[:, -N:, ...]
#             updated_v = combined_v[:, -N:, ...]
#             updated_g = combined_g[:, -N:, ...]

#             past_key_value.update(
#                 recurrent_state=recurrent_state,
#                 attn_state=(updated_k, updated_v, updated_g),
#                 layer_idx=self.layer_idx,
#                 offset=N
#             )

#         x = rearrange(o, 'b n h d -> b n (h d)')
#         x = self.proj(x)
#         x = self.proj_drop(x)
        
#         return x, past_key_value

#     def naive_attention(self, q, k, v):
#         """Simple scaled dot product attention. q,k,v: [B, N, H, D]"""
#         attn = torch.einsum('bnhd,bmhd->bhnm', q, k) * self.scale
#         attn = attn.softmax(dim=-1)
#         o = torch.einsum('bhnm,bmhd->bnhd', attn, v)
#         return o

#     def naive_swa(self, q: torch.Tensor, k: torch.Tensor, W: int):
#         """Sliding window attention. q,k: [B, N, H, D]"""
#         q_len, k_len = q.shape[1], k.shape[1]
        
#         i = torch.arange(q_len, device=q.device).view(-1, 1)
#         j = torch.arange(k_len, device=q.device).view(1, -1)

#         offset = k_len - q_len
#         i_ = i + offset

#         left_bound = torch.clamp(i_ - W + 1, min=0)
#         valid_mask = (j >= left_bound) & (j <= i_)

#         qk = torch.einsum('bthd,bnhd->bhtn', q, k) * self.scale
#         qk = qk.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), -torch.finfo(qk.dtype).max)
#         return qk.transpose(1, 2)




class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        layer_idx=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.layer_idx = layer_idx

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, 
        x: torch.Tensor, 
        pos=None, 
        attn_mask=None, 
        past_key_values=None, 
        use_cache=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        pos_k = pos
        if use_cache:
            # import pdb; pdb.set_trace()
            k = k.unsqueeze(2) # torch.Size([1, 16, 1, 1041, 64])
            v = v.unsqueeze(2)
            if past_key_values is not None:
                past_k, past_v = past_key_values
                k = torch.cat([past_k, k], dim=2) # torch.Size([1, 16, 2, 1041, 64])
                v = torch.cat([past_v, v], dim=2)
                
            new_kv = (k, v)
            a, b, c, d, e = k.shape
            k = k.reshape(a, b, c*d, e) # torch.Size([1, 16, 2082, 64]) 这里才展开
            v = v.reshape(a, b, c*d, e) # torch.Size([1, 16, 2082, 64])
            if pos_k is not None:
                #print(pos_k.shape)
                pos_k = pos_k.repeat(1, c, 1) # 第一帧 torch.Size([1, 1041, 2])  第二帧torch.Size([1, 2082, 2])
                #print(pos_k.shape)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            # import pdb; pdb.set_trace()
            q = self.rope(q, pos)   # torch.Size([1, 16, 1041, 64]) torch.Size([1, 1041, 2])
            k = self.rope(k, pos_k) # torch.Size([1, 16, 2082, 64])  位置编码帧与帧之间是一样的，可以做成emb完再存起来

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if use_cache:
            return x, new_kv
        return x




# class Attention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = True,
#         proj_bias: bool = True,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         norm_layer: nn.Module = nn.LayerNorm,
#         qk_norm: bool = False,
#         fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
#         rope=None,
#         layer_idx=None, # for compatibility
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim**-0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.fused_attn = fused_attn
#         self.rope = rope

#         if qk_norm:
#             self.q_norm = norm_layer(self.head_dim)
#             self.k_norm = norm_layer(self.head_dim)
#         else:
#             self.q_norm = nn.Identity()
#             self.k_norm = nn.Identity()

#     def forward(
#         self, x: Tensor, attn_mask=None, past_key_value=None, use_cache=False, pos=None, **kwargs
#     ) -> Union[Tensor, Tuple[Tensor, Dict]]:
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)

#         q, k = self.q_norm(q), self.k_norm(k)

#         if self.rope is not None:
#             q = self.rope(q, pos)
#             k = self.rope(k, pos)

#         if use_cache:
#             if past_key_value is None:
#                 past_key_value = {"k": k, "v": v}
#             else:
#                 k = torch.cat([past_key_value["k"], k], dim=2)
#                 v = torch.cat([past_key_value["v"], v], dim=2)
#                 past_key_value = {"k": k, "v": v}

#         if self.fused_attn:
#             x = F.scaled_dot_product_attention(
#                 q, k, v,
#                 attn_mask=attn_mask,
#                 dropout_p=self.attn_drop.p if self.training else 0.0,
#             )
#         else:
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             if attn_mask is not None:
#                 attn = attn + attn_mask
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v

#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         if use_cache:
#             return x, past_key_value
#         return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x