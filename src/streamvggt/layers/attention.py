import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional
import time
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
        fused_attn: bool = True,
        rope=None,
        layer_idx=None,
        cache_size: int = 8192,  # KV cache 最大长度 
        # window_keep: int = 0,   # 保留最新 token 个数（不剪掉）
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.fused_attn = fused_attn
        self.layer_idx = layer_idx
        self.rope = rope


        # 剪枝参数
        self.prune_mode = os.getenv("KV_PRUNE_MODE", "FastVGGT") # FastVGGT / SlidingWindow / Random
        self.cache_size = int(os.getenv("KV_CACHE_SIZE", cache_size)) # cache的上限长度，默认2k
        self.pool_size = int(os.getenv("KV_POOL_SIZE", 16)) # 计算重要性时，对query进行pooling的窗口大小，默认16 
        


    def prune_kv_cache(self, kv, q_new): #, token_num_eachframe=0):
        # import pdb; pdb.set_trace()
        mode = self.prune_mode
        if mode == "FastVGGT":
            # return self._prune_fast_vggt(kv, q_new, token_num_eachframe=token_num_eachframe)
            return self._prune_fast_vggt_protected(kv, q_new) #, token_num_eachframe=token_num_eachframe)
        elif mode == "SlidingWindow":
            # print(f"[Prune] SlidingWindow mode")
            return self._prune_sliding_window(kv)
        elif mode == "Random":
            return self._prune_random(kv)
        else: # 不剪枝
            return kv

    def _prune_fast_vggt_protected(self, kv, q_new): #, token_num_eachframe=0):
        k, v = kv
        B, H, T, D = k.shape
        N_new = q_new.shape[2]

        if T <= self.cache_size:
            return kv

        # 保护前1024个token
        protected_size = N_new # 一帧的token数量
        
        # window_keep = 新 q 的长度
        window_keep = N_new
        
        # 计算需要从中间区域保留多少token
        # total_keep = protected (1024) + middle_keep + window_keep
        total_keep = self.cache_size
        middle_keep = max(0, total_keep - protected_size - window_keep)
        
        # 可以被剪枝的中间区域: [protected_size : T-window_keep]
        prunable_start = protected_size
        prunable_end = T - window_keep
        prunable_length = max(0, prunable_end - prunable_start)
        
        if prunable_length == 0:
            # 没有可剪枝的区域，直接保留所有
            return kv
        
        # ----- Query pooling 逻辑 -----
        special_tokens = q_new[:, :, :5, :]
        normal_tokens = q_new[:, :, 5:, :]

        if normal_tokens.shape[2] > 1:
            pool_size = int(self.pool_size)
            n = normal_tokens.shape[2]
            n_pool = n // pool_size
            pooled = normal_tokens[:, :, :n_pool*pool_size, :].reshape(
                B, H, n_pool, pool_size, D
            ).mean(dim=3)

            remainder = normal_tokens[:, :, n_pool*pool_size:, :]
            if remainder.shape[2] > 0:
                pooled = torch.cat([pooled, remainder.mean(dim=2, keepdim=True)], dim=2)

            normal_tokens_pooled = pooled
        else:
            normal_tokens_pooled = normal_tokens

        q_light = torch.cat([special_tokens, normal_tokens_pooled], dim=2)
        q_score = q_light.mean(dim=1)

        # 只对可剪枝区域计算分数
        k_prunable = k[:, :, prunable_start:prunable_end, :]
        k_score = k_prunable.mean(dim=1)
        scores = (q_score @ k_score.transpose(-2, -1)).mean(dim=1)

        # 从可剪枝区域选择top-k
        topk_count = min(middle_keep, scores.shape[1])
        if topk_count > 0:
            topk_idx_relative = torch.topk(scores, topk_count, dim=-1).indices
            # 转换为绝对索引
            topk_idx = topk_idx_relative + prunable_start
        else:
            topk_idx = torch.empty(B, 0, dtype=torch.long, device=k.device)

        # 构建最终的keep索引: protected + middle_topk + window
        protected_idx = torch.arange(0, protected_size, device=k.device).unsqueeze(0).expand(B, -1)
        window_idx = torch.arange(T-window_keep, T, device=k.device).unsqueeze(0).expand(B, -1)

        keep_idx = torch.cat([protected_idx, topk_idx, window_idx], dim=-1)
        keep_idx = torch.sort(keep_idx, dim=-1).values

        # Gather操作
        keep_idx_expanded = keep_idx.unsqueeze(1).unsqueeze(-1).expand(B, H, -1, D)
        new_k = torch.gather(k, 2, keep_idx_expanded)
        new_v = torch.gather(v, 2, keep_idx_expanded)

        return new_k, new_v


    def _prune_fast_vggt(self, kv, q_new): #, token_num_eachframe=0):
        k, v = kv
        B, H, T, D = k.shape
        N_new = q_new.shape[2]

        if T <= self.cache_size:
            return kv

        # import pdb; pdb.set_trace()

        # window_keep = 新 q 的长度
        window_keep = N_new
        num_keep = max(0, self.cache_size - window_keep)

        # ----- 你的 Query pooling 逻辑 -----
        special_tokens = q_new[:, :, :5, :]
        normal_tokens = q_new[:, :, 5:, :]

        if normal_tokens.shape[2] > 1:
            pool_size = int(self.pool_size)
            n = normal_tokens.shape[2]
            n_pool = n // pool_size
            pooled = normal_tokens[:, :, :n_pool*pool_size, :].reshape(
                B, H, n_pool, pool_size, D
            ).mean(dim=3)

            remainder = normal_tokens[:, :, n_pool*pool_size:, :]
            if remainder.shape[2] > 0:
                pooled = torch.cat([pooled, remainder.mean(dim=2, keepdim=True)], dim=2)

            normal_tokens_pooled = pooled
        else:
            normal_tokens_pooled = normal_tokens

        # import pdb; pdb.set_trace()
        q_light = torch.cat([special_tokens, normal_tokens_pooled], dim=2) # torch.Size([1, 16, 5, 64]) torch.Size([1, 16, 35, 64]) -> torch.Size([1, 16, 40, 64])
        q_score = q_light.mean(dim=1) # torch.Size([1, 40, 64])

        k_score = k[:, :, :T-window_keep, :].mean(dim=1)
        scores = (q_score @ k_score.transpose(-2, -1)).mean(dim=1)

        topk_idx = torch.topk(scores, min(num_keep, scores.shape[1]), dim=-1).indices
        window_idx = torch.arange(T-window_keep, T, device=k.device).unsqueeze(0).expand(B, -1)

        keep_idx = torch.cat([topk_idx, window_idx], dim=-1)
        keep_idx = torch.sort(keep_idx, dim=-1).values

        keep_idx_expanded = keep_idx.unsqueeze(1).unsqueeze(-1).expand(B, H, -1, D)
        new_k = torch.gather(k, 2, keep_idx_expanded)
        new_v = torch.gather(v, 2, keep_idx_expanded)

        return new_k, new_v
    
    def _prune_sliding_window(self, kv):
        k, v = kv
        B, H, T, D = k.shape

        if T <= self.cache_size:
            return kv

        # 只保留 T - cache_size → T
        new_k = k[:, :, -self.cache_size:, :]
        new_v = v[:, :, -self.cache_size:, :]
        return new_k, new_v

    def _prune_random(self, kv):
        k, v = kv
        B, H, T, D = k.shape

        if T <= self.cache_size:
            return kv

        # 随机选 cache_size 个 index
        idx = torch.randperm(T, device=k.device)[:self.cache_size]
        idx = torch.sort(idx).values           # 保序，使 sequence 不乱
        idx = idx.unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(B, H, -1, D)

        new_k = torch.gather(k, 2, idx)
        new_v = torch.gather(v, 2, idx)
        return new_k, new_v


    # def prune_kv_cache(self, kv: Tuple[torch.Tensor, torch.Tensor], q_new: torch.Tensor):
    #     k, v = kv
    #     B, H, T, D = k.shape
    #     N_new = q_new.shape[2]
        
    #     # 边界
    #     if T <= self.cache_size:
    #         return kv
    

    #     # 确保 window_keep 不超过 cache_size
    #     window_keep = N_new # 默认用q的长度
    #     num_keep = max(0, self.cache_size - window_keep)
        
    #     # Query 降采样
    #     special_tokens = q_new[:, :, :5, :]
    #     normal_tokens = q_new[:, :, 5:, :]
        
    #     if normal_tokens.shape[2] > 1:
    #         pool_size = 16
    #         n = normal_tokens.shape[2]
    #         n_pool = n // pool_size
    #         pooled = normal_tokens[:, :, :n_pool*pool_size, :].reshape(B, H, n_pool, pool_size, D).mean(dim=3)
            
    #         # 多余的
    #         remainder = normal_tokens[:, :, n_pool*pool_size:, :]
    #         if remainder.shape[2] > 0:
    #             remainder_pooled = remainder.mean(dim=2, keepdim=True)
    #             pooled = torch.cat([pooled, remainder_pooled], dim=2)
            
    #         normal_tokens_pooled = pooled
    #     else:
    #         normal_tokens_pooled = normal_tokens
        
    #     q_light = torch.cat([special_tokens, normal_tokens_pooled], dim=2)
        
    #     # 计算重要性（只针对历史 token）
    #     q_score = q_light.mean(dim=1)  # [B, N', D]
    #     k_score = k[:, :, :T-window_keep, :].mean(dim=1)  # [B, T-window_keep, D]
    #     scores = (q_score @ k_score.transpose(-2, -1)).mean(dim=1)  # [B, T-window_keep]
        
    #     # Top-K 选择
    #     topk_idx = torch.topk(scores, min(num_keep, scores.shape[1]), dim=-1).indices
        
    #     # 窗口索引
    #     window_idx = torch.arange(T-window_keep, T, device=k.device).unsqueeze(0).expand(B, -1)
        
    #     # 合并并排序（关键修复）
    #     keep_idx = torch.cat([topk_idx, window_idx], dim=-1)
    #     keep_idx = torch.sort(keep_idx, dim=-1).values
        
    #     # 更新
    #     # batch_idx = torch.arange(B, device=k.device).unsqueeze(1).expand(-1, keep_idx.shape[1])
    #     # new_k = k[batch_idx, :, keep_idx]
    #     # new_v = v[batch_idx, :, keep_idx]
    #     keep_idx_expanded = keep_idx.unsqueeze(1).unsqueeze(-1).expand(B, H, -1, D)
    #     new_k = torch.gather(k, dim=2, index=keep_idx_expanded)  # [B, H, num_keep, D]
    #     new_v = torch.gather(v, dim=2, index=keep_idx_expanded)
        
    #     return new_k, new_v


    def remove_special_tokens_from_attn(attn, tokens_per_frame=1041, special_tokens=5):
        """
        从attention map中移除特殊token，保留纯patch的attention关系
        
        Args:
            attn: attention map, shape (q_len, k_len)
            tokens_per_frame: 每帧的token数量（包括特殊token）
            special_tokens: 每帧开头的特殊token数量
        
        Returns:
            clean_attn: 移除特殊token后的attention map
        """
        q_len, k_len = attn.shape
        
        # 计算帧数
        num_frames_q = q_len // tokens_per_frame
        num_frames_k = k_len // tokens_per_frame
        
        # 验证维度是否正确
        assert q_len == num_frames_q * tokens_per_frame, f"q_len {q_len} 不能被 tokens_per_frame {tokens_per_frame} 整除"
        assert k_len == num_frames_k * tokens_per_frame, f"k_len {k_len} 不能被 tokens_per_frame {tokens_per_frame} 整除"
        
        # 计算每帧的patch数量
        patches_per_frame = tokens_per_frame - special_tokens
        
        # 创建mask来标记要保留的位置
        q_mask = torch.zeros(q_len, dtype=torch.bool)
        k_mask = torch.zeros(k_len, dtype=torch.bool)
        
        # 标记每帧中patch的位置（跳过前special_tokens个）
        for i in range(num_frames_q):
            start = i * tokens_per_frame + special_tokens
            end = (i + 1) * tokens_per_frame
            q_mask[start:end] = True
        
        for i in range(num_frames_k):
            start = i * tokens_per_frame + special_tokens
            end = (i + 1) * tokens_per_frame
            k_mask[start:end] = True
        
        # 使用mask提取干净的attention map
        clean_attn = attn[q_mask][:, k_mask]
        
        return clean_attn

    def forward(
        self,
        x,
        pos=None,
        attn_mask=None,
        past_key_values=None,   # (past_k, past_v)
        use_cache=False,
        **kwargs
    ):
        B, N, C = x.shape

        # import pdb; pdb.set_trace()

        # ========= 计时：Attention 耗时 =========
        # torch.cuda.synchronize()
        # t0 = time.time()
        # =======================================

        # ---- QKV projection ----
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)  # [B,H,N,D]

        # ---- Norm ----
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ---- RoPE ----
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # ---- KV cache update ----
        if use_cache:
            if past_key_values is not None:
                past_k, past_v = past_key_values  # [B,H,T,D]
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
                # import pdb; pdb.set_trace()
            new_kv = (k, v)
        else:
            new_kv = None
        

        # ---- Attention ----
        # if k.shape[2] > 1100:
        #     import pdb; pdb.set_trace()
        #     q_scaled = q * self.scale # torch.Size([1, 16, 1041, 64])
        #     attn = q_scaled @ k.transpose(-2,-1)
        #     if attn_mask is not None:
        #         attn += attn_mask
        #     attn = attn.softmax(dim=-1) # torch.Size([1, 16, 1041, 1041])
        #     attn = self.attn_drop(attn)
        #     x = attn @ v
        #     # 画图
        #     attn_map_vis = attn[0,:,5:,5:].max(dim=0)[0]  # torch.Size([1036, 1036])
        #     attn_map_vis = self.remove_special_tokens_from_attn(attn_map_vis, tokens_per_frame=q.shape[2], special_tokens=5)
        # el
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, # torch.Size([1, 16, 1041, 64])
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )


        x = x.transpose(1,2).reshape(B,N,C)
        x = self.proj_drop(self.proj(x))

        # ========= 计时结束：Attention 耗时 =========
        # torch.cuda.synchronize()
        # t1 = time.time()
        # attn_time = t1 - t0
        # print(f"[Time] Attention: {attn_time*1000:.3f} ms")
        # ===========================================


        # ---- attention结束后做 KV cache pruning ---- 
        if use_cache:

            # # ========= 计时：cache prune 耗时 =========
            # torch.cuda.synchronize()
            # t2 = time.time()
            # # ===========================================

            new_kv = self.prune_kv_cache(new_kv, q) #, token_num_eachframe=kwargs.get("token_num_eachframe", 0))

            # torch.cuda.synchronize()
            # t3 = time.time()
            # prune_time = t3 - t2
            # print(f"[Time] Cache Prune: {prune_time*1000:.3f} ms")

            return x, new_kv

        return x


# # baseline原版attention
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
#         layer_idx=None,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim**-0.5
#         self.fused_attn = fused_attn
#         self.layer_idx = layer_idx

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.rope = rope

#     def forward(self, 
#         x: torch.Tensor, 
#         pos=None, 
#         attn_mask=None, 
#         past_key_values=None, 
#         use_cache=False
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)

#         pos_k = pos
#         if use_cache:
#             # import pdb; pdb.set_trace()
#             k = k.unsqueeze(2) # torch.Size([1, 16, 1, 1041, 64])
#             v = v.unsqueeze(2)
#             if past_key_values is not None:
#                 #import pdb; pdb.set_trace()
#                 past_k, past_v = past_key_values # torch.Size([1, 16, 1, 1, 128])
#                 k = torch.cat([past_k, k], dim=2) # torch.Size([1, 16, 2, 1041, 64])
#                 v = torch.cat([past_v, v], dim=2)
                
#             new_kv = (k, v)
#             a, b, c, d, e = k.shape
#             k = k.reshape(a, b, c*d, e) # torch.Size([1, 16, 2082, 64]) 这里才展开
#             v = v.reshape(a, b, c*d, e) # torch.Size([1, 16, 2082, 64])
#             if pos_k is not None:
#                 #print(pos_k.shape)
#                 pos_k = pos_k.repeat(1, c, 1) # 第一帧 torch.Size([1, 1041, 2])  第二帧torch.Size([1, 2082, 2])
#                 #print(pos_k.shape)

#         q, k = self.q_norm(q), self.k_norm(k)

#         if self.rope is not None:
#             # import pdb; pdb.set_trace()
#             q = self.rope(q, pos)   # torch.Size([1, 16, 1041, 64]) torch.Size([1, 1041, 2])
#             k = self.rope(k, pos_k) # torch.Size([1, 16, 2082, 64])  位置编码帧与帧之间是一样的，可以做成emb完再存起来

#         if self.fused_attn:
#             x = F.scaled_dot_product_attention(
#                 q,
#                 k,
#                 v,
#                 attn_mask=attn_mask,
#                 dropout_p=self.attn_drop.p if self.training else 0.0,
#             )

#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)

#             # Mask
#             if attn_mask is not None:
#                 assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
#                 attn = attn + attn_mask

#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v

#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         if use_cache:
#             return x, new_kv
#         return x




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
