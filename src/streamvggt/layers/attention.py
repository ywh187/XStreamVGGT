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
        cache_size: int = 2048,  # KV cache 最大长度 
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
        self.cache_size = os.getenv("KV_CACHE_SIZE", cache_size) # cache的上限长度，默认2k
        self.pool_size = os.getenv("KV_POOL_SIZE", 16) # 计算重要性时，对query进行pooling的窗口大小，默认16 
        

    
    def prune_kv_cache(self, kv, q_new):
        # import pdb; pdb.set_trace()
        mode = self.prune_mode
        if mode == "FastVGGT":
            return self._prune_fast_vggt(kv, q_new)
        elif mode == "SlidingWindow":
            # print(f"[Prune] SlidingWindow mode")
            return self._prune_sliding_window(kv)
        elif mode == "Random":
            return self._prune_random(kv)
        else:
            raise ValueError(f"Unknown KV prune mode: {mode}")

    def _prune_fast_vggt(self, kv, q_new):
        k, v = kv
        B, H, T, D = k.shape
        N_new = q_new.shape[2]

        if T <= self.cache_size:
            return kv

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

        q_light = torch.cat([special_tokens, normal_tokens_pooled], dim=2)
        q_score = q_light.mean(dim=1)

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
            new_kv = (k, v)
        else:
            new_kv = None

        # ---- Attention ----
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q_scaled = q * self.scale
            attn = q_scaled @ k.transpose(-2,-1)
            if attn_mask is not None:
                attn += attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

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

            new_kv = self.prune_kv_cache(new_kv, q)

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
