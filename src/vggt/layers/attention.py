# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from typing import Union, Tuple, Dict, Optional

from einops import rearrange

XFORMERS_AVAILABLE = False


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
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(
        self, 
        x: torch.Tensor, 
        pos=None, 
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# class MemEffAttention(Attention):
#     """
#     Linear Attention (memory-efficient) version of the original Attention.
#     Uses feature maps φ(q), φ(k) to approximate softmax(qk^T)v in O(N) time.
#     """
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
#         eps: float = 1e-6,  # numerical stability
#     ):
#         super().__init__(
#             dim=dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             proj_bias=proj_bias,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#             qk_norm=qk_norm,
#             fused_attn=False,   # 禁止使用原生softmax注意力
#             rope=rope,
#         )
#         self.eps = eps

#     @staticmethod
#     def feature_map(x: torch.Tensor) -> torch.Tensor:
#         """ φ(x) = elu(x) + 1 — 常见的正值特征映射，避免负数 """
#         return F.elu(x) + 1

#     def forward(
#         self,
#         x: torch.Tensor,
#         attn_bias=None,
#         pos=None,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:

#         B, N, C = x.shape # (10, 708, 1024)

#         # 1. 生成 Q, K, V
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
#         q, k, v = qkv.unbind(0)

#         q, k = self.q_norm(q), self.k_norm(k)

#         # 2. 位置编码 (Rope)
#         if self.rope is not None:
#             q = self.rope(q, pos)
#             k = self.rope(k, pos)

#         # 3. 特征映射 φ(q), φ(k)
#         q_phi = self.feature_map(q)   # [B, H, N, D]
#         k_phi = self.feature_map(k)

#         # 4. 计算 KV = Σ φ(k) * v
#         kv = torch.einsum('b h n d, b h n e -> b h d e', k_phi, v)  # [B, H, D, D]

#         # 5. 计算分母 normalizer = φ(q) • Σ φ(k)
#         z = 1 / (torch.einsum('b h n d, b h d -> b h n', q_phi, k_phi.sum(dim=2)) + self.eps)  # [B, H, N]

#         # 6. 输出 = φ(q) @ KV * z
#         out = torch.einsum('b h n d, b h d e -> b h n e', q_phi, kv)  # [B, H, N, D]
#         out = out * z.unsqueeze(-1)

#         # 7. 还原回原形状
#         out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
#         out = self.proj(out)
#         out = self.proj_drop(out)

#         return out
    


class MemEffAttention(Attention):
    def forward(
        self, 
        x: Tensor, 
        attn_bias=None, 
        pos=None, 
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv.unbind(2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


