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
        cache_size: int = 8192,  # KV cache max len
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


        # xstreamvggt parameter
        self.prune_mode = os.getenv("KV_PRUNE_MODE", "XStreamVGGT") # XStreamVGGT / SlidingWindow / Random
        self.cache_size = int(os.getenv("KV_CACHE_SIZE", cache_size)) # cache max len
        self.pool_size = int(os.getenv("KV_POOL_SIZE", 16)) # pooling len
        self.kv_quant = os.getenv("KV_QUANT_MODE", "KCVT")
        


    def prune_kv_cache(self, kv, q_new): #, token_num_eachframe=0):
        mode = self.prune_mode
        if mode == "XStreamVGGT":
            return self._prune_fast_vggt_protected(kv, q_new) #, token_num_eachframe=token_num_eachframe)
        elif mode == "SlidingWindow":
            return self._prune_sliding_window(kv)
        elif mode == "Random":
            return self._prune_random(kv)
        else: 
            return kv

    def _prune_fast_vggt_protected(self, kv, q_new): #, token_num_eachframe=0):
        k, v = kv
        B, H, T, D = k.shape
        N_new = q_new.shape[2]

        if T <= self.cache_size:
            return kv

        protected_size = N_new 
        
        window_keep = N_new
        
        total_keep = self.cache_size
        middle_keep = max(0, total_keep - protected_size - window_keep)
        
        prunable_start = protected_size
        prunable_end = T - window_keep
        prunable_length = max(0, prunable_end - prunable_start)
        
        if prunable_length == 0:
            return kv
        
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

        k_prunable = k[:, :, prunable_start:prunable_end, :]
        k_score = k_prunable.mean(dim=1)
        scores = (q_score @ k_score.transpose(-2, -1)).mean(dim=1)

        topk_count = min(middle_keep, scores.shape[1])
        if topk_count > 0:
            topk_idx_relative = torch.topk(scores, topk_count, dim=-1).indices
            topk_idx = topk_idx_relative + prunable_start
        else:
            topk_idx = torch.empty(B, 0, dtype=torch.long, device=k.device)

        protected_idx = torch.arange(0, protected_size, device=k.device).unsqueeze(0).expand(B, -1)
        window_idx = torch.arange(T-window_keep, T, device=k.device).unsqueeze(0).expand(B, -1)

        keep_idx = torch.cat([protected_idx, topk_idx, window_idx], dim=-1)
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

        # T - cache_size → T
        new_k = k[:, :, -self.cache_size:, :]
        new_v = v[:, :, -self.cache_size:, :]
        return new_k, new_v

    def _prune_random(self, kv):
        k, v = kv
        B, H, T, D = k.shape

        if T <= self.cache_size:
            return kv

        idx = torch.randperm(T, device=k.device)[:self.cache_size]
        idx = torch.sort(idx).values           
        idx = idx.unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(B, H, -1, D)

        new_k = torch.gather(k, 2, idx)
        new_v = torch.gather(v, 2, idx)
        return new_k, new_v



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

            if self.kv_quant:
                preserve_tokens = 0
                kvtc = self.kv_quant
                k_bit = 4
                v_bit = 4
                gs = 64


                # fake_quant
                if k.shape[-1] == 64:
                        k_before_quant = new_kv[0]
                        v_before_quant = new_kv[1]

                        print(f"kvtc={kvtc}")
                        print(f"k_bit={k_bit}, v_bit={v_bit}")
                        if kvtc == "KTVT":
                            k_quant = fake_quant_preserve_tokens(k_before_quant, num_skip=preserve_tokens, skip_dim=-2,clip_ratio=1, bit=k_bit, quant_gs=gs)
                            v_quant = fake_quant_preserve_tokens(v_before_quant, num_skip=preserve_tokens, skip_dim=-2, clip_ratio=1, bit=v_bit, quant_gs=gs)
                        elif kvtc == "KCVT":
                            k_quant = fake_quant_preserve_tokens(k_before_quant.transpose(-1,-2), num_skip=preserve_tokens, skip_dim=-1, clip_ratio=1, bit=k_bit, quant_gs=gs).transpose(-1,-2)
                            v_quant = fake_quant_preserve_tokens(v_before_quant, num_skip=preserve_tokens, skip_dim=-2, clip_ratio=1, bit=v_bit, quant_gs=gs)  
                        elif kvtc == "KTVC":
                            k_quant = fake_quant_preserve_tokens(k_before_quant, num_skip=preserve_tokens, skip_dim=-2, clip_ratio=1, bit=k_bit, quant_gs=gs)
                            v_quant = fake_quant_preserve_tokens(v_before_quant.transpose(-1,-2), num_skip=preserve_tokens, skip_dim=-1, clip_ratio=1, bit=v_bit, quant_gs=gs).transpose(-1,-2)                                           
                        elif kvtc == "KCVC":
                            k_quant = fake_quant_preserve_tokens(k_before_quant.transpose(-1,-2), num_skip=preserve_tokens, skip_dim=-1, clip_ratio=1, bit=k_bit, quant_gs=gs).transpose(-1,-2)
                            v_quant = fake_quant_preserve_tokens(v_before_quant.transpose(-1,-2), num_skip=preserve_tokens, skip_dim=-1, clip_ratio=1, bit=v_bit, quant_gs=gs).transpose(-1,-2) 
                        else:
                            k_quant = k_before_quant
                            v_quant = v_before_quant
                        new_kv = (k_quant, v_quant)


        else:
            new_kv = None
        

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, # torch.Size([1, 16, 1041, 64])
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )


        x = x.transpose(1,2).reshape(B,N,C)
        x = self.proj_drop(self.proj(x))


        # ---- KV cache pruning ---- 
        if use_cache:

            new_kv = self.prune_kv_cache(new_kv, q) #, token_num_eachframe=kwargs.get("token_num_eachframe", 0))

            return x, new_kv

        return x



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

def find_params_groupwise_lastdim_safe(x, clip_ratio, bit, quant_gs):
    """
    Group-wise asymmetric quantization parameters (SAFE VERSION)

    Args:
        x: input tensor, shape (..., D)
        clip_ratio: float > 0
        bit: quantization bit-width
        quant_gs: group size on last dimension

    Returns:
        scale: (..., D_pad)
        zero:  (..., D_pad)
        pad_len: int
    """
    *prefix, D = x.shape

    # ---------- Pad last dim ----------
    remainder = D % quant_gs
    if remainder != 0:
        pad_len = quant_gs - remainder
        x = F.pad(x, (0, pad_len))
        D_pad = D + pad_len
    else:
        pad_len = 0
        D_pad = D

    x_grouped = x.reshape(*prefix, D_pad // quant_gs, quant_gs)

    # ---------- Min / Max ----------
    clip_ratio = max(float(clip_ratio), 1e-6)

    xmax = torch.amax(x_grouped, dim=-1, keepdim=True) * clip_ratio
    xmin = torch.amin(x_grouped, dim=-1, keepdim=True) * clip_ratio

    maxq = (1 << bit) - 1

    # ---------- Scale (critical) ----------
    if x.dtype in (torch.float16, torch.bfloat16):
        eps = 1e-6
    else:
        eps = 1e-12

    scale = (xmax - xmin) / maxq
    scale = torch.clamp(scale, min=eps)

    # ---------- Zero (critical) ----------
    zero = torch.round(-xmin / scale)
    zero = torch.nan_to_num(zero, nan=0.0, posinf=maxq, neginf=0.0)
    zero = torch.clamp(zero, 0, maxq)

    # ---------- Expand back ----------
    scale = scale.repeat_interleave(quant_gs, dim=-1).reshape(*prefix, D_pad)
    zero  = zero.repeat_interleave(quant_gs, dim=-1).reshape(*prefix, D_pad)

    return scale, zero, pad_len


def asym_quant_safe(x, scale, zero, bit):
    """
    Safe asymmetric quantization
    """
    maxq = (1 << bit) - 1
    q = torch.round(x / scale) + zero
    q = torch.nan_to_num(q, nan=0.0, posinf=maxq, neginf=0.0)
    q = torch.clamp(q, 0, maxq)
    return q


def asym_dequant_safe(q, scale, zero):
    """
    Safe asymmetric dequantization
    """
    return scale * (q - zero)


def fake_quant_safe(x, clip_ratio, bit, quant_gs):
    """
    Group-wise fake quantization (SAFE)
    """
    scale, zero, pad_len = find_params_groupwise_lastdim_safe(
        x, clip_ratio, bit, quant_gs
    )

    # ---------- Pad x ----------
    if pad_len > 0:
        x_pad = F.pad(x, (0, pad_len))
    else:
        x_pad = x

    q = asym_quant_safe(x_pad, scale, zero, bit)
    deq = asym_dequant_safe(q, scale, zero)

    # ---------- Remove pad ----------
    if pad_len > 0:
        deq = deq[..., :-pad_len]

    return deq


def fake_quant_preserve_tokens(
    x, num_skip, skip_dim, clip_ratio, bit, quant_gs
):
    """
    Skip first num_skip elements along skip_dim, quantize the rest (SAFE)
    """
    assert num_skip >= 0
    D = x.shape[skip_dim]
    assert num_skip <= D

    idx_all = torch.arange(D, device=x.device)
    idx_skip = idx_all[:num_skip]
    idx_quant = idx_all[num_skip:]

    x_skip = torch.index_select(x, dim=skip_dim, index=idx_skip)
    x_quant = torch.index_select(x, dim=skip_dim, index=idx_quant)

    x_quant_q = fake_quant_safe(
        x_quant, clip_ratio, bit, quant_gs
    )

    x_out = torch.cat([x_skip, x_quant_q], dim=skip_dim)
    return x_out
