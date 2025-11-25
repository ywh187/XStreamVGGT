import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

try:
    from fla.layers import GatedLinearAttention
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    print("Warning: flash-linear-attention not installed. Install with: pip install -U git+https://github.com/fla-org/flash-linear-attention")


class GLAAttention(nn.Module):
    """
    Gated Linear Attention wrapper that mimics the interface of your original Attention class.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        qkv_bias: Not used in GLA (kept for compatibility)
        proj_bias: Not used in GLA (kept for compatibility)
        attn_drop: Dropout rate (not used in GLA, kept for compatibility)
        proj_drop: Dropout rate for output projection
        norm_layer: Normalization layer (GLA uses RMSNorm internally)
        qk_norm: Not used in GLA (kept for compatibility)
        fused_attn: Not used in GLA (kept for compatibility)
        rope: RoPE is not directly supported in GLA (position encoding handled differently)
        mode: GLA computation mode ('chunk', 'fused_chunk', 'fused_recurrent')
        expand_k: Key expansion ratio (default 0.5 means key_dim = dim * 0.5)
        expand_v: Value expansion ratio (default 1.0 means value_dim = dim)
        use_short_conv: Whether to use short convolutions
        conv_size: Convolution kernel size
        gate_fn: Gate activation function ('swish' or 'silu')
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,  # Not used in GLA
        proj_bias: bool = True,  # Not used in GLA
        attn_drop: float = 0.0,  # Not used in GLA
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,  # GLA uses RMSNorm
        qk_norm: bool = False,  # Not used in GLA
        fused_attn: bool = True,  # Not used in GLA
        rope=None,  # Not directly supported
        # GLA specific parameters
        mode: str = 'chunk',  # 'chunk', 'fused_chunk', or 'fused_recurrent'
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        use_short_conv: bool = False,
        conv_size: int = 4,
        gate_fn: str = 'swish',
    ) -> None:
        super().__init__()
        
        if not FLA_AVAILABLE:
            raise ImportError(
                "flash-linear-attention is not installed. "
                "Install it with: pip install -U git+https://github.com/fla-org/flash-linear-attention"
            )
        
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.rope = rope
        
        if rope is not None:
            print("Warning: RoPE is passed but GLA handles position encoding differently. "
                  "The rope parameter will be ignored.")
        
        # Initialize GLA layer
        self.gla = GatedLinearAttention(
            mode=mode,
            hidden_size=dim,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            num_kv_heads=None,  # Use same as num_heads
            feature_map=None,  # Use default
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=False,
            use_output_gate=True,
            gate_fn=gate_fn,
            elementwise_affine=True,
            norm_eps=1e-5,
            gate_logit_normalizer=16,
            gate_low_rank_dim=16,
            clamp_min=None,
            fuse_norm=True,
            layer_idx=None,
        )
        
        # Output projection dropout (GLA has internal dropout in o_proj)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        
    def forward(
        self, 
        x: torch.Tensor, 
        pos=None,  # Not used in GLA
        attn_mask=None,  # GLA uses attention_mask
        past_key_values=None,  # GLA uses Cache object
        use_cache=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Forward pass compatible with your original Attention interface.
        
        Args:
            x: Input tensor of shape (B, N, C)
            pos: Position encodings (not used in GLA)
            attn_mask: Attention mask (0-1 matrix, 0 for padding)
            past_key_values: Past key-value cache
            use_cache: Whether to return cache
            
        Returns:
            If use_cache=False: output tensor of shape (B, N, C)
            If use_cache=True: (output, new_cache) tuple
        """
        B, N, C = x.shape
        
        if pos is not None:
            print("Warning: pos parameter is not used in GLA. Position encoding is handled internally.")
        
        # Convert attn_mask format if needed
        # Your mask: None or (N, N) or (..., N, N) additive mask
        # GLA mask: None or (B, N) 0-1 mask where 0 indicates padding
        attention_mask = None
        if attn_mask is not None:
            # If it's a causal mask, GLA handles this internally
            # If it's a padding mask, we need to convert it
            # For simplicity, we'll pass None and let GLA use its default causal masking
            # If you have specific padding requirements, you'll need to convert the mask
            if attn_mask.dim() == 2:  # (N, N)
                attention_mask = None  # Use GLA's default causal mask
            else:
                attention_mask = None  # Use GLA's default causal mask
        
        # Handle past_key_values
        # GLA expects a Cache object, but for simplicity we'll use None
        # and rely on GLA's internal cache mechanism
        gla_cache = None
        if past_key_values is not None:
            # You would need to convert your cache format to GLA's Cache format
            # For now, we'll start fresh
            print("Warning: past_key_values conversion not fully implemented. Starting fresh.")
            gla_cache = None
        
        # Forward through GLA
        # GLA returns: (output, attention_weights, past_key_values)
        output = self.gla(
            hidden_states=x,
            attention_mask=attention_mask,
            past_key_values=gla_cache,
            use_cache=use_cache,
            output_attentions=False,
        )
        
        # GLA returns a tuple: (output, attention_weights, cache)
        if isinstance(output, tuple):
            out, attn_weights, new_cache = output
        else:
            out = output
            new_cache = None
        
        # Apply output dropout
        out = self.proj_drop(out)
        
        if use_cache:
            # Convert GLA cache back to your format if needed
            # For now, return as-is
            return out, new_cache
        else:
            return out


# Example usage and initialization
if __name__ == "__main__":
    # Example: Replace your Attention with GLAAttention
    dim = 1024
    num_heads = 16
    
    # Original initialization style
    """
    self.attn = Attention(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        proj_bias=proj_bias,
        attn_drop=attn_drop,
        proj_drop=drop,
        qk_norm=qk_norm,
        fused_attn=fused_attn,
        rope=rope,
    )
    """
    
    # New GLA initialization
    gla_attn = GLAAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,  # Not used, kept for compatibility
        proj_bias=True,  # Not used, kept for compatibility
        attn_drop=0.0,  # Not used, kept for compatibility
        proj_drop=0.1,  # Applied to output
        qk_norm=False,  # Not used, kept for compatibility
        fused_attn=True,  # Not used, kept for compatibility
        rope=None,  # Not supported
        # GLA specific params
        mode='chunk',  # or 'fused_chunk' for faster training
        expand_k=0.5,
        expand_v=1.0,
        use_short_conv=False,  # Set True for better performance
        conv_size=4,
        gate_fn='swish',
    )
    
    # Test forward pass
    B, N, C = 10, 708, 1024
    x = torch.randn(B, N, C)
    
    # Without cache
    output = gla_attn(x, use_cache=False)
    print(f"Output shape: {output.shape}")  # Should be (10, 708, 1024)
    
    # With cache
    output, cache = gla_attn(x, use_cache=True)
    print(f"Output shape: {output.shape}")
    print(f"Cache type: {type(cache)}")