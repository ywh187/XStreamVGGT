# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

"""Qwen3MoE model configuration"""

from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen3MoeNHAConfig(PretrainedConfig):

    model_type = "qwen3_moe_nha"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3Moe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        num_slots: int = 64,
        window_size: int = 32,
        block_size: Optional[int] = None,
        transformer_idx: Optional[int] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.num_slots = num_slots
        self.window_size = window_size
        self.block_size = block_size
        self.transformer_idx = transformer_idx
