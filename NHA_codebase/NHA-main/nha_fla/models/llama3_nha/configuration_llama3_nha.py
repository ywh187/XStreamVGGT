# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig

class LlamaNHAConfig(LlamaConfig, PretrainedConfig):
    model_type = 'llama_nha'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        # llama config
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        num_slots: int = 32,
        window_size: int = 32,
        block_size: Optional[int] = None,
        transformer_idx: Optional[int] = None,
        rand_window: bool = False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
            **kwargs,
        )

        self.num_slots = num_slots
        self.window_size = window_size
        self.block_size = block_size
        self.transformer_idx = transformer_idx
        self.rand_window = rand_window