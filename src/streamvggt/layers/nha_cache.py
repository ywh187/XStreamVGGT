# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers


class NHACache(transformers.cache_utils.Cache):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    is_compileable = False

    def __init__(
        self,
        seen_tokens: int = 0
    ) -> NHACache:
        super().__init__(layers=[])

        self.states: List[Dict[str, Any]] = []

        self.former_tokens = seen_tokens
        self._seen_tokens = seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        recurrent_state: torch.Tensor = None,
        attn_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        conv_state: Tuple[torch.Tensor] = None,
        ffn_state: torch.Tensor = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new `recurrent_state`/`attn_state`/`conv_state` for the layer `layer_idx`.

        Args:
            recurrent_state (`torch.Tensor`, `optional`):
                The new recurrent state to cache.
            attn_state (`Tuple[torch.Tensor, torch.Tensor]`, `optional`):
                The new attention key/value states to cache.
            conv_state (`Tuple[torch.Tensor]`, `optional`):
                The new convolution state to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, `optional`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """

        # Update the number of seen tokens
        if layer_idx == 0:
            self.former_tokens = self._seen_tokens
            self._seen_tokens += offset

        if attn_state is not None:
            input_size = attn_state[0].shape[-2]
            window_size = cache_kwargs.get('window_size', None)
            if not isinstance(attn_state, Tuple) or len(attn_state) != 3:
                raise ValueError("`attn_state` must be a tuple of two tensors for key/value states")
        if len(self.states) <= layer_idx:
            if attn_state is not None:
                if window_size is not None and input_size > window_size:
                    attn_state = (attn_state[0][..., -window_size:, :].contiguous(),
                                  attn_state[1][..., -window_size:, :].contiguous(),
                                  attn_state[2][..., -window_size:, :].contiguous())
            state = dict(
                recurrent_state=recurrent_state,
                attn_state=attn_state,
                conv_state=conv_state,
                ffn_state=ffn_state
            )
            self.states.append(state)
        else:
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state['recurrent_state'] = recurrent_state
            if attn_state is not None:
                key_state, value_state, f_state = state['attn_state']
                if window_size is not None and key_state.shape[-2] == window_size:
                    # DO NOT allocate new memory if the cache is full
                    # roll the key/value states to the left by `input_size`
                    key_state = key_state.roll(-input_size, -2)
                    value_state = value_state.roll(-input_size, -2)
                    f_state = f_state.roll(-input_size, -2)
                    # replace the last `input_size` tokens with the new key/value states
                    key_state[..., -input_size:, :] = attn_state[0]
                    value_state[..., -input_size:, :] = attn_state[1]
                    f_state[..., -input_size:, :] = attn_state[2]
                    attn_state = (key_state, value_state, f_state)
                else:
                    attn_state = (torch.cat([key_state, attn_state[0]], -2),
                                  torch.cat([value_state, attn_state[1]], -2),
                                  torch.cat([f_state, attn_state[2]], -2),)
                state['attn_state'] = attn_state
            if conv_state is not None:
                state['conv_state'] = conv_state
            if ffn_state is not None:
                state['ffn_state'] = ffn_state

        return state
    
    def update_no_g(
        self,
        recurrent_state: torch.Tensor = None,
        attn_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        conv_state: Tuple[torch.Tensor] = None,
        ffn_state: torch.Tensor = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new `recurrent_state`/`attn_state`/`conv_state` for the layer `layer_idx`.

        Args:
            recurrent_state (`torch.Tensor`, `optional`):
                The new recurrent state to cache.
            attn_state (`Tuple[torch.Tensor, torch.Tensor]`, `optional`):
                The new attention key/value states to cache.
            conv_state (`Tuple[torch.Tensor]`, `optional`):
                The new convolution state to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, `optional`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """

        # Update the number of seen tokens
        if layer_idx == 0:
            self.former_tokens = self._seen_tokens
            self._seen_tokens += offset

        if cache_kwargs is None:
            cache_kwargs = {}
        
        if attn_state is not None:
            input_size = attn_state[0].shape[-2]
            window_size = cache_kwargs.get('window_size', None)
            if not isinstance(attn_state, Tuple) or len(attn_state) != 2:
                raise ValueError("`attn_state` must be a tuple of two tensors for key/value states")
        if len(self.states) <= layer_idx:
            if attn_state is not None:
                if window_size is not None and input_size > window_size:
                    attn_state = (attn_state[0][..., -window_size:, :].contiguous(),
                                  attn_state[1][..., -window_size:, :].contiguous())
            state = dict(
                recurrent_state=recurrent_state,
                attn_state=attn_state,
                conv_state=conv_state,
                ffn_state=ffn_state
            )
            self.states.append(state)
        else:
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state['recurrent_state'] = recurrent_state
            if attn_state is not None:
                key_state, value_state = state['attn_state']
                if window_size is not None and key_state.shape[-2] == window_size:
                    # DO NOT allocate new memory if the cache is full
                    # roll the key/value states to the left by `input_size`
                    key_state = key_state.roll(-input_size, -2)
                    value_state = value_state.roll(-input_size, -2)
                    # replace the last `input_size` tokens with the new key/value states
                    key_state[..., -input_size:, :] = attn_state[0]
                    value_state[..., -input_size:, :] = attn_state[1]
                    attn_state = (key_state, value_state)
                else:
                    attn_state = (torch.cat([key_state, attn_state[0]], -2),
                                  torch.cat([value_state, attn_state[1]], -2),)
                state['attn_state'] = attn_state
            if conv_state is not None:
                state['conv_state'] = conv_state
            if ffn_state is not None:
                state['ffn_state'] = ffn_state

        return state

    def get_pop_kvf(self, layer_idx, window_size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the last key, value, f in the cached states. We use these tensors to update GSA"""
        if len(self.states) <= layer_idx:
            return (None, None, None)
        state = self.states[layer_idx]
        attn_state = state['attn_state']
        if attn_state is not None and attn_state[0].shape[-2] >= window_size:
            return (attn_state[0][..., 0:1, :],
                    attn_state[1][..., 0:1, :],
                    attn_state[2][..., 0:1, :],)
        else:
            return (None, None, None)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self.former_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple:
        return tuple(self.states)

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple] = None,
        seen_tokens: int = 0
    ) -> NHACache:
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(seen_tokens)
        if isinstance(past_key_values, list):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache


class StreamNHACache(NHACache):
    """
    A specialized cache for streaming attention with a static, slot, and new token partition.
    """
    def __init__(
        self,
        seen_tokens: int = 0,
        static_size: int = 768,
        slot_size: int = 768,
        num_layers: int = 0,
    ) -> StreamNHACache:
        super().__init__(seen_tokens)
        self.static_size = static_size
        self.slot_size = slot_size
        self.cache_status = ['EMPTY'] * num_layers # EMPTY, STATIC_FILLED, SLOT_FILLED
        for _ in range(num_layers):
            self.states.append(dict(
                recurrent_state=None,
                static_attn_state=None,
                slot_attn_state=None,
            ))

    def update(
        self,
        recurrent_state: torch.Tensor = None,
        attn_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new states.
        """
        if layer_idx == 0:
            self.former_tokens = self._seen_tokens
            self._seen_tokens += offset

        if len(self.states) <= layer_idx:
            state = dict(
                recurrent_state=None,
                static_attn_state=None,
                slot_attn_state=None,
            )
            self.states.append(state)
        else:
            state = self.states[layer_idx]

        if recurrent_state is not None:
            state['recurrent_state'] = recurrent_state

        if attn_state is not None:
            # The shape of attn_state is (B, N, H, D)
            if self.cache_status[layer_idx] == 'EMPTY':
                # First 768 tokens, fill static cache
                state['static_attn_state'] = attn_state
                if attn_state[0].shape[1] == self.static_size:
                    self.cache_status[layer_idx] = 'STATIC_FILLED'
            elif self.cache_status[layer_idx] == 'STATIC_FILLED':
                # Second 768 tokens, fill slot cache
                state['slot_attn_state'] = attn_state
                if attn_state[0].shape[1] == self.slot_size:
                    self.cache_status[layer_idx] = 'SLOT_FILLED'
            elif self.cache_status[layer_idx] == 'SLOT_FILLED':
                # GSA update for the slot
                state['slot_attn_state'] = attn_state

        return state

    def get_kv(self, partition: str, layer_idx: int = 0) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Returns the key, value, and gate tensors for a specific partition.
        """
        if len(self.states) <= layer_idx:
            return None, None, None

        state = self.states[layer_idx]
        if partition == 'static':
            return state.get('static_attn_state', (None, None, None))
        elif partition == 'slot':
            return state.get('slot_attn_state', (None, None, None))
        else:
            raise ValueError(f"Unknown partition: {partition}")