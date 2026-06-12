"""Instance-level Llama attention wrapper for TurboPolar decode."""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.integrations.mlx_lm.cache import TurboPolarFastCache


class TurboPolarLlamaAttention(nn.Module):
    """Wrap one mlx_lm Llama Attention module to route decode steps to TurboPolar.

    The wrapper is installed by replacing ``layer.attention`` in the model with
    an instance of this class. It preserves the original module and delegates
    prefill / non-TurboPolar caches to the original forward unchanged.
    """

    def __init__(
        self,
        original_attention: nn.Module,
        turbo_config: TurboPolarConfig,
        layer_index: int,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.turbo_config = turbo_config
        self.layer_index = layer_index
        # Capture the bound instance method so we can call it later.
        self._original_call = original_attention.__call__

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        if isinstance(cache, TurboPolarFastCache) and x.shape[1] == 1:
            if mask is not None:
                raise NotImplementedError(
                    "TurboPolar fused decode currently supports mask=None only."
                )

            B, L, D = x.shape
            attn = self.original_attention

            queries = attn.q_proj(x)
            keys = attn.k_proj(x)
            values = attn.v_proj(x)

            queries = queries.reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

            queries = attn.rope(queries, offset=cache.offset)
            keys = attn.rope(keys, offset=cache.offset)

            output = cache.decode_attention(queries, keys, values, attn.scale, mask=mask)
            # output: [B, H_q, D] -> [B, L, H_q * D]
            output = output[:, None, :, :].transpose(0, 2, 1, 3).reshape(B, L, -1)
            return attn.o_proj(output)

        return self._original_call(x, mask=mask, cache=cache)

    def __getattr__(self, name: str):
        """Transparently expose original attention attributes (e.g., for inspection)."""
        if name in ("original_attention", "turbo_config", "layer_index", "_original_call"):
            raise AttributeError(name)
        return getattr(self.original_attention, name)
