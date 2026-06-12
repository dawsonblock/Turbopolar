"""MLX-LM-compatible KV cache wrapper around TurboPolarKVCacheRuntime.

This wrapper decompresses K/V on every read so that standard MLX attention
runs unchanged. It is intended for quality benchmarking, not speed.
"""

from typing import Optional, Tuple

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer


class TurboPolarMLXLMCache:
    """Drop-in KV cache for mlx_lm that stores K/V in TurboPolar format."""

    def __init__(self, config: TurboPolarConfig):
        self.config = config
        self.runtime = TurboPolarKVCacheRuntime(config)
        self.decoder = PolarQuantDecoder()
        self.v_dequantizer = GroupedVQuantizer(group_size=32)
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Append keys/values and return the decompressed full history."""
        original_dtype = keys.dtype
        # TurboPolar kernels and quantizers are tuned for float16.
        if keys.dtype != mx.float16:
            keys = keys.astype(mx.float16)
        if values.dtype != mx.float16:
            values = values.astype(mx.float16)
        self.runtime.append(keys, values)
        block, quant_v, dense_v, _qjl, actual_len = self.runtime.get_blocks_for_attention()
        if block is None:
            raise RuntimeError("TurboPolar cache returned no blocks after append")

        # Decompress keys.
        k_dense = self.decoder.decode_block(block)[:, :, :actual_len, :]

        # Decompress values.
        B, H_kv, S, L, _ = block.radii.shape
        if dense_v is not None:
            v_full = dense_v.reshape(B, H_kv, S * L, self.config.head_dim)
        elif quant_v is not None:
            v_full = self.v_dequantizer.dequantize_block(quant_v).reshape(B, H_kv, S * L, self.config.head_dim)
        else:
            raise RuntimeError("TurboPolar cache has no V payload")
        v_dense = v_full[:, :, :actual_len, :]

        self.offset = actual_len
        if original_dtype != k_dense.dtype:
            k_dense = k_dense.astype(original_dtype)
            v_dense = v_dense.astype(original_dtype)
        return k_dense, v_dense

    def make_mask(self, N: int, return_array: bool = False, window_size: Optional[int] = None):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(N, self.offset, return_array, window_size)

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        telem = self.runtime.get_io_telemetry()
        return int(telem.get("actual_cache_bytes", 0))

    def empty(self) -> bool:
        return self.offset == 0
