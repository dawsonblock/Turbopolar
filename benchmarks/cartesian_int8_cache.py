"""Simple Cartesian int8 KV cache baseline for mlx_lm.

Unlike TurboPolar, this baseline keeps keys and values in their original
Cartesian (coordinate) form and quantizes each token vector independently to
int8 using a per-token scale.  It decompresses back to fp16 on every read so
standard MLX attention can run unchanged.
"""

from typing import Optional, Tuple

import mlx.core as mx


class CartesianInt8Cache:
    """MLX-LM-compatible KV cache that stores K/V as per-token int8 vectors."""

    def __init__(self):
        self.k_codes: Optional[mx.array] = None
        self.k_scales: Optional[mx.array] = None
        self.v_codes: Optional[mx.array] = None
        self.v_scales: Optional[mx.array] = None
        self.offset = 0

    def _quantize(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Per-token symmetric int8 quantization.

        x shape: (B, H, T, D).  Returns (codes int8, scales fp16).
        scales shape: (B, H, T, 1).
        """
        original_dtype = x.dtype
        x = x.astype(mx.float16)
        max_abs = mx.max(mx.abs(x), axis=-1, keepdims=True)
        # Avoid division by zero; a zero scale is harmless because the codes are zero too.
        scale = (max_abs / 127.0).astype(mx.float16)
        scale = mx.where(scale == 0, mx.array(1e-4, dtype=mx.float16), scale)
        codes = mx.clip(mx.round(x / scale), -128, 127).astype(mx.int8)
        if original_dtype != mx.float16:
            scale = scale.astype(original_dtype)
        return codes, scale

    def _dequantize(self, codes: mx.array, scales: mx.array, dtype) -> mx.array:
        return (codes.astype(scales.dtype) * scales).astype(dtype)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        original_dtype = keys.dtype
        k_codes, k_scales = self._quantize(keys)
        v_codes, v_scales = self._quantize(values)

        if self.k_codes is None:
            self.k_codes = k_codes
            self.k_scales = k_scales
            self.v_codes = v_codes
            self.v_scales = v_scales
        else:
            self.k_codes = mx.concatenate([self.k_codes, k_codes], axis=2)
            self.k_scales = mx.concatenate([self.k_scales, k_scales], axis=2)
            self.v_codes = mx.concatenate([self.v_codes, v_codes], axis=2)
            self.v_scales = mx.concatenate([self.v_scales, v_scales], axis=2)

        self.offset = self.k_codes.shape[2]
        k_dense = self._dequantize(self.k_codes, self.k_scales, original_dtype)
        v_dense = self._dequantize(self.v_codes, self.v_scales, original_dtype)
        return k_dense, v_dense

    def make_mask(self, N: int, return_array: bool = False, window_size: Optional[int] = None):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(N, self.offset, return_array, window_size)

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        if self.k_codes is None:
            return 0
        # int8 codes + fp16 scales per token; K and V.
        return int(
            self.k_codes.size * self.k_codes.itemsize
            + self.k_scales.size * self.k_scales.itemsize
            + self.v_codes.size * self.v_codes.itemsize
            + self.v_scales.size * self.v_scales.itemsize
        )

    def empty(self) -> bool:
        return self.offset == 0
