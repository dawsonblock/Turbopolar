import mlx.core as mx
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizedVBlock:
    codes: mx.array
    scales: mx.array
    group_size: int
    # Asymmetric quantization: zero_points for each group.
    # If None, symmetric quantization is used (codes * scales).
    zero_points: mx.array = None


class GroupedVQuantizer:
    def __init__(self, group_size: int = 32, asymmetric: bool = False):
        self.group_size = group_size
        self.asymmetric = asymmetric
        # Asymmetric uses full uint8 range [0, 255]; symmetric uses [-128, 127]
        self._qmax = 255.0 if asymmetric else 127.0
        self._qmin = 0.0 if asymmetric else -128.0
        self._min_scale = 1e-4

    def _quantize(self, reshaped: mx.array) -> tuple:
        """Shared quantization logic for symmetric and asymmetric modes."""
        if self.asymmetric:
            # Asymmetric: scale = (max - min) / 255, zp = -min / scale
            vmin = mx.min(reshaped, axis=-1, keepdims=True)
            vmax = mx.max(reshaped, axis=-1, keepdims=True)
            scale = (vmax - vmin) / self._qmax
            scale = mx.where(
                scale == 0, mx.array(self._min_scale, dtype=scale.dtype), scale
            )
            zero_point = mx.round(-vmin / scale)
            zero_point = mx.clip(zero_point, self._qmin, self._qmax)
            quantized = mx.round(reshaped / scale) + zero_point
            codes = mx.clip(quantized, self._qmin, self._qmax)
            if self._qmin == 0:
                codes = codes.astype(mx.uint8)
            else:
                codes = codes.astype(mx.int8)
            return codes, scale.astype(mx.float16), zero_point.astype(mx.float16)
        else:
            # Symmetric: scale = max_abs / 127
            max_abs = mx.max(mx.abs(reshaped), axis=-1, keepdims=True)
            scale = (max_abs / self._qmax).astype(mx.float16)
            scale = mx.where(
                scale == 0, mx.array(self._min_scale, dtype=mx.float16), scale
            )
            quantized = mx.round(reshaped / scale)
            codes = mx.clip(quantized, self._qmin, self._qmax).astype(mx.int8)
            return codes, scale, None

    def quantize_block(self, v_block: mx.array) -> QuantizedVBlock:
        B, H, S, L, D = v_block.shape
        assert D % self.group_size == 0
        num_groups = D // self.group_size
        reshaped = v_block.reshape(B, H, S, L, num_groups, self.group_size)
        codes, scales, zero_points = self._quantize(reshaped)
        return QuantizedVBlock(
            codes=codes.reshape(B, H, S, L, D),
            scales=scales.squeeze(-1),
            group_size=self.group_size,
            zero_points=zero_points.squeeze(-1) if zero_points is not None else None,
        )

    def encode_blocks(self, v_blocks: mx.array) -> QuantizedVBlock:
        """Quantize multiple blocks in a single batch operation.

        Args:
            v_blocks: [B, H, N, block_size, D] where N is the number of blocks.

        Returns:
            QuantizedVBlock with codes [B, H, N, block_size, D].
        """
        B, H, N, L, D = v_blocks.shape
        assert D % self.group_size == 0
        num_groups = D // self.group_size
        reshaped = v_blocks.reshape(B, H, N, L, num_groups, self.group_size)
        codes, scales, zero_points = self._quantize(reshaped)
        return QuantizedVBlock(
            codes=codes.reshape(B, H, N, L, D),
            scales=scales.squeeze(-1),
            group_size=self.group_size,
            zero_points=zero_points.squeeze(-1) if zero_points is not None else None,
        )

    def dequantize_block(self, block: QuantizedVBlock) -> mx.array:
        B, H, S, L, D = block.codes.shape
        num_groups = D // block.group_size
        reshaped_codes = block.codes.reshape(
            B, H, S, L, num_groups, block.group_size
        ).astype(mx.float32)
        reshaped_scales = mx.expand_dims(block.scales, axis=-1).astype(mx.float32)
        if block.zero_points is not None:
            reshaped_zp = mx.expand_dims(block.zero_points, axis=-1).astype(
                mx.float32
            )
            dequantized = (reshaped_codes - reshaped_zp) * reshaped_scales
        else:
            dequantized = reshaped_codes * reshaped_scales
        return dequantized.reshape(B, H, S, L, D).astype(mx.float16)
