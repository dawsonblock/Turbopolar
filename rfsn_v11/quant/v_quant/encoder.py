import mlx.core as mx
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizedVBlock:
    codes: mx.array
    scales: mx.array
    group_size: int


class GroupedVQuantizer:
    def __init__(self, group_size: int = 32):
        self.group_size = group_size

    def quantize_block(self, v_block: mx.array) -> QuantizedVBlock:
        B, H, S, L, D = v_block.shape
        assert D % self.group_size == 0
        num_groups = D // self.group_size
        reshaped = v_block.reshape(B, H, S, L, num_groups, self.group_size)
        max_abs = mx.max(mx.abs(reshaped), axis=-1, keepdims=True)
        scales = (max_abs / 127.0).astype(mx.float16)
        scales = mx.where(scales == 0, mx.array(1e-4, dtype=mx.float16), scales)
        quantized = mx.round(reshaped / scales)
        codes = mx.clip(quantized, -128, 127).astype(mx.int8)
        return QuantizedVBlock(
            codes=codes.reshape(B, H, S, L, D),
            scales=scales.squeeze(-1),
            group_size=self.group_size,
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
        max_abs = mx.max(mx.abs(reshaped), axis=-1, keepdims=True)
        scales = (max_abs / 127.0).astype(mx.float16)
        scales = mx.where(scales == 0, mx.array(1e-4, dtype=mx.float16), scales)
        quantized = mx.round(reshaped / scales)
        codes = mx.clip(quantized, -128, 127).astype(mx.int8)
        return QuantizedVBlock(
            codes=codes.reshape(B, H, N, L, D),
            scales=scales.squeeze(-1),
            group_size=self.group_size,
        )

    def dequantize_block(self, block: QuantizedVBlock) -> mx.array:
        B, H, S, L, D = block.codes.shape
        num_groups = D // block.group_size
        reshaped_codes = block.codes.reshape(B, H, S, L, num_groups, block.group_size).astype(mx.float32)
        reshaped_scales = mx.expand_dims(block.scales, axis=-1).astype(mx.float32)
        dequantized = reshaped_codes * reshaped_scales
        return dequantized.reshape(B, H, S, L, D).astype(mx.float16)
