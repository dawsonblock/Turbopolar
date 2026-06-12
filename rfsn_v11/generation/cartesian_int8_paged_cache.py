"""Paged Cartesian-int8 KV cache with the same page layout as TurboPolar.

This is the fair baseline: same block size, same GQA handling, same page
allocation strategy, but using per-token symmetric int8 quantization instead
of polar encoding.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx


@dataclass
class CartesianInt8Block:
    """One block of int8 K/V data."""

    k_codes: mx.array  # [B, H, L, D]
    k_scales: mx.array  # [B, H, L, 1]
    v_codes: mx.array  # [B, H, L, D]
    v_scales: mx.array  # [B, H, L, 1]


@dataclass
class CartesianInt8Page:
    """One fixed-size page of Cartesian-int8 blocks."""

    k_codes: mx.array  # [B, H, C, L, D]
    k_scales: mx.array  # [B, H, C, L, 1]
    v_codes: mx.array  # [B, H, C, L, D]
    v_scales: mx.array  # [B, H, C, L, 1]
    valid_blocks: int
    capacity_blocks: int


class PagedCartesianInt8KStorage:
    """Paged storage for Cartesian-int8 keys."""

    def __init__(self):
        self.pages: List[CartesianInt8Page] = []
        self.block_count = 0
        self.capacity = 0
        self.reallocation_count = 0

    def _allocate_page(self, B: int, H: int, L: int, D: int, capacity_blocks: int = 16):
        self.pages.append(
            CartesianInt8Page(
                k_codes=mx.zeros((B, H, capacity_blocks, L, D), dtype=mx.int8),
                k_scales=mx.zeros((B, H, capacity_blocks, L, 1), dtype=mx.float16),
                v_codes=mx.zeros((B, H, capacity_blocks, L, D), dtype=mx.int8),
                v_scales=mx.zeros((B, H, capacity_blocks, L, 1), dtype=mx.float16),
                valid_blocks=0,
                capacity_blocks=capacity_blocks,
            )
        )
        self.capacity += capacity_blocks
        self.reallocation_count += 1

    def append(self, block: CartesianInt8Block):
        B, H, L, D = block.k_codes.shape
        if not self.pages:
            self._allocate_page(B, H, L, D)
        last = self.pages[-1]
        if last.valid_blocks >= last.capacity_blocks:
            self._allocate_page(B, H, L, D)
            last = self.pages[-1]
        idx = last.valid_blocks
        last.k_codes[:, :, idx, :, :] = block.k_codes
        last.k_scales[:, :, idx, :, :] = block.k_scales
        last.v_codes[:, :, idx, :, :] = block.v_codes
        last.v_scales[:, :, idx, :, :] = block.v_scales
        last.valid_blocks += 1
        self.block_count += 1

    def get_block(self, page_index: int, block_index: int) -> CartesianInt8Block:
        page = self.pages[page_index]
        return CartesianInt8Block(
            k_codes=page.k_codes[:, :, block_index, :, :],
            k_scales=page.k_scales[:, :, block_index, :, :],
            v_codes=page.v_codes[:, :, block_index, :, :],
            v_scales=page.v_scales[:, :, block_index, :, :],
        )

    def debug_materialize_all_blocks(self) -> CartesianInt8Block:
        """Materialize all valid blocks into a single monolithic block."""
        if not self.pages:
            raise ValueError("No pages allocated")
        blocks = []
        for page in self.pages:
            for i in range(page.valid_blocks):
                blocks.append(self.get_block(len(blocks) // page.capacity_blocks, i))
        if not blocks:
            raise ValueError("No valid blocks to materialize")
        # Concatenate along the block dimension (axis=2).
        return CartesianInt8Block(
            k_codes=mx.concatenate([b.k_codes for b in blocks], axis=2),
            k_scales=mx.concatenate([b.k_scales for b in blocks], axis=2),
            v_codes=mx.concatenate([b.v_codes for b in blocks], axis=2),
            v_scales=mx.concatenate([b.v_scales for b in blocks], axis=2),
        )


class PagedCartesianInt8KVCache:
    """MLX-LM-compatible KV cache using paged Cartesian-int8 quantization.

    Same page layout as TurboPolar but with per-token symmetric int8 encoding.
    """

    def __init__(self, block_size: int = 64):
        self.block_size = block_size
        self.storage = PagedCartesianInt8KStorage()
        self.partial_k_buffer: Optional[mx.array] = None
        self.partial_v_buffer: Optional[mx.array] = None
        self.partial_length = 0
        self.actual_seq_len = 0
        self._batch_size = 0
        self._num_kv_heads = 0
        self._head_dim = 0

    def _init_buffers(self, B: int, H: int, D: int):
        self.partial_k_buffer = mx.zeros((B, H, self.block_size, D), dtype=mx.float16)
        self.partial_v_buffer = mx.zeros((B, H, self.block_size, D), dtype=mx.float16)
        self._batch_size = B
        self._num_kv_heads = H
        self._head_dim = D

    @staticmethod
    def _quantize_block(x: mx.array) -> Tuple[mx.array, mx.array]:
        x = x.astype(mx.float16)
        max_abs = mx.max(mx.abs(x), axis=-1, keepdims=True)
        scale = (max_abs / 127.0).astype(mx.float16)
        scale = mx.where(scale == 0, mx.array(1e-4, dtype=mx.float16), scale)
        codes = mx.clip(mx.round(x / scale), -128, 127).astype(mx.int8)
        return codes, scale

    @staticmethod
    def _dequantize(codes: mx.array, scales: mx.array, dtype) -> mx.array:
        return (codes.astype(scales.dtype) * scales).astype(dtype)

    def append(self, k_new: mx.array, v_new: mx.array):
        if k_new.dtype != mx.float16:
            k_new = k_new.astype(mx.float16)
        if v_new.dtype != mx.float16:
            v_new = v_new.astype(mx.float16)

        if self.partial_k_buffer is None:
            self._init_buffers(k_new.shape[0], k_new.shape[1], k_new.shape[3])

        B, H, T_new, D = k_new.shape
        L = self.block_size
        for t in range(T_new):
            self.partial_k_buffer[:, :, self.partial_length, :] = k_new[:, :, t, :]
            self.partial_v_buffer[:, :, self.partial_length, :] = v_new[:, :, t, :]
            self.partial_length += 1
            self.actual_seq_len += 1
            if self.partial_length >= L:
                k_block = self.partial_k_buffer[:, :, :L, :]
                v_block = self.partial_v_buffer[:, :, :L, :]
                k_codes, k_scales = self._quantize_block(k_block)
                v_codes, v_scales = self._quantize_block(v_block)
                self.storage.append(
                    CartesianInt8Block(
                        k_codes=k_codes,
                        k_scales=k_scales,
                        v_codes=v_codes,
                        v_scales=v_scales,
                    )
                )
                self.partial_length = 0
                self.partial_k_buffer = mx.zeros_like(self.partial_k_buffer)
                self.partial_v_buffer = mx.zeros_like(self.partial_v_buffer)

    def get_history(self) -> Tuple[mx.array, mx.array]:
        """Return decompressed K and V history."""
        if self.storage.block_count == 0:
            if self.partial_length > 0:
                return (
                    self.partial_k_buffer[:, :, : self.partial_length, :],
                    self.partial_v_buffer[:, :, : self.partial_length, :],
                )
            # Empty history: infer shape from buffers if allocated, else generic.
            if self.partial_k_buffer is not None:
                B, H, _, D = self.partial_k_buffer.shape
                return (
                    mx.zeros((B, H, 0, D), dtype=mx.float16),
                    mx.zeros((B, H, 0, D), dtype=mx.float16),
                )
            return (
                mx.zeros((1, 1, 0, 128), dtype=mx.float16),
                mx.zeros((1, 1, 0, 128), dtype=mx.float16),
            )

        mono = self.storage.debug_materialize_all_blocks()
        k_dense = self._dequantize(mono.k_codes, mono.k_scales, mx.float16)
        v_dense = self._dequantize(mono.v_codes, mono.v_scales, mx.float16)

        if self.partial_length > 0:
            k_dense = mx.concatenate(
                [
                    k_dense,
                    self.partial_k_buffer[:, :, : self.partial_length, :],
                ],
                axis=2,
            )
            v_dense = mx.concatenate(
                [
                    v_dense,
                    self.partial_v_buffer[:, :, : self.partial_length, :],
                ],
                axis=2,
            )

        return k_dense, v_dense

    @property
    def offset(self) -> int:
        return self.actual_seq_len

    def make_mask(
        self, N: int, return_array: bool = False, window_size: Optional[int] = None
    ):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(N, self.offset, return_array, window_size)

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        total = 0
        for page in self.storage.pages:
            valid = page.valid_blocks
            total += int(
                page.k_codes[:, :, :valid, :, :].size * page.k_codes.itemsize
                + page.k_scales[:, :, :valid, :, :].size * page.k_scales.itemsize
                + page.v_codes[:, :, :valid, :, :].size * page.v_codes.itemsize
                + page.v_scales[:, :, :valid, :, :].size * page.v_scales.itemsize
            )
        if self.partial_k_buffer is not None:
            total += int(
                self.partial_k_buffer.size * self.partial_k_buffer.itemsize
                + self.partial_v_buffer.size * self.partial_v_buffer.itemsize
            )
        return total
