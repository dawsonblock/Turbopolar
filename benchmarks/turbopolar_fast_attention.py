"""Fused TurboPolar attention integration for mlx_lm Llama models.

Decode steps bypass K/V decompression and use custom Metal kernels.
Prefill steps fall back to decompression + standard MLX attention.
"""

import dataclasses
from typing import Any, List, Optional, Tuple

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer


class TurboPolarFastCache:
    """MLX-LM-compatible cache that uses fused Metal attention for decode."""

    def __init__(self, config: TurboPolarConfig):
        self.config = config
        self.runtime = TurboPolarKVCacheRuntime(config)
        self.bridge = MetalKernelBridge()
        self.decoder = PolarQuantDecoder()
        self.v_dequantizer = GroupedVQuantizer(group_size=32)
        # Share the runtime's QJL projector so query signs match key residual sketches.
        self.qjl_encoder: Optional[QJLResidualEncoder] = (
            self.runtime.qjl_encoder if config.use_qjl else None
        )

    @property
    def offset(self) -> int:
        """Sequence length; used by mlx_lm RoPE to apply the correct position."""
        return self.runtime.actual_seq_len

    def _compute_qjl_signs(self, q: mx.array) -> mx.array:
        """Pack query projection signs to match the kernel's bit-packed layout.

        Args:
            q: [B, H_q, D] single-token query after RoPE.

        Returns:
            [B, H_q, proj_dim // 8] packed uint8 signs.
        """
        proj = mx.matmul(q, self.qjl_encoder.W)  # [B, H_q, proj_dim]
        signs = proj >= 0
        B, H_q, P = signs.shape
        reshaped = signs.reshape(B, H_q, P // 8, 8)
        powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)
        packed = mx.sum(reshaped.astype(mx.uint8) * powers, axis=-1).astype(mx.uint8)
        return packed

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Prefill path: append keys/values and return the decompressed full history."""
        original_dtype = keys.dtype
        if keys.dtype != mx.float16:
            keys = keys.astype(mx.float16)
        if values.dtype != mx.float16:
            values = values.astype(mx.float16)

        self.runtime.append(keys, values)
        block, quant_v, dense_v, _qjl, actual_len = self.runtime.get_blocks_for_attention()
        if block is None:
            raise RuntimeError("TurboPolar cache returned no blocks after append")

        k_dense = self.decoder.decode_block(block)[:, :, :actual_len, :]

        B, H_kv, S, L, _ = block.radii.shape
        if dense_v is not None:
            v_full = dense_v.reshape(B, H_kv, S * L, self.config.head_dim)
        elif quant_v is not None:
            v_full = self.v_dequantizer.dequantize_block(quant_v).reshape(
                B, H_kv, S * L, self.config.head_dim
            )
        else:
            raise RuntimeError("TurboPolar cache has no V payload")
        v_dense = v_full[:, :, :actual_len, :]

        if original_dtype != k_dense.dtype:
            k_dense = k_dense.astype(original_dtype)
            v_dense = v_dense.astype(original_dtype)

        return k_dense, v_dense

    def decode_attention(
        self,
        q: mx.array,
        k_new: mx.array,
        v_new: mx.array,
        scale: float,
    ) -> mx.array:
        """Decode path: append one token and run fused Metal attention.

        Args:
            q: [B, H_q, 1, D] already RoPE'd query.
            k_new: [B, H_kv, 1, D] already RoPE'd key token.
            v_new: [B, H_kv, 1, D] already RoPE'd value token.
            scale: attention scale (typically 1/sqrt(head_dim)).

        Returns:
            [B, H_q, D] attention output.
        """
        B, H_q, T, D = q.shape
        if T != 1:
            raise ValueError("decode_attention only supports a single query token")
        _, H_kv, T_k, _ = k_new.shape
        if T_k != 1:
            raise ValueError("decode_attention only supports a single key/value token")
        _, _H_v, T_v, _ = v_new.shape
        if T_v != 1:
            raise ValueError("decode_attention only supports a single key/value token")

        if k_new.dtype != mx.float16:
            k_new = k_new.astype(mx.float16)
        if v_new.dtype != mx.float16:
            v_new = v_new.astype(mx.float16)

        self.runtime.append(k_new, v_new)

        block, quant_v, _dense_v, qjl_payload, actual_seq_len = (
            self.runtime.get_blocks_for_attention()
        )
        if block is None:
            raise RuntimeError("TurboPolar cache returned no blocks after append")
        if quant_v is None:
            raise RuntimeError("Fused attention requires kv_quant storage mode")

        q_squeezed = q.squeeze(2)  # [B, H_q, D]

        q_proj_signs = None
        if self.config.use_qjl:
            if qjl_payload is None:
                raise RuntimeError("use_qjl=True but runtime produced no QJL payload")
            q_proj_signs = self._compute_qjl_signs(q_squeezed)

        cfg = dataclasses.replace(self.config, attention_scale=scale)
        output, _ = self.bridge.execute_online_attention_quant_v(
            q_squeezed,
            block,
            quant_v,
            qjl_payload,
            q_proj_signs,
            cfg,
            actual_seq_len,
            self.config.use_qjl,
        )
        return output

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


def make_turbo_caches(
    num_layers: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    use_qjl: bool = False,
) -> List[TurboPolarFastCache]:
    """Create a list of TurboPolarFastCache layers with benchmark-quality defaults."""
    if head_dim not in (64, 128):
        raise ValueError(f"TurboPolar only supports head_dim 64 or 128, got {head_dim}")
    config = TurboPolarConfig(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=64,
        qjl_proj_dim=32 if head_dim == 64 else 64,
        use_qjl=use_qjl,
        storage_mode="kv_quant",
        use_int8_radii=True,
        k_angle_bits_deep=8,
        split_dim=0,
    )
    return [TurboPolarFastCache(config) for _ in range(num_layers)]


_orig_llama_attention_call: Optional[Any] = None


def _turbo_attention_call(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
):
    if isinstance(cache, TurboPolarFastCache) and x.shape[1] == 1:
        B, L, D = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)

        output = cache.decode_attention(queries, keys, values, self.scale)
        # output: [B, H_q, D] -> [B, L, H_q * D]
        output = output[:, None, :, :].transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    return _orig_llama_attention_call(self, x, mask=mask, cache=cache)


def patch_llama_attention(model) -> None:
    """Monkey-patch mlx_lm Llama Attention to use TurboPolarFastCache for decode."""
    import mlx_lm.models.llama as llama_module

    global _orig_llama_attention_call
    if _orig_llama_attention_call is None:
        _orig_llama_attention_call = llama_module.Attention.__call__
    llama_module.Attention.__call__ = _turbo_attention_call


def unpatch_llama_attention(model) -> None:
    """Restore original mlx_lm Llama Attention forward."""
    import mlx_lm.models.llama as llama_module

    global _orig_llama_attention_call
    if _orig_llama_attention_call is not None:
        llama_module.Attention.__call__ = _orig_llama_attention_call
        _orig_llama_attention_call = None
