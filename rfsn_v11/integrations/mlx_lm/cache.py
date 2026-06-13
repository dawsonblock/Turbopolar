"""MLX-LM-compatible cache that uses fused Metal attention for TurboPolar decode."""

import dataclasses
from typing import List, Optional, Tuple

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.kernels.turbo_polar.execution import ExecutionMode
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.integrations.mlx_lm.telemetry import KernelExecutionStats
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

    def reset_execution_stats(self):
        """Reset Metal kernel execution counters."""
        self.bridge.reset_execution_stats()

    def execution_stats(self) -> KernelExecutionStats:
        """Return Metal kernel execution counters."""
        return self.bridge.execution_stats()

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

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Prefill path: append keys/values and return the decompressed full history."""
        original_dtype = keys.dtype
        if keys.dtype != mx.float16:
            keys = keys.astype(mx.float16)
        if values.dtype != mx.float16:
            values = values.astype(mx.float16)

        self.runtime.append_many(keys, values)
        block, quant_v, dense_v, _qjl, actual_len = (
            self.runtime.get_blocks_for_attention()
        )
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

    @staticmethod
    def _validate_decode_shape(
        q: mx.array, k_new: mx.array, v_new: mx.array, config: TurboPolarConfig
    ):
        if q.ndim != 4 or k_new.ndim != 4 or v_new.ndim != 4:
            raise ValueError("decode_attention inputs must be 4-D (B, H, T, D)")
        B, H_q, T, D = q.shape
        _, H_kv, T_k, _ = k_new.shape
        _, _, T_v, _ = v_new.shape
        if B != 1:
            raise NotImplementedError(
                "TurboPolar fused decode only supports batch size 1."
            )
        if T != 1 or T_k != 1 or T_v != 1:
            raise ValueError(
                "decode_attention only supports a single query/key/value token"
            )
        if D != 128:
            raise NotImplementedError(
                "TurboPolar fused decode only supports head_dim == 128."
            )
        if D != config.head_dim:
            raise ValueError(
                f"decode_attention head_dim {D} does not match config {config.head_dim}"
            )
        if H_q % H_kv != 0:
            raise ValueError(
                f"decode_attention requires GQA ratio to divide evenly, got {H_q} and {H_kv}"
            )

    def decode_attention(
        self,
        q: mx.array,
        k_new: mx.array,
        v_new: mx.array,
        scale: float,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Decode path: append one token and run fused Metal attention.

        Args:
            q: [B, H_q, 1, D] already RoPE'd query.
            k_new: [B, H_kv, 1, D] already RoPE'd key token.
            v_new: [B, H_kv, 1, D] already RoPE'd value token.
            scale: attention scale (typically 1/sqrt(head_dim)).
            mask: must be None in the supported configuration.

        Returns:
            [B, H_q, D] attention output.
        """
        self._validate_decode_shape(q, k_new, v_new, self.config)

        if mask is not None:
            raise NotImplementedError(
                "TurboPolar fused decode currently supports mask=None only."
            )

        if k_new.dtype != mx.float16:
            k_new = k_new.astype(mx.float16)
        if v_new.dtype != mx.float16:
            v_new = v_new.astype(mx.float16)

        self.runtime.append(k_new, v_new)

        view = self.runtime.attention_view()
        if not view.pages and view.partial_k is None:
            raise RuntimeError("TurboPolar cache returned no blocks after append")

        q_squeezed = q.squeeze(2)  # [B, H_q, D]
        cfg = dataclasses.replace(self.config, attention_scale=scale)

        # Page-based online-softmax attention without full-cache materialization.
        output, trace = self.bridge.execute_paged_online_attention(
            q_squeezed,
            view.pages,
            view.partial_k,
            view.partial_v,
            cfg,
            view.total_tokens,
            mode=cfg.execution_mode,
        )
        if trace.get("fallback_used") and cfg.execution_mode is ExecutionMode.METAL_STRICT:
            from rfsn_v11.kernels.turbo_polar.execution import MetalExecutionRequiredError
            raise MetalExecutionRequiredError(
                f"Strict mode encountered fallback: {trace.get('fallback_reason', 'unknown')}"
            )
        return output

    def make_mask(
        self, N: int, return_array: bool = False, window_size: Optional[int] = None
    ):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(N, self.offset, return_array, window_size)

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        stats = self.runtime.get_memory_stats()
        return int(stats.allocated_capacity_bytes)

    def empty(self) -> bool:
        return self.offset == 0

    def get_memory_stats(self):
        """Return truthful memory accounting for the underlying cache."""
        return self.runtime.get_memory_stats()

    def measure_decode_peak_memory(
        self,
        q: mx.array,
        k_new: mx.array,
        v_new: mx.array,
        scale: float,
        mask: Optional[mx.array] = None,
    ) -> Tuple[int, mx.array]:
        """Run decode_attention and return (peak MLX allocator bytes, output)."""
        mx.reset_peak_memory()
        output = self.decode_attention(q, k_new, v_new, scale, mask=mask)
        mx.eval(output)
        self.runtime._eval_state()
        return int(mx.get_peak_memory()), output


def make_turbo_caches(
    num_layers: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    use_qjl: bool = False,
    execution_mode: Optional[ExecutionMode] = None,
) -> List[TurboPolarFastCache]:
    """Create a list of TurboPolarFastCache layers with benchmark-quality defaults."""
    if head_dim != 128:
        raise ValueError(
            f"TurboPolar fused MLX path only supports head_dim=128, got {head_dim}"
        )
    config = TurboPolarConfig(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=64,
        qjl_proj_dim=64,
        use_qjl=use_qjl,
        storage_mode="kv_quant",
        use_int8_radii=True,
        k_angle_bits_deep=8,
        split_dim=0,
        execution_mode=execution_mode if execution_mode is not None else ExecutionMode.DEVELOPMENT_AUTO,
    )
    return [TurboPolarFastCache(config) for _ in range(num_layers)]
