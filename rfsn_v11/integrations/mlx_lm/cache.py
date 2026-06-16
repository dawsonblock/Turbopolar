"""MLX-LM cache with fused Metal attention for TurboPolar decode."""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.kernels.turbo_polar.execution import (
    ExecutionMode,
    MetalExecutionRequiredError,
    TraceValidationMode,
)
from rfsn_v11.generation.turbo_polar_cache import TurboPolarKVCacheRuntime
from rfsn_v11.integrations.mlx_lm.telemetry import KernelExecutionStats
from rfsn_v11.kernels.turbo_polar.metal import MetalKernelBridge
from rfsn_v11.evidence.execution_trace import (
    ExecutionTraceCollector,
    AttentionExecutionTrace,
    KernelOperationTrace,
)
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.qjl.encoder import QJLResidualEncoder
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer


class TurboPolarFastCache:
    """MLX-LM-compatible cache that uses fused Metal attention for decode."""

    def __init__(
        self,
        config: TurboPolarConfig,
        trace_collector: Optional[ExecutionTraceCollector] = None,
    ):
        self.config = config
        self.runtime = TurboPolarKVCacheRuntime(config)
        self.bridge = MetalKernelBridge()
        self.decoder = PolarQuantDecoder()
        self.v_dequantizer = GroupedVQuantizer(group_size=32)
        # Share the runtime's QJL projector so query signs match key
        # residual sketches.
        self.qjl_encoder: Optional[QJLResidualEncoder] = (
            self.runtime.qjl_encoder if config.use_qjl else None
        )
        self._trace_collector = (
            trace_collector
            if trace_collector is not None
            else ExecutionTraceCollector()
        )

    @property
    def offset(self) -> int:
        """Sequence length; used by mlx_lm RoPE for correct position."""
        return self.runtime.actual_seq_len

    def reset_execution_stats(self):
        """Reset process-global Metal kernel execution counters.

        Because MetalKernelBridge is a singleton, this resets counters for
        all caches in the process. Call once before an experiment.
        """
        self.bridge.reset_execution_stats()
        self._trace_collector.clear()

    def execution_stats(self) -> KernelExecutionStats:
        """Return process-global Metal kernel execution counters.

        These totals are identical for every cache in the process because
        all caches share one MetalKernelBridge singleton. Do not sum stats
        across multiple caches; read once from any cache or the bridge.
        """
        return self.bridge.execution_stats()

    def execution_traces(self) -> List[AttentionExecutionTrace]:
        """Return collected operation-level execution traces."""
        return self._trace_collector.snapshot()

    def clear_execution_traces(self) -> None:
        """Clear collected traces."""
        self._trace_collector.clear()

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
        packed = mx.sum(
            reshaped.astype(mx.uint8) * powers, axis=-1
        ).astype(mx.uint8)
        return packed

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Prefill: append keys/values and return decompressed full history."""
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
            raise RuntimeError(
                "TurboPolar cache returned no blocks after append"
            )

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
            raise ValueError(
                "decode_attention inputs must be 4-D (B, H, T, D)"
            )
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
                f"decode_attention head_dim {D} does not match "
                f"config {config.head_dim}"
            )
        if H_q % H_kv != 0:
            raise ValueError(
                f"decode_attention requires GQA ratio to divide "
                f"evenly, got {H_q} and {H_kv}"
            )

    def decode_attention(
        self,
        q: mx.array,
        k_new: mx.array,
        v_new: mx.array,
        scale: float,
        mask: Optional[mx.array] = None,
        *,
        layer_index: Optional[int] = None,
        decode_step: Optional[int] = None,
        decode_ordinal: Optional[int] = None,
        initial_context_length: Optional[int] = None,
        fixture_id: Optional[str] = None,
        experiment_id: str = "",
    ) -> mx.array:
        """Decode path: append one token and run fused Metal attention.

        Args:
            q: [B, H_q, 1, D] already RoPE'd query.
            k_new: [B, H_kv, 1, D] already RoPE'd key token.
            v_new: [B, H_kv, 1, D] already RoPE'd value token.
            scale: attention scale (typically 1/sqrt(head_dim)).
            mask: must be None in the supported configuration.
            layer_index: layer index for trace collection (optional).
            decode_step: decode step for trace collection (optional).
            decode_ordinal: fixture-local decode ordinal (optional).
            initial_context_length: initial context length for this fixture
                (optional).
            fixture_id: fixture identifier for trace collection (optional).
            experiment_id: experiment identifier for trace collection.

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
            raise RuntimeError(
                "TurboPolar cache returned no blocks after append"
            )

        q_squeezed = q.squeeze(2)  # [B, H_q, D]
        cfg = dataclasses.replace(self.config, attention_scale=scale)

        # Page-based online-softmax attention without full-cache
        # materialization.
        output, trace = self.bridge.execute_paged_online_attention(
            q_squeezed,
            view.pages,
            view.partial_k,
            view.partial_v,
            cfg,
            view.total_tokens,
            mode=cfg.execution_mode,
            trace_validation_mode=cfg.trace_validation_mode,
            tail_k_full=view.partial_k_full,
            tail_v_full=view.partial_v_full,
            tail_length=view.partial_length,
        )

        # In SYNCHRONOUS_EVIDENCE mode the bridge evaluates outputs
        # internally. In ASYNC_PERFORMANCE mode evaluation is deferred.
        synchronous = (
            cfg.trace_validation_mode
            is TraceValidationMode.SYNCHRONOUS_EVIDENCE
        )
        output_evaluated = (
            cfg.execution_mode is ExecutionMode.METAL_STRICT
            and synchronous
        )

        # Build and store operation-level trace if identity is provided.
        if layer_index is not None and decode_step is not None:
            # Use passed parameters for trace identity fields
            # If not provided, fall back to derived values
            # (for backward compatibility)
            context_length = (
                initial_context_length
                if initial_context_length is not None
                else (view.total_tokens - 1)
            )
            fixture_id = (
                fixture_id if fixture_id is not None else experiment_id
            )
            decode_ordinal = (
                decode_ordinal if decode_ordinal is not None else 0
            )
            cache_offset_before = decode_step
            cache_tokens_before = view.total_tokens - 1  # Before this append
            cache_tokens_after = view.total_tokens  # After this append
            partial_tail_length = (
                view.partial_k.shape[2]
                if view.partial_k is not None
                else 0
            )

            step_trace = self._build_attention_trace(
                trace=trace,
                view=view,
                layer_index=layer_index,
                decode_step=decode_step,
                experiment_id=experiment_id,
                execution_mode=cfg.execution_mode.value,
                output_evaluated=output_evaluated,
                context_length=context_length,
                fixture_id=fixture_id,
                decode_ordinal=decode_ordinal,
                cache_offset_before=cache_offset_before,
                cache_tokens_before=cache_tokens_before,
                cache_tokens_after=cache_tokens_after,
                partial_tail_length=partial_tail_length,
            )
            if synchronous:
                self._trace_collector.record(step_trace)
            else:
                self._trace_collector.record_provisional(step_trace)

        # Strict validation: exact page count, all Metal, zero fallback.
        # In SYNCHRONOUS_EVIDENCE mode outputs are evaluated inside the bridge.
        # In ASYNC_PERFORMANCE mode the caller is responsible for evaluation.
        if cfg.execution_mode is ExecutionMode.METAL_STRICT:
            page_traces_raw = trace.get("page_traces", [])
            expected_page_count = len(view.pages)
            if len(page_traces_raw) != expected_page_count:
                raise MetalExecutionRequiredError(
                    f"Missing page dispatch: expected {expected_page_count}, "
                    f"got {len(page_traces_raw)}"
                )
            if any(not pt.get("metal_used", False) for pt in page_traces_raw):
                raise MetalExecutionRequiredError(
                    "Non-Metal compressed page detected"
                )
            if trace.get("fallback_used"):
                raise MetalExecutionRequiredError(
                    f"Strict mode encountered fallback: "
                    f"{trace.get('fallback_reason', 'unknown')}"
                )
        return output

    def _build_attention_trace(
        self,
        trace: dict,
        view,
        layer_index: int,
        decode_step: int,
        experiment_id: str,
        execution_mode: str,
        output_evaluated: bool = False,
        context_length: int = 0,
        fixture_id: str = "",
        decode_ordinal: int = 0,
        cache_offset_before: int = 0,
        cache_tokens_before: int = 0,
        cache_tokens_after: int = 0,
        partial_tail_length: int = 0,
    ) -> AttentionExecutionTrace:
        """Build an AttentionExecutionTrace from the bridge execution trace."""
        page_traces_raw = trace.get("page_traces", [])
        expected_page_count = len(view.pages)

        page_traces: list[KernelOperationTrace] = []
        for page_idx, pt in enumerate(page_traces_raw):
            page_traces.append(
                KernelOperationTrace(
                    experiment_id=experiment_id,
                    context_length=context_length,
                    fixture_id=fixture_id,
                    layer_index=layer_index,
                    decode_step=decode_step,
                    decode_ordinal=decode_ordinal,
                    cache_offset_before=cache_offset_before,
                    operation="compressed_page",
                    page_index=page_idx,
                    kernel_name=pt.get("kernel_name", "unknown"),
                    execution_mode=execution_mode,
                    metal_requested=True,
                    metal_executed=pt.get("metal_used", False),
                    fallback_used=pt.get("fallback_used", False),
                    fallback_reason=pt.get("fallback_reason"),
                    expected_tokens=pt.get("actual_seq_len", 0),
                    processed_tokens=pt.get("actual_seq_len", 0),
                    output_evaluated=output_evaluated,
                )
            )

        dense_tail_metal = trace.get("dense_tail_metal", False)
        dense_tail_trace = None
        if view.partial_k is not None and view.partial_k.shape[2] > 0:
            dense_tail_trace = KernelOperationTrace(
                experiment_id=experiment_id,
                context_length=context_length,
                fixture_id=fixture_id,
                layer_index=layer_index,
                decode_step=decode_step,
                decode_ordinal=decode_ordinal,
                cache_offset_before=cache_offset_before,
                operation="dense_tail",
                page_index=None,
                kernel_name="dense_tail_raw"
                if dense_tail_metal
                else "dense_tail_reference",
                execution_mode=execution_mode,
                metal_requested=True,
                metal_executed=dense_tail_metal,
                fallback_used=(
                    not dense_tail_metal and trace.get("fallback_used", False)
                ),
                fallback_reason=(
                    trace.get("fallback_reason")
                    if not dense_tail_metal
                    else None
                ),
                expected_tokens=view.partial_k.shape[2],
                processed_tokens=view.partial_k.shape[2],
                output_evaluated=output_evaluated,
            )

        return AttentionExecutionTrace(
            experiment_id=experiment_id,
            context_length=context_length,
            fixture_id=fixture_id,
            layer_index=layer_index,
            decode_step=decode_step,
            decode_ordinal=decode_ordinal,
            cache_offset_before=cache_offset_before,
            cache_tokens_before=cache_tokens_before,
            cache_tokens_after=cache_tokens_after,
            partial_tail_length=partial_tail_length,
            expected_page_count=expected_page_count,
            page_traces=page_traces,
            dense_tail_trace=dense_tail_trace,
        )

    def commit_provisional_traces(self, output_evaluated: bool = True) -> None:
        """Commit all provisional traces held by the shared collector."""
        self._trace_collector.commit_provisional(
            output_evaluated=output_evaluated
        )

    def clear_provisional_traces(self) -> None:
        """Discard all provisional traces without committing them."""
        self._trace_collector.clear_provisional()

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: Optional[int] = None,
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
        """Run decode_attention and return (peak allocator bytes, output)."""
        mx.reset_peak_memory()
        output = self.decode_attention(q, k_new, v_new, scale, mask=mask)
        mx.eval(output)
        self.runtime._eval_state()
        return int(mx.get_peak_memory()), output


def make_turbo_caches(
    num_layers: int,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    use_qjl: Optional[bool] = None,
    execution_mode: Optional[ExecutionMode] = None,
    trace_validation_mode: Optional[TraceValidationMode] = None,
    experiment_id: str = "",
    config: Optional[TurboPolarConfig] = None,
) -> List[TurboPolarFastCache]:
    """Create TurboPolarFastCache layers with benchmark-quality defaults.

    If ``config`` is provided, the exact immutable config is cloned with
    ``dataclasses.replace()`` for any fields explicitly overridden.
    Otherwise a new default config is constructed from the individual fields.
    """
    if config is not None:
        overrides: Dict[str, Any] = {}
        if num_q_heads is not None:
            overrides["num_q_heads"] = num_q_heads
        if num_kv_heads is not None:
            overrides["num_kv_heads"] = num_kv_heads
        if head_dim is not None:
            overrides["head_dim"] = head_dim
        if use_qjl is not None:
            overrides["use_qjl"] = use_qjl
        if execution_mode is not None:
            overrides["execution_mode"] = execution_mode
        if trace_validation_mode is not None:
            overrides["trace_validation_mode"] = trace_validation_mode
        config = dataclasses.replace(config, **overrides)
    else:
        if head_dim is None:
            raise ValueError(
                "head_dim is required when config is not provided"
            )
        config = TurboPolarConfig(
            num_q_heads=num_q_heads or 32,
            num_kv_heads=num_kv_heads or 8,
            head_dim=head_dim,
            block_size=64,
            qjl_proj_dim=64,
            use_qjl=use_qjl if use_qjl is not None else False,
            storage_mode="kv_quant",
            use_int8_radii=True,
            k_angle_bits_deep=8,
            split_dim=0,
            execution_mode=(
                execution_mode
                if execution_mode is not None
                else ExecutionMode.DEVELOPMENT_AUTO
            ),
            trace_validation_mode=(
                trace_validation_mode
                if trace_validation_mode is not None
                else TraceValidationMode.SYNCHRONOUS_EVIDENCE
            ),
        )
    shared_collector = ExecutionTraceCollector(
        experiment_id=experiment_id
    )
    return [
        TurboPolarFastCache(config, trace_collector=shared_collector)
        for _ in range(num_layers)
    ]
