from dataclasses import dataclass
from importlib.resources import files
import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from rfsn_v11.kernels.turbo_polar.execution import (
    ExecutionMode,
    MetalExecutionRequiredError,
    MetalKernelDispatchError,
    MetalKernelInitializationError,
)
from rfsn_v11.quant.polar.decoder import PolarQuantDecoder
from rfsn_v11.quant.polar.payload import PolarKeyBlock
from rfsn_v11.quant.qjl.encoder import QJLPayload
from rfsn_v11.quant.qjl.score_estimate import qjl_dot_estimate
from rfsn_v11.quant.v_quant.encoder import GroupedVQuantizer, QuantizedVBlock


@dataclass
class KernelExecutionStats:
    fused_qk_calls: int = 0
    online_attention_calls: int = 0
    dense_tail_calls: int = 0
    fallback_calls: int = 0
    compressed_page_dispatches: int = 0
    dense_tail_dispatches: int = 0


def _probe_metal_dispatch() -> Tuple[bool, str]:
    """
    Detect how the current MLX/Metal runtime interprets the `grid` argument.

    MLX 0.31.2 treats `grid` as the *total* number of threads in each dimension
    and subdivides them into threadgroups of size `threadgroup`. Earlier/later
    versions may treat `grid` as the number of threadgroups. We probe both
    interpretations so the launch dimensions can be adjusted accordingly.

    Returns:
        (supported, semantics) where semantics is "threadgroups" or "total_threads".
    """
    source = """
    uint tid = thread_index_in_threadgroup;
    out[tid] = float(tid) + 1.0h;
    """
    try:
        k = mx.fast.metal_kernel(
            name="turbo_polar_tg_probe",
            input_names=[],
            output_names=["out"],
            source=source,
        )

        # Standard Metal semantics: grid = threadgroups, threadgroup = threads/group.
        # grid=(2,1,1), tg=(32,1,1) -> 2 threadgroups of 32 threads -> tid=31 runs.
        out_tg = k(
            inputs=[],
            output_shapes=[(32,)],
            output_dtypes=[mx.float16],
            grid=(2, 1, 1),
            threadgroup=(32, 1, 1),
        )
        arr_tg = np.array(out_tg[0])
        if len(arr_tg) >= 32 and float(arr_tg[31]) == 32.0:
            return True, "threadgroups"

        # MLX 0.31.2 semantics: grid = total threads, subdivided into threadgroups.
        # grid=(32,1,1), tg=(32,1,1) -> 1 threadgroup of 32 threads -> tid=31 runs.
        out_tt = k(
            inputs=[],
            output_shapes=[(32,)],
            output_dtypes=[mx.float16],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
        )
        arr_tt = np.array(out_tt[0])
        if len(arr_tt) >= 32 and float(arr_tt[31]) == 32.0:
            return True, "total_threads"

        return False, "unknown"
    except Exception:
        return False, "unknown"


class MetalKernelBridge:
    """
    Unified compilation & execution bridge for TurboPolar Metal shaders.
    Handles GQA via num_queries_per_kv parameter and bit-packed angle codes.

    MLX's mx.fast.metal_kernel expects ``source`` to be the body of a function;
    the function signature is generated automatically. Helper code (includes,
    namespace declarations, inline functions) is passed via the ``header`` argument.
    The extracted body is injected with local aliases for the thread identifiers
    that the original kernel signature removed.

    Implemented as a process-wide singleton because ``mx.fast.metal_kernel``
    appears to cache compiled kernels and outputs by name; creating multiple
    instances with identical names can return stale results in MLX 0.31.2.
    """

    _instance: "MetalKernelBridge | None" = None
    _initialized: bool = False

    def __new__(cls, source_dir: Path | None = None) -> "MetalKernelBridge":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, source_dir: Path | None = None):
        # Early exit for the singleton: __init__ is invoked on every
        # MetalKernelBridge() call even when __new__ returns the existing
        # instance, so avoid re-parsing shaders and re-compiling kernels.
        if MetalKernelBridge._initialized:
            return
        self._initialize(source_dir)

    def _initialize(self, source_dir: Path | None = None) -> None:
        """Atomic initialization; all kernels built or none."""
        if MetalKernelBridge._initialized:
            return
        try:
            self._build_kernels(source_dir)
        except Exception as _exc:
            MetalKernelBridge._initialized = False
            self._kernel_qk = None
            self._kernel_qk_qjl = None
            self._kernel_attn_dense = None
            self._kernel_attn_quant = None
            self._kernel_attn_quant_dt = None
            self._kernel_attn_quant_raw = None
            self._kernel_attn_quant_dense_tail = None
            self._kernel_dense_tail_raw = None
            self.threadgroup_supported = False
            self.grid_semantics = "unknown"
            self._tg_x = 32
            self._stats = KernelExecutionStats()
            raise MetalKernelInitializationError(
                f"TurboPolar Metal initialization failed: {_exc}"
            ) from _exc
        MetalKernelBridge._initialized = True

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset singleton state so tests get a fresh bridge."""
        cls._instance = None
        cls._initialized = False

    def _build_kernels(self, source_dir: Path | None = None) -> None:
        if source_dir is None:
            kernel_dir = files("rfsn_v11.kernels.turbo_polar")
            qk_source = kernel_dir.joinpath("tqpolar_fused_qk.metal").read_text()
            attn_source = kernel_dir.joinpath(
                "tqpolar_online_attention.metal"
            ).read_text()
        else:
            qk_path = source_dir / "tqpolar_fused_qk.metal"
            attn_path = source_dir / "tqpolar_online_attention.metal"
            if not qk_path.exists():
                raise FileNotFoundError(f"Missing QK shader: {qk_path}")
            if not attn_path.exists():
                raise FileNotFoundError(f"Missing attention shader: {attn_path}")
            qk_source = qk_path.read_text()
            attn_source = attn_path.read_text()

        qk_header, qk_body = self._extract_kernel_parts(
            qk_source, "tqpolar_fused_dequant_qk"
        )
        _kernel_qk = mx.fast.metal_kernel(
            name="tqpolar_fused_dequant_qk",
            input_names=[
                "q",
                "polar_radii",
                "polar_radii_i8",
                "radii_scales",
                "angle_codes_l1",
                "angle_codes_deep",
                "head_dim",
                "split_dim",
                "block_size",
                "l1_scale",
                "deep_scale",
                "attention_scale",
                "num_queries_per_kv",
                "int8_radii",
                "log_radii",
                "l1_bits",
                "deep_bits",
                "strides",
            ],
            output_names=["scores"],
            header=qk_header,
            source=qk_body,
        )

        qk_qjl_header, qk_qjl_body = self._extract_kernel_parts(
            qk_source, "tqpolar_fused_dequant_qk_qjl"
        )
        _kernel_qk_qjl = mx.fast.metal_kernel(
            name="tqpolar_fused_dequant_qk_qjl",
            input_names=[
                "q",
                "polar_radii",
                "polar_radii_i8",
                "radii_scales",
                "angle_codes_l1",
                "angle_codes_deep",
                "qjl_packed_signs",
                "qjl_norms",
                "q_proj_signs",
                "head_dim",
                "split_dim",
                "block_size",
                "qjl_proj_dim",
                "l1_scale",
                "deep_scale",
                "attention_scale",
                "num_queries_per_kv",
                "int8_radii",
                "log_radii",
                "l1_bits",
                "deep_bits",
                "strides",
            ],
            output_names=["scores"],
            header=qk_qjl_header,
            source=qk_qjl_body,
        )

        attn_dense_header, attn_dense_body = self._extract_kernel_parts(
            attn_source, "tqpolar_online_attention_dense_v"
        )
        _kernel_attn_dense = mx.fast.metal_kernel(
            name="tqpolar_online_attention_dense_v",
            input_names=[
                "q",
                "polar_radii",
                "polar_radii_i8",
                "radii_scales",
                "angle_codes_l1",
                "angle_codes_deep",
                "v_dense",
                "qjl_packed_signs",
                "qjl_norms",
                "q_proj_signs",
                "head_dim",
                "split_dim",
                "block_size",
                "total_blocks",
                "qjl_proj_dim",
                "use_qjl",
                "l1_scale",
                "deep_scale",
                "attention_scale",
                "int8_radii",
                "log_radii",
                "l1_bits",
                "deep_bits",
                "strides",
                "actual_seq_len",
                "num_queries_per_kv",
            ],
            output_names=["output"],
            header=attn_dense_header,
            source=attn_dense_body,
        )

        attn_quant_header, attn_quant_body = self._extract_kernel_parts(
            attn_source, "tqpolar_online_attention_quant_v"
        )
        _kernel_attn_quant = mx.fast.metal_kernel(
            name="tqpolar_online_attention_quant_v",
            input_names=[
                "q",
                "polar_radii",
                "polar_radii_i8",
                "radii_scales",
                "angle_codes_l1",
                "angle_codes_deep",
                "v_codes",
                "v_scales",
                "qjl_packed_signs",
                "qjl_norms",
                "q_proj_signs",
                "head_dim",
                "split_dim",
                "block_size",
                "total_blocks",
                "qjl_proj_dim",
                "group_size",
                "use_qjl",
                "l1_scale",
                "deep_scale",
                "attention_scale",
                "int8_radii",
                "log_radii",
                "l1_bits",
                "deep_bits",
                "strides",
                "actual_seq_len",
                "num_queries_per_kv",
            ],
            output_names=["output"],
            header=attn_quant_header,
            source=attn_quant_body,
        )

        attn_quant_raw_header, attn_quant_raw_body = self._extract_kernel_parts(
            attn_source, "tqpolar_online_attention_quant_v_raw"
        )
        _kernel_attn_quant_raw = mx.fast.metal_kernel(
            name="tqpolar_online_attention_quant_v_raw",
            input_names=[
                "q",
                "polar_radii",
                "polar_radii_i8",
                "radii_scales",
                "angle_codes_l1",
                "angle_codes_deep",
                "v_codes",
                "v_scales",
                "qjl_packed_signs",
                "qjl_norms",
                "q_proj_signs",
                "head_dim",
                "split_dim",
                "block_size",
                "total_blocks",
                "qjl_proj_dim",
                "group_size",
                "use_qjl",
                "l1_scale",
                "deep_scale",
                "attention_scale",
                "int8_radii",
                "log_radii",
                "l1_bits",
                "deep_bits",
                "strides",
                "actual_seq_len",
                "num_queries_per_kv",
            ],
            output_names=["out_weighted", "out_max_score", "out_exp_sum"],
            header=attn_quant_raw_header,
            source=attn_quant_raw_body,
        )

        attn_quant_dt_header, attn_quant_dt_body = self._extract_kernel_parts(
            attn_source, "tqpolar_online_attention_quant_v_dense_tail"
        )
        _kernel_attn_quant_dense_tail = mx.fast.metal_kernel(
            name="tqpolar_online_attention_quant_v_dense_tail",
            input_names=[
                "q",
                "polar_radii",
                "polar_radii_i8",
                "radii_scales",
                "angle_codes_l1",
                "angle_codes_deep",
                "v_codes",
                "v_scales",
                "tail_k",
                "tail_v",
                "qjl_packed_signs",
                "qjl_norms",
                "q_proj_signs",
                "constants",
                "l1_scale",
                "deep_scale",
                "attention_scale",
                "strides",
            ],
            output_names=["output"],
            header=attn_quant_dt_header,
            source=attn_quant_dt_body,
        )

        dense_tail_raw_header, dense_tail_raw_body = self._extract_kernel_parts(
            attn_source, "tqpolar_dense_tail_state_raw"
        )
        _kernel_dense_tail_raw = mx.fast.metal_kernel(
            name="tqpolar_dense_tail_state_raw",
            input_names=[
                "q",
                "tail_k",
                "tail_v",
                "head_dim",
                "tail_length",
                "attention_scale",
                "num_queries_per_kv",
                "strides",
            ],
            output_names=["out_weighted", "out_max_score", "out_exp_sum"],
            header=dense_tail_raw_header,
            source=dense_tail_raw_body,
        )

        _threadgroup_supported, _grid_semantics = _probe_metal_dispatch()

        # Atomic assignment: all kernels built successfully or none are retained.
        self._kernel_qk = _kernel_qk
        self._kernel_qk_qjl = _kernel_qk_qjl
        self._kernel_attn_dense = _kernel_attn_dense
        self._kernel_attn_quant = _kernel_attn_quant
        self._kernel_attn_quant_raw = _kernel_attn_quant_raw
        self._kernel_attn_quant_dense_tail = _kernel_attn_quant_dense_tail
        self._kernel_dense_tail_raw = _kernel_dense_tail_raw
        self.threadgroup_supported = _threadgroup_supported
        self.grid_semantics = _grid_semantics
        self._tg_x = 32
        self._stats = KernelExecutionStats()

    def reset_execution_stats(self):
        self._stats = KernelExecutionStats()

    def execution_stats(self) -> KernelExecutionStats:
        return self._stats

    @staticmethod
    def _extract_kernel_parts(source: str, kernel_name: str) -> Tuple[str, str]:
        start = source.find(f"kernel void {kernel_name}")
        if start == -1:
            raise ValueError(f"Kernel {kernel_name} not found in source")

        # Everything before the kernel signature belongs in the header (helpers,
        # namespace declarations, inline functions). Strip #include directives
        # because MLX injects its own metal_stdlib include context.
        header = source[:start]
        header = "\n".join(
            line
            for line in header.splitlines()
            if not line.strip().startswith("#include")
        ).strip()

        brace_start = source.find("{", start)
        if brace_start == -1:
            raise ValueError(f"Could not find body start for {kernel_name}")
        depth = 1
        i = brace_start + 1
        while i < len(source) and depth > 0:
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
            i += 1
        body = source[brace_start + 1 : i - 1].strip()

        # The original kernel signature is removed by extraction. Re-create the
        # identifiers it provided using the built-in variables MLX generates.
        aliases = (
            "uint3 tgid = threadgroup_position_in_grid;\n"
            "uint tid = thread_index_in_threadgroup;"
        )
        body = aliases + "\n" + body
        return header, body

    @staticmethod
    def _contiguous_strides(shape: tuple) -> list:
        """Return row-major strides in elements (matches shader pointer indexing)."""
        strides = [0] * len(shape)
        stride = 1
        for i in range(len(shape) - 1, -1, -1):
            strides[i] = stride
            stride *= int(shape[i])
        return strides

    def _ensure_contiguous(self, *arrays):
        return [mx.contiguous(a) for a in arrays]

    def _compute_grid(self, b_dim: int, h_dim: int, s_dim: int = 1) -> tuple:
        """
        Compute the MLX `grid` dimensions for a kernel that expects
        `threadgroup_position_in_grid` to enumerate (b, h, s) tiles.

        MLX 0.31.2 interprets `grid` as total threads and subdivides them into
        threadgroups, so we must multiply the batch dimension by the threadgroup
        size in x. Versions that treat `grid` as threadgroups use the raw dims.
        """
        if self.grid_semantics == "total_threads":
            b_dim *= self._tg_x
        return (b_dim, h_dim, s_dim)

    # ------------------------------------------------------------------
    # CPU / MLX fallback paths
    # These are used when the Metal runtime does not dispatch threads inside
    # a threadgroup (e.g. certain MLX versions), and as a debug reference.
    # ------------------------------------------------------------------
    def _cpu_fused_qk(self, q: mx.array, block: PolarKeyBlock, config) -> mx.array:
        k_recon = PolarQuantDecoder().decode_block(block)
        H_kv = block.radii.shape[1]
        H_q = q.shape[1]
        num_queries_per_kv = H_q // H_kv
        k_broadcast = mx.repeat(k_recon, num_queries_per_kv, axis=1)
        return mx.sum(q[:, :, None, :] * k_broadcast, axis=-1) * config.attention_scale

    def _cpu_fused_qk_qjl(
        self,
        q: mx.array,
        block: PolarKeyBlock,
        qjl_payload: QJLPayload,
        q_proj_signs: mx.array,
        config,
    ) -> mx.array:
        scores = self._cpu_fused_qk(q, block, config)
        qjl_corr = qjl_dot_estimate(q, qjl_payload, q_proj_signs)
        return scores + qjl_corr * config.attention_scale

    def _cpu_online_attention(
        self,
        q: mx.array,
        block: PolarKeyBlock,
        v: mx.array,
        qjl_payload: QJLPayload,
        q_proj_signs: mx.array,
        config,
        actual_seq_len: int,
        use_qjl: bool,
        quant_v_used: bool,
    ) -> Tuple[mx.array, Dict[str, Any]]:
        scores = self._cpu_fused_qk(q, block, config)
        if use_qjl:
            qjl_corr = qjl_dot_estimate(q, qjl_payload, q_proj_signs)
            scores = scores + qjl_corr * config.attention_scale

        B, H_q, T = scores.shape
        seq_mask = mx.arange(T) < actual_seq_len
        scores = mx.where(
            seq_mask[None, None, :], scores, mx.array(-1e9, dtype=scores.dtype)
        )
        weights = mx.softmax(scores, axis=-1)
        output = mx.sum(weights[:, :, :, None] * v, axis=-2)
        trace = {
            "kernel_name": "cpu_online_attention",
            "metal_used": False,
            "fallback_used": True,
            "qjl_used": use_qjl,
            "quant_v_used": quant_v_used,
            "actual_seq_len": actual_seq_len,
            "num_queries_per_kv": H_q // block.radii.shape[1],
        }
        return output, trace

    def _cpu_online_attention_dense_tail(
        self,
        q: mx.array,
        block: PolarKeyBlock,
        quant_v: QuantizedVBlock,
        tail_k: mx.array,
        tail_v: mx.array,
        qjl_payload: QJLPayload,
        q_proj_signs: mx.array,
        config,
        actual_seq_len: int,
        use_qjl: bool,
    ) -> Tuple[mx.array, Dict[str, Any]]:
        v_dequant = GroupedVQuantizer(group_size=quant_v.group_size).dequantize_block(
            quant_v
        )
        k_comp = PolarQuantDecoder().decode_block(block)
        B, H_kv, S, L, _ = block.radii.shape
        num_queries_per_kv = q.shape[1] // H_kv
        k_comp_flat = k_comp.reshape(B, H_kv, S * L, config.head_dim)
        v_comp_flat = v_dequant.reshape(B, H_kv, S * L, config.head_dim)
        full_k = mx.concatenate([k_comp_flat, tail_k], axis=2)
        full_v = mx.concatenate([v_comp_flat, tail_v], axis=2)
        full_k = mx.repeat(full_k, num_queries_per_kv, axis=1)
        full_v = mx.repeat(full_v, num_queries_per_kv, axis=1)
        scores = mx.sum(q[:, :, None, :] * full_k, axis=-1) * config.attention_scale
        if use_qjl:
            comp_len = S * L
            tail_len = tail_k.shape[2]
            qjl_corr = qjl_dot_estimate(q, qjl_payload, q_proj_signs)
            qjl_corr = (
                qjl_corr.reshape(B, q.shape[1], comp_len) * config.attention_scale
            )
            qjl_corr = mx.concatenate(
                [qjl_corr, mx.zeros((B, q.shape[1], tail_len), dtype=qjl_corr.dtype)],
                axis=2,
            )
            scores = scores + qjl_corr
        T = full_k.shape[2]
        seq_mask = mx.arange(T) < actual_seq_len
        scores = mx.where(
            seq_mask[None, None, :], scores, mx.array(-1e9, dtype=scores.dtype)
        )
        weights = mx.softmax(scores, axis=-1)
        output = mx.sum(weights[:, :, :, None] * full_v, axis=-2)
        trace = {
            "kernel_name": "cpu_online_attention_dense_tail",
            "metal_used": False,
            "fallback_used": True,
            "qjl_used": use_qjl,
            "quant_v_used": True,
            "tail_length": tail_k.shape[2],
            "actual_seq_len": actual_seq_len,
            "num_queries_per_kv": num_queries_per_kv,
        }
        return output, trace

    # ------------------------------------------------------------------
    # Page-based online-softmax attention (Stage A correctness-first).
    # Processes one page at a time without materializing the full cache.
    # ------------------------------------------------------------------
    @dataclass
    class OnlineSoftmaxState:
        max_score: mx.array  # [B, H_q]
        exp_sum: mx.array  # [B, H_q]
        weighted_value_sum: mx.array  # [B, H_q, D]

    def _online_softmax_combine(
        self,
        state: "MetalKernelBridge.OnlineSoftmaxState",
        scores: mx.array,  # [B, H_q, T]
        values: mx.array,  # [B, H_q, T, D]
    ) -> "MetalKernelBridge.OnlineSoftmaxState":
        """Update an existing online-softmax state with a new page of scores/values."""
        page_max = mx.max(scores, axis=-1)  # [B, H_q]
        new_max = mx.maximum(state.max_score, page_max)
        # exp(m_old - m_new) * l_old
        exp_old = mx.exp(state.max_score - new_max) * state.exp_sum
        # exp(m_page - m_new) * sum(exp(s - m_page))
        exp_page = mx.exp(page_max - new_max) * mx.sum(
            mx.exp(scores - page_max[:, :, None]), axis=-1
        )
        new_exp_sum = exp_old + exp_page
        # a = exp(m_old - m_new) * a_old + exp(m_page - m_new) * sum(exp(s - m_page) * v)
        page_weights = mx.exp(scores - page_max[:, :, None])
        page_weighted = mx.sum(page_weights[:, :, :, None] * values, axis=-2)
        new_weighted = (
            mx.exp(state.max_score - new_max)[:, :, None] * state.weighted_value_sum
            + mx.exp(page_max - new_max)[:, :, None] * page_weighted
        )
        return MetalKernelBridge.OnlineSoftmaxState(
            max_score=new_max,
            exp_sum=new_exp_sum,
            weighted_value_sum=new_weighted,
        )

    def _online_softmax_combine_raw(
        self,
        state: "MetalKernelBridge.OnlineSoftmaxState",
        page_max: mx.array,  # [B, H_q]
        page_exp_sum: mx.array,  # [B, H_q]
        page_weighted: mx.array,  # [B, H_q, D]
    ) -> "MetalKernelBridge.OnlineSoftmaxState":
        """Update an existing online-softmax state with raw page state."""
        new_max = mx.maximum(state.max_score, page_max)
        alpha = mx.exp(state.max_score - new_max)
        beta = mx.exp(page_max - new_max)
        new_exp_sum = state.exp_sum * alpha + page_exp_sum * beta
        new_weighted = (
            state.weighted_value_sum * alpha[:, :, None]
            + page_weighted * beta[:, :, None]
        )
        return MetalKernelBridge.OnlineSoftmaxState(
            max_score=new_max,
            exp_sum=new_exp_sum,
            weighted_value_sum=new_weighted,
        )

    def execute_paged_online_attention(
        self,
        q: mx.array,
        pages,
        tail_k: Optional[mx.array],
        tail_v: Optional[mx.array],
        config,
        actual_seq_len: int,
        mode: ExecutionMode = ExecutionMode.DEVELOPMENT_AUTO,
    ) -> Tuple[mx.array, Dict[str, Any]]:
        """Page-based online-softmax attention without full-cache materialization.

        Args:
            q: [B, H_q, D] single-token query.
            pages: iterable of CompressedPageView-like objects with k_page, v_page, valid_blocks, metadata.
            tail_k: [B, H_kv, T_tail, D] dense partial tail, or None.
            tail_v: [B, H_kv, T_tail, D] dense partial tail values, or None.
            config: TurboPolarConfig with attention_scale.
            actual_seq_len: total valid tokens for masking.
            mode: ExecutionMode. METAL_STRICT raises on any fallback.

        Returns:
            [B, H_q, D] attention output and execution trace dict.
        """
        if mode is ExecutionMode.REFERENCE:
            return self._execute_paged_online_attention_reference(
                q, pages, tail_k, tail_v, config, actual_seq_len
            )
        if mode is ExecutionMode.METAL_STRICT:
            return self._execute_paged_online_attention_metal_strict(
                q, pages, tail_k, tail_v, config, actual_seq_len
            )
        # DEVELOPMENT_AUTO: try strict, fall back to reference on documented
        # Metal availability/dispatch failures only. Programming errors propagate.
        try:
            return self._execute_paged_online_attention_metal_strict(
                q, pages, tail_k, tail_v, config, actual_seq_len
            )
        except (MetalExecutionRequiredError, MetalKernelInitializationError, MetalKernelDispatchError, RuntimeError) as _exc:
            self._stats.fallback_calls += 1
            out, trace = self._execute_paged_online_attention_reference(
                q, pages, tail_k, tail_v, config, actual_seq_len
            )
            trace["fallback_used"] = True
            trace["fallback_reason"] = f"{_exc.__class__.__name__}: {_exc}"
            return out, trace

    def _execute_paged_online_attention_reference(
        self,
        q: mx.array,
        pages,
        tail_k: Optional[mx.array],
        tail_v: Optional[mx.array],
        config,
        actual_seq_len: int,
    ) -> Tuple[mx.array, Dict[str, Any]]:
        """Reference implementation using dense MLX attention."""
        B, H_q, D = q.shape[0], q.shape[1], config.head_dim
        state = MetalKernelBridge.OnlineSoftmaxState(
            max_score=mx.full((B, H_q), -float("inf"), dtype=mx.float32),
            exp_sum=mx.zeros((B, H_q), dtype=mx.float32),
            weighted_value_sum=mx.zeros((B, H_q, D), dtype=mx.float32),
        )
        total_tokens = 0
        for page_view in pages:
            valid_blocks = page_view.valid_blocks
            if valid_blocks == 0:
                continue
            decoder = PolarQuantDecoder()
            vq = GroupedVQuantizer(group_size=page_view.v_page.group_size)
            k_page = page_view.k_page
            block = PolarKeyBlock(
                radii=k_page.radii[:, :, :valid_blocks, :, :],
                angle_codes_l1=k_page.angle_codes_l1[:, :, :valid_blocks, :, :],
                angle_codes_deep=k_page.angle_codes_deep[:, :, :valid_blocks, :, :],
                radii_scales=(
                    k_page.radii_scales[:, :, :valid_blocks, :, :]
                    if k_page.radii_scales is not None
                    else None
                ),
                shape=(B, k_page.radii.shape[1], valid_blocks * config.block_size, D),
                block_size=config.block_size,
                head_dim=D,
                metadata=page_view.metadata,
            )
            k_dense = decoder.decode_block(block)
            v_page = page_view.v_page
            v_dense = vq.dequantize_block(
                QuantizedVBlock(
                    codes=v_page.codes[:, :, :valid_blocks, :, :],
                    scales=v_page.scales[:, :, :valid_blocks, :, :],
                    group_size=v_page.group_size,
                )
            ).reshape(B, v_page.codes.shape[1], valid_blocks * config.block_size, D)
            scores = (
                mx.sum(q[:, :, None, :] * k_dense, axis=-1) * config.attention_scale
            )
            state = self._online_softmax_combine(state, scores, v_dense)
            total_tokens += valid_blocks * config.block_size
        if tail_k is not None and tail_k.shape[2] > 0:
            H_kv = tail_k.shape[1]
            nq = H_q // H_kv
            tk = mx.repeat(tail_k, nq, axis=1)
            tv = mx.repeat(tail_v, nq, axis=1)
            scores = mx.sum(q[:, :, None, :] * tk, axis=-1) * config.attention_scale
            state = self._online_softmax_combine(state, scores, tv)
            total_tokens += tail_k.shape[2]
        output = state.weighted_value_sum / state.exp_sum[:, :, None]
        trace = {
            "kernel_name": "paged_online_attention_reference",
            "execution_mode": "reference",
            "metal_used": False,
            "fallback_used": False,
            "actual_seq_len": actual_seq_len,
            "total_tokens_processed": total_tokens,
        }
        return output.astype(mx.float16), trace

    def _execute_paged_online_attention_metal_strict(
        self,
        q: mx.array,
        pages,
        tail_k: Optional[mx.array],
        tail_v: Optional[mx.array],
        config,
        actual_seq_len: int,
    ) -> Tuple[mx.array, Dict[str, Any]]:
        """Strict Metal path: any missing kernel or dispatch error is fatal."""
        if self._kernel_attn_quant_raw is None:
            raise MetalExecutionRequiredError(
                "Raw compressed-page Metal kernel is unavailable."
            )
        B = q.shape[0]
        H_q = q.shape[1]
        D = config.head_dim
        num_queries_per_kv = (
            H_q // config.num_kv_heads if config.num_kv_heads > 0 else 1
        )

        state = MetalKernelBridge.OnlineSoftmaxState(
            max_score=mx.full((B, H_q), -float("inf"), dtype=mx.float32),
            exp_sum=mx.zeros((B, H_q), dtype=mx.float32),
            weighted_value_sum=mx.zeros((B, H_q, D), dtype=mx.float32),
        )

        total_tokens = 0
        page_traces: list[Dict[str, Any]] = []

        for page_view in pages:
            valid_blocks = page_view.valid_blocks
            if valid_blocks == 0:
                continue

            k_page = page_view.k_page
            block = PolarKeyBlock(
                radii=k_page.radii[:, :, :valid_blocks, :, :],
                angle_codes_l1=k_page.angle_codes_l1[:, :, :valid_blocks, :, :],
                angle_codes_deep=k_page.angle_codes_deep[:, :, :valid_blocks, :, :],
                radii_scales=(
                    k_page.radii_scales[:, :, :valid_blocks, :, :]
                    if k_page.radii_scales is not None
                    else None
                ),
                shape=(B, k_page.radii.shape[1], valid_blocks * config.block_size, D),
                block_size=config.block_size,
                head_dim=D,
                metadata=page_view.metadata,
            )

            v_page = page_view.v_page
            quant_v = QuantizedVBlock(
                codes=v_page.codes[:, :, :valid_blocks, :, :],
                scales=v_page.scales[:, :, :valid_blocks, :, :],
                group_size=v_page.group_size,
            )

            page_weighted, page_max, page_exp, page_trace = (
                self.execute_online_attention_quant_v_raw(
                    q,
                    block,
                    quant_v,
                    config,
                    actual_seq_len=valid_blocks * config.block_size,
                    strict=True,
                )
            )
            if page_trace.get("fallback_used"):
                raise MetalExecutionRequiredError(
                    f"Compressed-page trace reported fallback_used=True for page with {valid_blocks} blocks"
                )
            state = self._online_softmax_combine_raw(
                state, page_max, page_exp, page_weighted
            )
            total_tokens += valid_blocks * config.block_size
            page_traces.append(page_trace)
            self._stats.compressed_page_dispatches += 1

        # Dense tail via Metal raw-state kernel.
        dense_tail_metal = False
        if tail_k is not None and tail_k.shape[2] > 0:
            if self._kernel_dense_tail_raw is None:
                raise MetalExecutionRequiredError(
                    "Dense-tail raw-state Metal kernel is unavailable."
                )
            try:
                tail_weighted, tail_max, tail_exp = self._execute_dense_tail_raw(
                    q, tail_k, tail_v, config
                )
            except Exception as _exc:
                raise MetalKernelDispatchError(
                    f"Dense-tail raw-state Metal kernel dispatch failed: {_exc}"
                ) from _exc
            state = self._online_softmax_combine_raw(
                state, tail_max, tail_exp, tail_weighted
            )
            total_tokens += tail_k.shape[2]
            dense_tail_metal = True
            self._stats.dense_tail_dispatches += 1

        output = state.weighted_value_sum / state.exp_sum[:, :, None]
        output = output.astype(mx.float16)

        self._stats.online_attention_calls += 1
        if pages and tail_k is not None and tail_k.shape[2] > 0:
            self._stats.dense_tail_calls += 1

        trace = {
            "kernel_name": "paged_online_attention_full_metal",
            "execution_mode": "metal_strict",
            "metal_used": True,
            "attn_metal_used": len(page_traces) > 0,
            "dense_tail_metal": dense_tail_metal,
            "fallback_used": False,
            "qjl_used": False,
            "quant_v_used": True,
            "actual_seq_len": actual_seq_len,
            "total_tokens_processed": total_tokens,
            "num_queries_per_kv": num_queries_per_kv,
            "page_traces": page_traces,
        }
        return output, trace

    def _execute_dense_tail_raw(
        self,
        q: mx.array,
        tail_k: mx.array,
        tail_v: mx.array,
        config,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Dispatch dense-tail raw-state Metal kernel."""
        B, H_q, D = q.shape[0], q.shape[1], config.head_dim
        tail_length = tail_k.shape[2]
        num_queries_per_kv = (
            H_q // config.num_kv_heads if config.num_kv_heads > 0 else 1
        )

        out_shape = (B, H_q, D)
        max_shape = (B, H_q)
        exp_shape = (B, H_q)

        strides = self._build_strides_dense_tail(q, tail_k, tail_v)
        result = self._kernel_dense_tail_raw(
            inputs=[
                q,
                tail_k,
                tail_v,
                mx.array(D, dtype=mx.uint32),
                mx.array(tail_length, dtype=mx.uint32),
                mx.array(config.attention_scale, dtype=mx.float16),
                mx.array(num_queries_per_kv, dtype=mx.uint32),
                strides,
            ],
            output_shapes=[out_shape, max_shape, exp_shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=self._compute_grid(B, H_q, 1),
            threadgroup=(self._tg_x, 1, 1),
        )
        return result[0], result[1], result[2]

    def _build_strides_dense_tail(
        self, q: mx.array, tail_k: mx.array, tail_v: mx.array
    ) -> mx.array:
        qs = self._contiguous_strides(q.shape)
        tks = self._contiguous_strides(tail_k.shape)
        tvs = self._contiguous_strides(tail_v.shape)
        os = self._contiguous_strides((q.shape[0], q.shape[1], q.shape[2]))
        return mx.array(
            [
                qs[0],
                qs[1],
                tks[0],
                tks[1],
                tks[2],
                tks[3],
                tvs[0],
                tvs[1],
                tvs[2],
                tvs[3],
                os[0],
                os[1],
            ],
            dtype=mx.uint32,
        )

    def _build_strides_qk(
        self, q, radii, radii_scales, angle_l1, angle_deep, out_array
    ):
        qs = self._contiguous_strides(q.shape)
        rs = self._contiguous_strides(radii.shape)
        rss = self._contiguous_strides(radii_scales.shape)
        l1s = self._contiguous_strides(angle_l1.shape)
        ds = self._contiguous_strides(angle_deep.shape)
        os = self._contiguous_strides(out_array.shape)
        return mx.array(
            [
                qs[0],
                qs[1],
                rs[0],
                rs[1],
                rs[2],
                rs[3],
                rss[0],
                rss[1],
                rss[2],
                l1s[0],
                l1s[1],
                l1s[2],
                l1s[3],
                ds[0],
                ds[1],
                ds[2],
                ds[3],
                os[0],
                os[1],
                os[2],
            ],
            dtype=mx.uint32,
        )

    def _build_strides_qjl(
        self,
        q,
        radii,
        radii_scales,
        angle_l1,
        angle_deep,
        qjl_s,
        qjl_n,
        q_proj_signs,
        out_array,
    ):
        qs = self._contiguous_strides(q.shape)
        rs = self._contiguous_strides(radii.shape)
        rss = self._contiguous_strides(radii_scales.shape)
        l1s = self._contiguous_strides(angle_l1.shape)
        ds = self._contiguous_strides(angle_deep.shape)
        ss = self._contiguous_strides(qjl_s.shape)
        ns = self._contiguous_strides(qjl_n.shape)
        ps = self._contiguous_strides(q_proj_signs.shape)
        os = self._contiguous_strides(out_array.shape)
        return mx.array(
            [
                qs[0],
                qs[1],
                rs[0],
                rs[1],
                rs[2],
                rs[3],
                rss[0],
                rss[1],
                rss[2],
                l1s[0],
                l1s[1],
                l1s[2],
                l1s[3],
                ds[0],
                ds[1],
                ds[2],
                ds[3],
                ss[0],
                ss[1],
                ss[2],
                ss[3],
                ns[0],
                ns[1],
                ns[2],
                ns[3],
                ps[0],
                ps[1],
                os[0],
                os[1],
                os[2],
            ],
            dtype=mx.uint32,
        )

    def _build_strides_attn_dense(
        self,
        q,
        radii,
        radii_scales,
        angle_l1,
        angle_deep,
        v_dense,
        qjl_s,
        qjl_n,
        q_proj_signs,
        out_array,
    ):
        qs = self._contiguous_strides(q.shape)
        rs = self._contiguous_strides(radii.shape)
        rss = self._contiguous_strides(radii_scales.shape)
        l1s = self._contiguous_strides(angle_l1.shape)
        ds = self._contiguous_strides(angle_deep.shape)
        vs = self._contiguous_strides(v_dense.shape)
        ss = self._contiguous_strides(qjl_s.shape)
        ns = self._contiguous_strides(qjl_n.shape)
        ps = self._contiguous_strides(q_proj_signs.shape)
        os = self._contiguous_strides(out_array.shape)
        return mx.array(
            [
                qs[0],
                qs[1],
                rs[0],
                rs[1],
                rs[2],
                rs[3],
                rss[0],
                rss[1],
                rss[2],
                l1s[0],
                l1s[1],
                l1s[2],
                l1s[3],
                ds[0],
                ds[1],
                ds[2],
                ds[3],
                vs[0],
                vs[1],
                vs[2],
                vs[3],
                ss[0],
                ss[1],
                ss[2],
                ss[3],
                ns[0],
                ns[1],
                ns[2],
                ns[3],
                ps[0],
                ps[1],
                os[0],
                os[1],
            ],
            dtype=mx.uint32,
        )

    def _build_strides_attn_quant(
        self,
        q,
        radii,
        radii_scales,
        angle_l1,
        angle_deep,
        v_codes,
        v_scales,
        qjl_s,
        qjl_n,
        q_proj_signs,
        out_array,
    ):
        qs = self._contiguous_strides(q.shape)
        rs = self._contiguous_strides(radii.shape)
        rss = self._contiguous_strides(radii_scales.shape)
        l1s = self._contiguous_strides(angle_l1.shape)
        ds = self._contiguous_strides(angle_deep.shape)
        vcs = self._contiguous_strides(v_codes.shape)
        vss = self._contiguous_strides(v_scales.shape)
        ss = self._contiguous_strides(qjl_s.shape)
        ns = self._contiguous_strides(qjl_n.shape)
        ps = self._contiguous_strides(q_proj_signs.shape)
        os = self._contiguous_strides(out_array.shape)
        return mx.array(
            [
                qs[0],
                qs[1],
                rs[0],
                rs[1],
                rs[2],
                rs[3],
                rss[0],
                rss[1],
                rss[2],
                l1s[0],
                l1s[1],
                l1s[2],
                l1s[3],
                ds[0],
                ds[1],
                ds[2],
                ds[3],
                vcs[0],
                vcs[1],
                vcs[2],
                vcs[3],
                vss[0],
                vss[1],
                vss[2],
                vss[3],
                ss[0],
                ss[1],
                ss[2],
                ss[3],
                ns[0],
                ns[1],
                ns[2],
                ns[3],
                ps[0],
                ps[1],
                os[0],
                os[1],
            ],
            dtype=mx.uint32,
        )

    def _build_strides_attn_quant_dense_tail(
        self,
        q,
        radii,
        radii_scales,
        angle_l1,
        angle_deep,
        v_codes,
        v_scales,
        tail_k,
        tail_v,
        qjl_s,
        qjl_n,
        q_proj_signs,
        out_array,
    ):
        qs = self._contiguous_strides(q.shape)
        rs = self._contiguous_strides(radii.shape)
        rss = self._contiguous_strides(radii_scales.shape)
        l1s = self._contiguous_strides(angle_l1.shape)
        ds = self._contiguous_strides(angle_deep.shape)
        vcs = self._contiguous_strides(v_codes.shape)
        vss = self._contiguous_strides(v_scales.shape)
        tks = self._contiguous_strides(tail_k.shape)
        tvs = self._contiguous_strides(tail_v.shape)
        ss = self._contiguous_strides(qjl_s.shape)
        ns = self._contiguous_strides(qjl_n.shape)
        ps = self._contiguous_strides(q_proj_signs.shape)
        os = self._contiguous_strides(out_array.shape)
        return mx.array(
            [
                qs[0],
                qs[1],
                rs[0],
                rs[1],
                rs[2],
                rs[3],
                rss[0],
                rss[1],
                rss[2],
                l1s[0],
                l1s[1],
                l1s[2],
                l1s[3],
                ds[0],
                ds[1],
                ds[2],
                ds[3],
                vcs[0],
                vcs[1],
                vcs[2],
                vcs[3],
                vss[0],
                vss[1],
                vss[2],
                vss[3],
                tks[0],
                tks[1],
                tks[2],
                tks[3],
                tvs[0],
                tvs[1],
                tvs[2],
                tvs[3],
                ss[0],
                ss[1],
                ss[2],
                ss[3],
                ns[0],
                ns[1],
                ns[2],
                ns[3],
                ps[0],
                ps[1],
                os[0],
                os[1],
            ],
            dtype=mx.uint32,
        )

    def _prepare_radii_inputs(
        self, block: PolarKeyBlock
    ) -> Tuple[mx.array, mx.array, mx.array, int, int]:
        """Return (polar_radii_fp16, polar_radii_i8, radii_scales, int8_radii, log_radii)."""
        if block.radii.dtype == mx.int8:
            if block.radii_scales is None:
                raise ValueError("int8 radii require radii_scales")
            B, H, S, L, _ = block.radii.shape
            # Dummy buffers must have more than one element so MLX treats them as
            # device arrays rather than scalar constants.
            polar_radii = mx.zeros((2,), dtype=mx.float16)
            polar_radii_i8 = mx.contiguous(block.radii)
            radii_scales = mx.contiguous(block.radii_scales.reshape(B, H, S))
            int8_radii = 1
            log_radii = 1 if block.metadata.get("log_radii", False) else 0
        elif block.radii.dtype == mx.float16:
            polar_radii = mx.contiguous(block.radii)
            polar_radii_i8 = mx.zeros((2,), dtype=mx.int8)
            radii_scales = mx.zeros((2, 2, 2), dtype=mx.float16)
            int8_radii = 0
            log_radii = 0
        else:
            raise ValueError(f"unsupported radii dtype: {block.radii.dtype}")
        return polar_radii, polar_radii_i8, radii_scales, int8_radii, log_radii

    def _metal_supports_block(self, block: PolarKeyBlock) -> bool:
        """Return True only if the compiled Metal kernels can decode this block format."""
        # Metal shaders now support fp16 or int8+log radii.
        if block.radii.dtype == mx.int8:
            if block.radii_scales is None:
                return False
        elif block.radii.dtype != mx.float16:
            return False

        l1_bits = block.metadata.get("l1_bits", 4)
        deep_bits = block.metadata.get("deep_bits", 2)
        if l1_bits not in (4, 8):
            return False
        if deep_bits not in (2, 4, 8):
            return False
        return True

    def execute_fused_qk(self, q: mx.array, block: PolarKeyBlock, config) -> mx.array:
        if not self.threadgroup_supported or not self._metal_supports_block(block):
            self._stats.fallback_calls += 1
            return self._cpu_fused_qk(q, block, config)
        B, H_q, S, L, _ = block.radii.shape
        H_kv = block.radii.shape[1]
        num_queries_per_kv = q.shape[1] // H_kv
        out_shape = (B, q.shape[1], S * L)
        out_dtype = mx.float16
        out_array = mx.zeros(out_shape, dtype=out_dtype)

        polar_radii, polar_radii_i8, radii_scales, int8_radii, log_radii = (
            self._prepare_radii_inputs(block)
        )
        radii_for_strides = polar_radii_i8 if int8_radii else polar_radii
        q, angle_l1, angle_deep = self._ensure_contiguous(
            q, block.angle_codes_l1, block.angle_codes_deep
        )
        strides = self._build_strides_qk(
            q, radii_for_strides, radii_scales, angle_l1, angle_deep, out_array
        )
        result = self._kernel_qk(
            inputs=[
                q,
                polar_radii,
                polar_radii_i8,
                radii_scales,
                angle_l1,
                angle_deep,
                mx.array(config.head_dim, dtype=mx.uint32),
                mx.array(
                    getattr(config, "split_dim", config.head_dim // 2), dtype=mx.uint32
                ),
                mx.array(config.block_size, dtype=mx.uint32),
                mx.array(float(block.metadata.get("l1_scale", 15.0)), dtype=mx.float16),
                mx.array(
                    float(block.metadata.get("deep_scale", 3.0)), dtype=mx.float16
                ),
                mx.array(config.attention_scale, dtype=mx.float16),
                mx.array(num_queries_per_kv, dtype=mx.uint32),
                mx.array(int8_radii, dtype=mx.uint32),
                mx.array(log_radii, dtype=mx.uint32),
                mx.array(int(block.metadata.get("l1_bits", 4)), dtype=mx.uint32),
                mx.array(int(block.metadata.get("deep_bits", 2)), dtype=mx.uint32),
                strides,
            ],
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            grid=self._compute_grid(B, q.shape[1], S),
            threadgroup=(self._tg_x, 1, 1),
        )[0]
        self._stats.fused_qk_calls += 1
        return result

    def execute_fused_qk_qjl(
        self,
        q: mx.array,
        block: PolarKeyBlock,
        qjl_payload: QJLPayload,
        q_proj_signs: mx.array,
        config,
    ) -> mx.array:
        if not self.threadgroup_supported or not self._metal_supports_block(block):
            self._stats.fallback_calls += 1
            return self._cpu_fused_qk_qjl(q, block, qjl_payload, q_proj_signs, config)
        B, H_q, S, L, _ = block.radii.shape
        H_kv = block.radii.shape[1]
        num_queries_per_kv = q.shape[1] // H_kv
        out_shape = (B, q.shape[1], S * L)
        out_dtype = mx.float16
        out_array = mx.zeros(out_shape, dtype=out_dtype)

        polar_radii, polar_radii_i8, radii_scales, int8_radii, log_radii = (
            self._prepare_radii_inputs(block)
        )
        radii_for_strides = polar_radii_i8 if int8_radii else polar_radii
        q, angle_l1, angle_deep, qjl_s, qjl_n, q_signs = self._ensure_contiguous(
            q,
            block.angle_codes_l1,
            block.angle_codes_deep,
            qjl_payload.packed_signs,
            qjl_payload.norms,
            q_proj_signs,
        )
        strides = self._build_strides_qjl(
            q,
            radii_for_strides,
            radii_scales,
            angle_l1,
            angle_deep,
            qjl_s,
            qjl_n,
            q_signs,
            out_array,
        )
        result = self._kernel_qk_qjl(
            inputs=[
                q,
                polar_radii,
                polar_radii_i8,
                radii_scales,
                angle_l1,
                angle_deep,
                qjl_s,
                qjl_n,
                q_signs,
                mx.array(config.head_dim, dtype=mx.uint32),
                mx.array(
                    getattr(config, "split_dim", config.head_dim // 2), dtype=mx.uint32
                ),
                mx.array(config.block_size, dtype=mx.uint32),
                mx.array(config.qjl_proj_dim, dtype=mx.uint32),
                mx.array(float(block.metadata.get("l1_scale", 15.0)), dtype=mx.float16),
                mx.array(
                    float(block.metadata.get("deep_scale", 3.0)), dtype=mx.float16
                ),
                mx.array(config.attention_scale, dtype=mx.float16),
                mx.array(num_queries_per_kv, dtype=mx.uint32),
                mx.array(int8_radii, dtype=mx.uint32),
                mx.array(log_radii, dtype=mx.uint32),
                mx.array(int(block.metadata.get("l1_bits", 4)), dtype=mx.uint32),
                mx.array(int(block.metadata.get("deep_bits", 2)), dtype=mx.uint32),
                strides,
            ],
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            grid=self._compute_grid(B, q.shape[1], S),
            threadgroup=(self._tg_x, 1, 1),
        )[0]
        self._stats.fused_qk_calls += 1
        return result

    def _resolve_qjl_tensors(
        self,
        qjl_payload: Optional[QJLPayload],
        q_proj_signs: Optional[mx.array],
        B: int,
        H_q: int,
        H_kv: int,
        S: int,
        L: int,
        qjl_proj_dim: int,
        use_qjl: bool,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        if use_qjl:
            if qjl_payload is None or q_proj_signs is None:
                raise ValueError(
                    "qjl_payload and q_proj_signs are required when use_qjl=True"
                )
            return qjl_payload.packed_signs, qjl_payload.norms, q_proj_signs
        qjl_bytes = qjl_proj_dim // 8
        qjl_s = mx.zeros((B, H_kv, S, L, qjl_bytes), dtype=mx.uint8)
        qjl_n = mx.zeros((B, H_kv, S, L), dtype=mx.float16)
        q_proj = mx.zeros((B, H_q, qjl_bytes), dtype=mx.uint8)
        return qjl_s, qjl_n, q_proj

    def execute_online_attention_dense_v(
        self,
        q: mx.array,
        block: PolarKeyBlock,
        v_dense: mx.array,
        qjl_payload: Optional[QJLPayload],
        q_proj_signs: Optional[mx.array],
        config,
        actual_seq_len: int,
        use_qjl: bool = False,
    ) -> Tuple[mx.array, Dict[str, Any]]:
        B, H_kv, S, L, _ = block.radii.shape
        num_queries_per_kv = q.shape[1] // H_kv
        qjl_s, qjl_n, q_proj = self._resolve_qjl_tensors(
            qjl_payload,
            q_proj_signs,
            B,
            q.shape[1],
            H_kv,
            S,
            L,
            config.qjl_proj_dim,
            use_qjl,
        )
        if not self.threadgroup_supported or not self._metal_supports_block(block):
            self._stats.fallback_calls += 1
            v_broadcast = mx.repeat(
                v_dense.reshape(B, H_kv, S * L, config.head_dim),
                num_queries_per_kv,
                axis=1,
            )
            return self._cpu_online_attention(
                q,
                block,
                v_broadcast,
                qjl_payload,
                q_proj_signs,
                config,
                actual_seq_len,
                use_qjl,
                quant_v_used=False,
            )
        out_shape = (B, q.shape[1], config.head_dim)
        out_dtype = mx.float16
        out_array = mx.zeros(out_shape, dtype=out_dtype)

        polar_radii, polar_radii_i8, radii_scales, int8_radii, log_radii = (
            self._prepare_radii_inputs(block)
        )
        radii_for_strides = polar_radii_i8 if int8_radii else polar_radii
        q, angle_l1, angle_deep, v, qjl_s, qjl_n, q_signs = self._ensure_contiguous(
            q,
            block.angle_codes_l1,
            block.angle_codes_deep,
            v_dense,
            qjl_s,
            qjl_n,
            q_proj,
        )
        strides = self._build_strides_attn_dense(
            q,
            radii_for_strides,
            radii_scales,
            angle_l1,
            angle_deep,
            v,
            qjl_s,
            qjl_n,
            q_signs,
            out_array,
        )
        output = self._kernel_attn_dense(
            inputs=[
                q,
                polar_radii,
                polar_radii_i8,
                radii_scales,
                angle_l1,
                angle_deep,
                v,
                qjl_s,
                qjl_n,
                q_signs,
                mx.array(config.head_dim, dtype=mx.uint32),
                mx.array(
                    getattr(config, "split_dim", config.head_dim // 2), dtype=mx.uint32
                ),
                mx.array(config.block_size, dtype=mx.uint32),
                mx.array(S, dtype=mx.uint32),
                mx.array(config.qjl_proj_dim, dtype=mx.uint32),
                mx.array(1 if use_qjl else 0, dtype=mx.uint32),
                mx.array(float(block.metadata.get("l1_scale", 15.0)), dtype=mx.float16),
                mx.array(
                    float(block.metadata.get("deep_scale", 3.0)), dtype=mx.float16
                ),
                mx.array(config.attention_scale, dtype=mx.float16),
                mx.array(int8_radii, dtype=mx.uint32),
                mx.array(log_radii, dtype=mx.uint32),
                mx.array(int(block.metadata.get("l1_bits", 4)), dtype=mx.uint32),
                mx.array(int(block.metadata.get("deep_bits", 2)), dtype=mx.uint32),
                strides,
                mx.array(actual_seq_len, dtype=mx.uint32),
                mx.array(num_queries_per_kv, dtype=mx.uint32),
            ],
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            grid=self._compute_grid(B, q.shape[1], 1),
            threadgroup=(self._tg_x, 1, 1),
        )[0]
        self._stats.online_attention_calls += 1
        trace = {
            "kernel_name": "tqpolar_online_attention_dense_v",
            "metal_used": True,
            "fallback_used": False,
            "qjl_used": use_qjl,
            "quant_v_used": False,
            "actual_seq_len": actual_seq_len,
            "num_queries_per_kv": num_queries_per_kv,
        }
        return output, trace

    def execute_online_attention_quant_v(
        self,
        q: mx.array,
        block: PolarKeyBlock,
        quant_v: QuantizedVBlock,
        qjl_payload: Optional[QJLPayload],
        q_proj_signs: Optional[mx.array],
        config,
        actual_seq_len: int,
        use_qjl: bool = False,
    ) -> Tuple[mx.array, Dict[str, Any]]:
        B, H_kv, S, L, _ = block.radii.shape
        num_queries_per_kv = q.shape[1] // H_kv
        qjl_s, qjl_n, q_proj = self._resolve_qjl_tensors(
            qjl_payload,
            q_proj_signs,
            B,
            q.shape[1],
            H_kv,
            S,
            L,
            config.qjl_proj_dim,
            use_qjl,
        )
        if not self.threadgroup_supported or not self._metal_supports_block(block):
            self._stats.fallback_calls += 1
            v_dequant = GroupedVQuantizer(
                group_size=quant_v.group_size
            ).dequantize_block(quant_v)
            v_broadcast = mx.repeat(
                v_dequant.reshape(B, H_kv, S * L, config.head_dim),
                num_queries_per_kv,
                axis=1,
            )
            return self._cpu_online_attention(
                q,
                block,
                v_broadcast,
                qjl_payload,
                q_proj_signs,
                config,
                actual_seq_len,
                use_qjl,
                quant_v_used=True,
            )
        out_shape = (B, q.shape[1], config.head_dim)
        out_dtype = mx.float16
        out_array = mx.zeros(out_shape, dtype=out_dtype)

        polar_radii, polar_radii_i8, radii_scales, int8_radii, log_radii = (
            self._prepare_radii_inputs(block)
        )
        radii_for_strides = polar_radii_i8 if int8_radii else polar_radii
        q, angle_l1, angle_deep, v_codes, v_scales, qjl_s, qjl_n, q_signs = (
            self._ensure_contiguous(
                q,
                block.angle_codes_l1,
                block.angle_codes_deep,
                quant_v.codes,
                quant_v.scales,
                qjl_s,
                qjl_n,
                q_proj,
            )
        )
        strides = self._build_strides_attn_quant(
            q,
            radii_for_strides,
            radii_scales,
            angle_l1,
            angle_deep,
            v_codes,
            v_scales,
            qjl_s,
            qjl_n,
            q_signs,
            out_array,
        )
        output = self._kernel_attn_quant(
            inputs=[
                q,
                polar_radii,
                polar_radii_i8,
                radii_scales,
                angle_l1,
                angle_deep,
                v_codes,
                v_scales,
                qjl_s,
                qjl_n,
                q_signs,
                mx.array(config.head_dim, dtype=mx.uint32),
                mx.array(
                    getattr(config, "split_dim", config.head_dim // 2), dtype=mx.uint32
                ),
                mx.array(config.block_size, dtype=mx.uint32),
                mx.array(S, dtype=mx.uint32),
                mx.array(config.qjl_proj_dim, dtype=mx.uint32),
                mx.array(quant_v.group_size, dtype=mx.uint32),
                mx.array(1 if use_qjl else 0, dtype=mx.uint32),
                mx.array(float(block.metadata.get("l1_scale", 15.0)), dtype=mx.float16),
                mx.array(
                    float(block.metadata.get("deep_scale", 3.0)), dtype=mx.float16
                ),
                mx.array(config.attention_scale, dtype=mx.float16),
                mx.array(int8_radii, dtype=mx.uint32),
                mx.array(log_radii, dtype=mx.uint32),
                mx.array(int(block.metadata.get("l1_bits", 4)), dtype=mx.uint32),
                mx.array(int(block.metadata.get("deep_bits", 2)), dtype=mx.uint32),
                strides,
                mx.array(actual_seq_len, dtype=mx.uint32),
                mx.array(num_queries_per_kv, dtype=mx.uint32),
            ],
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            grid=self._compute_grid(B, q.shape[1], 1),
            threadgroup=(self._tg_x, 1, 1),
        )[0]
        self._stats.online_attention_calls += 1
        trace = {
            "kernel_name": "tqpolar_online_attention_quant_v",
            "metal_used": True,
            "fallback_used": False,
            "qjl_used": use_qjl,
            "quant_v_used": True,
            "actual_seq_len": actual_seq_len,
            "num_queries_per_kv": num_queries_per_kv,
        }
        return output, trace

    def execute_online_attention_quant_v_raw(
        self,
        q: mx.array,
        block: PolarKeyBlock,
        quant_v: QuantizedVBlock,
        config,
        actual_seq_len: int,
        use_qjl: bool = False,
        strict: bool = False,
    ) -> Tuple[mx.array, mx.array, mx.array, Dict[str, Any]]:
        """Return raw online-softmax state (weighted_sum, max_score, exp_sum) without normalizing."""
        B, H_kv, S, L, _ = block.radii.shape
        num_queries_per_kv = q.shape[1] // H_kv
        qjl_s, qjl_n, q_proj = self._resolve_qjl_tensors(
            None, None, B, q.shape[1], H_kv, S, L, config.qjl_proj_dim, use_qjl
        )
        if not self.threadgroup_supported or not self._metal_supports_block(block):
            if strict:
                reason = []
                if not self.threadgroup_supported:
                    reason.append("threadgroup probe failed")
                if not self._metal_supports_block(block):
                    reason.append("block metadata unsupported")
                raise MetalExecutionRequiredError(
                    f"Raw compressed-page Metal kernel unavailable: {', '.join(reason)}"
                )
            self._stats.fallback_calls += 1
            v_dequant = GroupedVQuantizer(
                group_size=quant_v.group_size
            ).dequantize_block(quant_v)
            v_broadcast = mx.repeat(
                v_dequant.reshape(B, H_kv, S * L, config.head_dim),
                num_queries_per_kv,
                axis=1,
            )
            scores = self._cpu_fused_qk(q, block, config)
            max_score = mx.max(scores, axis=-1)
            exp_sum = mx.sum(mx.exp(scores - max_score[..., None]), axis=-1)
            weighted = mx.sum(
                mx.exp(scores - max_score[..., None])[..., None] * v_broadcast,
                axis=-2,
            )
            trace = {
                "kernel_name": "tqpolar_online_attention_quant_v_raw_cpu",
                "metal_used": False,
                "fallback_used": True,
                "qjl_used": use_qjl,
                "quant_v_used": True,
                "actual_seq_len": actual_seq_len,
                "num_queries_per_kv": num_queries_per_kv,
            }
            return weighted, max_score, exp_sum, trace

        out_shape = (B, q.shape[1], config.head_dim)
        max_shape = (B, q.shape[1])
        exp_shape = (B, q.shape[1])

        polar_radii, polar_radii_i8, radii_scales, int8_radii, log_radii = (
            self._prepare_radii_inputs(block)
        )
        radii_for_strides = polar_radii_i8 if int8_radii else polar_radii
        q, angle_l1, angle_deep, v_codes, v_scales, qjl_s, qjl_n, q_signs = (
            self._ensure_contiguous(
                q,
                block.angle_codes_l1,
                block.angle_codes_deep,
                quant_v.codes,
                quant_v.scales,
                qjl_s,
                qjl_n,
                q_proj,
            )
        )
        strides = self._build_strides_attn_quant(
            q,
            radii_for_strides,
            radii_scales,
            angle_l1,
            angle_deep,
            v_codes,
            v_scales,
            qjl_s,
            qjl_n,
            q_signs,
            mx.zeros(out_shape, dtype=mx.float16),
        )
        result = self._kernel_attn_quant_raw(
            inputs=[
                q,
                polar_radii,
                polar_radii_i8,
                radii_scales,
                angle_l1,
                angle_deep,
                v_codes,
                v_scales,
                qjl_s,
                qjl_n,
                q_signs,
                mx.array(config.head_dim, dtype=mx.uint32),
                mx.array(
                    getattr(config, "split_dim", config.head_dim // 2), dtype=mx.uint32
                ),
                mx.array(config.block_size, dtype=mx.uint32),
                mx.array(S, dtype=mx.uint32),
                mx.array(config.qjl_proj_dim, dtype=mx.uint32),
                mx.array(quant_v.group_size, dtype=mx.uint32),
                mx.array(1 if use_qjl else 0, dtype=mx.uint32),
                mx.array(float(block.metadata.get("l1_scale", 15.0)), dtype=mx.float16),
                mx.array(
                    float(block.metadata.get("deep_scale", 3.0)), dtype=mx.float16
                ),
                mx.array(config.attention_scale, dtype=mx.float16),
                mx.array(int8_radii, dtype=mx.uint32),
                mx.array(log_radii, dtype=mx.uint32),
                mx.array(int(block.metadata.get("l1_bits", 4)), dtype=mx.uint32),
                mx.array(int(block.metadata.get("deep_bits", 2)), dtype=mx.uint32),
                strides,
                mx.array(actual_seq_len, dtype=mx.uint32),
                mx.array(num_queries_per_kv, dtype=mx.uint32),
            ],
            output_shapes=[out_shape, max_shape, exp_shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=self._compute_grid(B, q.shape[1], 1),
            threadgroup=(self._tg_x, 1, 1),
        )
        weighted_sum = result[0]
        max_score = result[1]
        exp_sum = result[2]
        trace = {
            "kernel_name": "tqpolar_online_attention_quant_v_raw",
            "metal_used": True,
            "fallback_used": False,
            "qjl_used": use_qjl,
            "quant_v_used": True,
            "actual_seq_len": actual_seq_len,
            "num_queries_per_kv": num_queries_per_kv,
        }
        return weighted_sum, max_score, exp_sum, trace

    def execute_online_attention_quant_v_dense_tail(
        self,
        q: mx.array,
        block: PolarKeyBlock,
        quant_v: QuantizedVBlock,
        tail_k: mx.array,
        tail_v: mx.array,
        qjl_payload: Optional[QJLPayload],
        q_proj_signs: Optional[mx.array],
        config,
        actual_seq_len: int,
        use_qjl: bool = False,
    ) -> Tuple[mx.array, Dict[str, Any]]:
        B, H_kv, S, L, _ = block.radii.shape
        tail_length = tail_k.shape[2]
        num_queries_per_kv = q.shape[1] // H_kv
        qjl_s, qjl_n, q_proj = self._resolve_qjl_tensors(
            qjl_payload,
            q_proj_signs,
            B,
            q.shape[1],
            H_kv,
            S,
            L,
            config.qjl_proj_dim,
            use_qjl,
        )
        if not self.threadgroup_supported or not self._metal_supports_block(block):
            self._stats.fallback_calls += 1
            return self._cpu_online_attention_dense_tail(
                q,
                block,
                quant_v,
                tail_k,
                tail_v,
                qjl_payload,
                q_proj_signs,
                config,
                actual_seq_len,
                use_qjl,
            )
        out_shape = (B, q.shape[1], config.head_dim)
        out_dtype = mx.float16
        out_array = mx.zeros(out_shape, dtype=out_dtype)

        polar_radii, polar_radii_i8, radii_scales, int8_radii, log_radii = (
            self._prepare_radii_inputs(block)
        )
        radii_for_strides = polar_radii_i8 if int8_radii else polar_radii
        (
            q,
            angle_l1,
            angle_deep,
            v_codes,
            v_scales,
            tail_k,
            tail_v,
            qjl_s,
            qjl_n,
            q_signs,
        ) = self._ensure_contiguous(
            q,
            block.angle_codes_l1,
            block.angle_codes_deep,
            quant_v.codes,
            quant_v.scales,
            tail_k.astype(mx.float16),
            tail_v.astype(mx.float16),
            qjl_s,
            qjl_n,
            q_proj,
        )
        strides = self._build_strides_attn_quant_dense_tail(
            q,
            radii_for_strides,
            radii_scales,
            angle_l1,
            angle_deep,
            v_codes,
            v_scales,
            tail_k,
            tail_v,
            qjl_s,
            qjl_n,
            q_signs,
            out_array,
        )
        constants = mx.array(
            [
                config.head_dim,
                getattr(config, "split_dim", config.head_dim // 2),
                config.block_size,
                S,
                tail_length,
                config.qjl_proj_dim,
                quant_v.group_size,
                1 if use_qjl else 0,
                int8_radii,
                log_radii,
                int(block.metadata.get("l1_bits", 4)),
                int(block.metadata.get("deep_bits", 2)),
                actual_seq_len,
                num_queries_per_kv,
            ],
            dtype=mx.uint32,
        )
        output = self._kernel_attn_quant_dense_tail(
            inputs=[
                q,
                polar_radii,
                polar_radii_i8,
                radii_scales,
                angle_l1,
                angle_deep,
                v_codes,
                v_scales,
                tail_k,
                tail_v,
                qjl_s,
                qjl_n,
                q_signs,
                constants,
                mx.array(float(block.metadata.get("l1_scale", 15.0)), dtype=mx.float16),
                mx.array(
                    float(block.metadata.get("deep_scale", 3.0)), dtype=mx.float16
                ),
                mx.array(config.attention_scale, dtype=mx.float16),
                strides,
            ],
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
            grid=self._compute_grid(B, q.shape[1], 1),
            threadgroup=(self._tg_x, 1, 1),
        )[0]
        self._stats.online_attention_calls += 1
        self._stats.dense_tail_calls += 1
        trace = {
            "kernel_name": "tqpolar_online_attention_quant_v_dense_tail",
            "metal_used": True,
            "fallback_used": False,
            "qjl_used": use_qjl,
            "quant_v_used": True,
            "tail_length": tail_length,
            "actual_seq_len": actual_seq_len,
            "num_queries_per_kv": num_queries_per_kv,
        }
        return output, trace
