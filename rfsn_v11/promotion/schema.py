"""Schema for promotion evidence and reports.

This module defines dataclasses only. Promotion decisions belong to gate.py.
Nested dictionaries can be converted back to dataclasses via the ``from_dict``
classmethods on each report and on ``PromotionEvidence``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


def _numeric_or_none(value: Any) -> Any:
    """Sanitize numeric fields: reject booleans, pass everything else through.

    Python ``bool`` is a subclass of ``int``, so naive ``isinstance(x, int)``
    would accept ``True``/``False`` as valid numbers.  This helper coerces
    booleans to ``None`` so that the promotion gate treats them as missing.
    """
    if isinstance(value, bool):
        return None
    return value


class PromotionState(str, Enum):
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    PROMOTED_EXPERIMENTAL = "PROMOTED_EXPERIMENTAL"


class GitTreeState(str, Enum):
    """Tri-state git working-tree condition."""

    CLEAN = "CLEAN"
    DIRTY = "DIRTY"
    UNKNOWN = "UNKNOWN"


@dataclass
class KernelReport:
    all_unit_tests_passed: bool = False
    all_kernel_tests_passed: bool = False
    all_integration_tests_passed: bool = False
    cpu_metal_agreement_verified: bool = False
    qjl_scaled_correctly: Optional[bool] = None
    required_metal_tests: List[str] = field(default_factory=list)
    metal_tests_present: List[str] = field(default_factory=list)
    metal_tests_passed: List[str] = field(default_factory=list)
    metal_tests_skipped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KernelReport":
        return cls(
            all_unit_tests_passed=bool(data.get("all_unit_tests_passed", False)),
            all_kernel_tests_passed=bool(data.get("all_kernel_tests_passed", False)),
            all_integration_tests_passed=bool(
                data.get("all_integration_tests_passed", False)
            ),
            cpu_metal_agreement_verified=bool(
                data.get("cpu_metal_agreement_verified", False)
            ),
            qjl_scaled_correctly=data.get("qjl_scaled_correctly"),
            required_metal_tests=list(data.get("required_metal_tests", [])),
            metal_tests_present=list(data.get("metal_tests_present", [])),
            metal_tests_passed=list(data.get("metal_tests_passed", [])),
            metal_tests_skipped=list(data.get("metal_tests_skipped", [])),
            notes=list(data.get("notes", [])),
        )


@dataclass
class TeacherForcedReport:
    model: str = ""
    evaluated_contexts: List[int] = field(default_factory=list)
    total_positions: int = 0
    mean_logit_cosine: Optional[float] = None
    p05_logit_cosine: Optional[float] = None
    min_logit_cosine: Optional[float] = None
    argmax_agreement: Optional[float] = None
    mean_top5_overlap: Optional[float] = None
    mean_top10_overlap: Optional[float] = None
    mean_perplexity_delta: Optional[float] = None
    any_nans_or_infs: bool = True
    raw_metrics_path: str = ""
    raw_metrics_hash: str = ""
    notes: List[str] = field(default_factory=list)
    # Deprecated field kept for backward compatibility.
    max_perplexity_delta: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherForcedReport":
        total_positions_raw = data.get("total_positions", 0)
        if isinstance(total_positions_raw, bool):
            total_positions_raw = 0
        return cls(
            model=data.get("model", ""),
            evaluated_contexts=list(data.get("evaluated_contexts", [])),
            total_positions=int(total_positions_raw),
            mean_logit_cosine=_numeric_or_none(data.get("mean_logit_cosine")),
            p05_logit_cosine=_numeric_or_none(data.get("p05_logit_cosine")),
            min_logit_cosine=_numeric_or_none(data.get("min_logit_cosine")),
            argmax_agreement=_numeric_or_none(data.get("argmax_agreement")),
            mean_top5_overlap=_numeric_or_none(data.get("mean_top5_overlap")),
            mean_top10_overlap=_numeric_or_none(data.get("mean_top10_overlap")),
            mean_perplexity_delta=_numeric_or_none(
                data.get("mean_perplexity_delta")
            ),
            any_nans_or_infs=bool(data.get("any_nans_or_infs", True)),
            raw_metrics_path=data.get("raw_metrics_path", ""),
            raw_metrics_hash=data.get("raw_metrics_hash", ""),
            notes=list(data.get("notes", [])),
            max_perplexity_delta=_numeric_or_none(
                data.get("max_perplexity_delta")
            ),
        )


@dataclass
class FusedDecodeReport:
    model: str = ""
    # Topology expectations
    model_layer_count: int = 0
    # Per-context completeness (primary evidence).
    contexts_evaluated: List[int] = field(default_factory=list)
    requested_fused_positions_per_context: int = 0
    positions_per_context: Dict[int, int] = field(default_factory=dict)
    failed_positions_per_context: Dict[int, int] = field(default_factory=dict)
    compressed_page_dispatches_per_context: Dict[int, int] = field(default_factory=dict)
    dense_tail_dispatches_per_context: Dict[int, int] = field(default_factory=dict)
    fallback_calls_per_context: Dict[int, int] = field(default_factory=dict)
    trace_artifact_path: str = ""
    trace_artifact_hash: str = ""
    # Quality summaries across all contexts.
    mean_logit_cosine: Optional[float] = None
    p05_logit_cosine: Optional[float] = None
    min_logit_cosine: Optional[float] = None
    mean_top5_overlap: Optional[float] = None
    mean_top10_overlap: Optional[float] = None
    argmax_agreement: Optional[float] = None
    mean_perplexity_delta: Optional[float] = None
    any_nans_or_infs: bool = True
    # Strict Metal execution evidence. None means "not reported" (incomplete).
    execution_mode: Optional[str] = None
    # Legacy global fields kept for backward compatibility.
    compressed_page_metal_calls: Optional[int] = None
    dense_tail_metal_calls: Optional[int] = None
    merge_metal_calls: Optional[int] = None
    finalization_metal_calls: Optional[int] = None
    compressed_page_fallback_calls: Optional[int] = None
    dense_tail_fallback_calls: Optional[int] = None
    full_attention_fallback_calls: Optional[int] = None
    fallback_reasons: Optional[List[str]] = None
    fallback_calls: int = 0  # deprecated; kept for backward compat
    first_argmax_divergence_step: Optional[int] = None
    actual_fused_positions: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusedDecodeReport":
        def _int_or_zero(val: Any) -> int:
            if isinstance(val, bool):
                return 0
            return int(val) if val is not None else 0

        def _optional_int(val: Any) -> Optional[int]:
            if isinstance(val, bool):
                return None
            return int(val) if val is not None else None

        return cls(
            model=data.get("model", ""),
            model_layer_count=_int_or_zero(data.get("model_layer_count", 0)),
            contexts_evaluated=list(data.get("contexts_evaluated", [])),
            requested_fused_positions_per_context=_int_or_zero(
                data.get("requested_fused_positions_per_context", 0)
            ),
            positions_per_context=dict(data.get("positions_per_context", {})),
            failed_positions_per_context=dict(
                data.get("failed_positions_per_context", {})
            ),
            compressed_page_dispatches_per_context=dict(
                data.get("compressed_page_dispatches_per_context", {})
            ),
            dense_tail_dispatches_per_context=dict(
                data.get("dense_tail_dispatches_per_context", {})
            ),
            fallback_calls_per_context=dict(
                data.get("fallback_calls_per_context", {})
            ),
            trace_artifact_path=data.get("trace_artifact_path", ""),
            trace_artifact_hash=data.get("trace_artifact_hash", ""),
            mean_logit_cosine=_numeric_or_none(data.get("mean_logit_cosine")),
            p05_logit_cosine=_numeric_or_none(data.get("p05_logit_cosine")),
            min_logit_cosine=_numeric_or_none(data.get("min_logit_cosine")),
            mean_top5_overlap=_numeric_or_none(data.get("mean_top5_overlap")),
            mean_top10_overlap=_numeric_or_none(data.get("mean_top10_overlap")),
            argmax_agreement=_numeric_or_none(data.get("argmax_agreement")),
            mean_perplexity_delta=_numeric_or_none(
                data.get("mean_perplexity_delta")
            ),
            any_nans_or_infs=bool(data.get("any_nans_or_infs", True)),
            execution_mode=(
                data.get("execution_mode", "").strip().lower()
                if data.get("execution_mode")
                else None
            ),
            compressed_page_metal_calls=_optional_int(
                data.get("compressed_page_metal_calls")
            ),
            dense_tail_metal_calls=_optional_int(
                data.get("dense_tail_metal_calls")
            ),
            merge_metal_calls=_optional_int(data.get("merge_metal_calls")),
            finalization_metal_calls=_optional_int(
                data.get("finalization_metal_calls")
            ),
            compressed_page_fallback_calls=_optional_int(
                data.get("compressed_page_fallback_calls")
            ),
            dense_tail_fallback_calls=_optional_int(
                data.get("dense_tail_fallback_calls")
            ),
            full_attention_fallback_calls=_optional_int(
                data.get("full_attention_fallback_calls")
            ),
            fallback_reasons=list(data.get("fallback_reasons") or []),
            fallback_calls=_int_or_zero(data.get("fallback_calls", 0)),
            first_argmax_divergence_step=_optional_int(
                data.get("first_argmax_divergence_step")
            ),
            actual_fused_positions=_optional_int(
                data.get("actual_fused_positions")
            ),
            notes=list(data.get("notes", [])),
        )


@dataclass
class SpeedReport:
    model: str = ""
    contexts_evaluated: List[int] = field(default_factory=list)
    trials_per_context: int = 0
    dense_decode_tok_s: Dict[int, List[float]] = field(default_factory=dict)
    turbo_decode_tok_s: Dict[int, List[float]] = field(default_factory=dict)
    median_ratio: Optional[float] = None
    min_ratio_at_4096_plus: Optional[float] = None
    max_ratio_at_4096_plus: Optional[float] = None
    median_ratio_at_8192_plus: Optional[float] = None
    execution_mode: Optional[str] = None
    fallback_calls: int = 0
    raw_timing_path: str = ""
    raw_timing_hash: str = ""
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeedReport":
        dense = data.get("dense_decode_tok_s", {})
        turbo = data.get("turbo_decode_tok_s", {})
        trials_raw = data.get("trials_per_context", 0)
        fallback_raw = data.get("fallback_calls", 0)
        return cls(
            model=data.get("model", ""),
            contexts_evaluated=list(data.get("contexts_evaluated", [])),
            trials_per_context=int(trials_raw) if not isinstance(trials_raw, bool) else 0,
            dense_decode_tok_s={int(k): list(v) for k, v in dense.items()},
            turbo_decode_tok_s={int(k): list(v) for k, v in turbo.items()},
            median_ratio=_numeric_or_none(data.get("median_ratio")),
            min_ratio_at_4096_plus=_numeric_or_none(
                data.get("min_ratio_at_4096_plus")
            ),
            max_ratio_at_4096_plus=_numeric_or_none(
                data.get("max_ratio_at_4096_plus")
            ),
            median_ratio_at_8192_plus=_numeric_or_none(
                data.get("median_ratio_at_8192_plus")
            ),
            execution_mode=data.get("execution_mode"),
            fallback_calls=int(fallback_raw) if not isinstance(fallback_raw, bool) else 0,
            raw_timing_path=data.get("raw_timing_path", ""),
            raw_timing_hash=data.get("raw_timing_hash", ""),
            notes=list(data.get("notes", [])),
        )


@dataclass
class MemoryReport:
    model: str = ""
    contexts_evaluated: List[int] = field(default_factory=list)
    logical_kv_ratio: Optional[float] = None
    persistent_storage_ratio: Optional[float] = None
    peak_device_memory_ratio_at_8192_plus: Optional[float] = None
    hidden_dense_cache_detected: bool = True
    fallback_calls: int = 0
    raw_memory_path: str = ""
    raw_memory_hash: str = ""
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryReport":
        fallback_raw = data.get("fallback_calls", 0)
        return cls(
            model=data.get("model", ""),
            contexts_evaluated=list(data.get("contexts_evaluated", [])),
            logical_kv_ratio=_numeric_or_none(data.get("logical_kv_ratio")),
            persistent_storage_ratio=_numeric_or_none(
                data.get("persistent_storage_ratio")
            ),
            peak_device_memory_ratio_at_8192_plus=_numeric_or_none(
                data.get("peak_device_memory_ratio_at_8192_plus")
            ),
            hidden_dense_cache_detected=bool(
                data.get("hidden_dense_cache_detected", True)
            ),
            fallback_calls=int(fallback_raw) if not isinstance(fallback_raw, bool) else 0,
            raw_memory_path=data.get("raw_memory_path", ""),
            raw_memory_hash=data.get("raw_memory_hash", ""),
            notes=list(data.get("notes", [])),
        )


@dataclass
class BaselineComparisonReport:
    model: str = ""
    contexts_evaluated: List[int] = field(default_factory=list)
    cartesian_int8_baseline_implemented: bool = False
    turbo_polar_wins_on_quality: Optional[bool] = None
    turbo_polar_wins_on_memory: Optional[bool] = None
    turbo_polar_wins_on_speed: Optional[bool] = None
    recommendation: str = ""
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineComparisonReport":
        return cls(
            model=data.get("model", ""),
            contexts_evaluated=list(data.get("contexts_evaluated", [])),
            cartesian_int8_baseline_implemented=bool(
                data.get("cartesian_int8_baseline_implemented", False)
            ),
            turbo_polar_wins_on_quality=data.get("turbo_polar_wins_on_quality"),
            turbo_polar_wins_on_memory=data.get("turbo_polar_wins_on_memory"),
            turbo_polar_wins_on_speed=data.get("turbo_polar_wins_on_speed"),
            recommendation=data.get("recommendation", ""),
            notes=list(data.get("notes", [])),
        )


@dataclass
class BenchmarkProvenance:
    run_id: str = ""
    timestamp_utc: str = ""
    git_commit: str = ""
    git_tree_state: GitTreeState = GitTreeState.UNKNOWN
    git_diff_hash: str = ""
    python_version: str = ""
    mlx_version: str = ""
    mlx_lm_version: str = ""
    macos_version: str = ""
    chip_model: str = ""
    system_memory_gb: Optional[float] = None
    model_repo_id: str = ""
    model_revision: str = ""
    tokenizer_revision: str = ""
    prompt_suite_hash: str = ""
    token_fixtures_hash: str = ""  # P1-29: Hash of actual token arrays
    turbopolar_config_hash: str = ""
    turbopolar_config: Dict[str, Any] = field(default_factory=dict)
    benchmark_command: str = ""
    warmup_count: int = 0
    trial_count: int = 0
    context_lengths: List[int] = field(default_factory=list)
    decode_token_count: int = 0
    qjl_enabled: bool = False
    metal_kernel_source_hash: str = ""
    kernel_binding_hash: str = ""
    execution_mode: str = ""
    evidence_kind: str = "experimental"  # "experimental" | "synthetic_dry_run"
    # P1-31: Separate workload hashes for each benchmark family
    # These allow tracking workload changes independently per benchmark type
    speed_workload_hash: str = ""  # Hash of speed benchmark workload (contexts, trials, fixtures)
    memory_workload_hash: str = ""  # Hash of memory benchmark workload
    fused_decode_workload_hash: str = ""  # Hash of fused decode workload
    cartesian_workload_hash: str = ""  # Hash of cartesian comparison workload
    teacher_forced_workload_hash: str = ""  # Hash of teacher-forced benchmark workload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkProvenance":
        state_str = data.get("git_tree_state", "UNKNOWN")
        try:
            git_tree_state = GitTreeState(state_str)
        except ValueError:
            git_tree_state = GitTreeState.UNKNOWN
        return cls(
            run_id=data.get("run_id", ""),
            timestamp_utc=data.get("timestamp_utc", ""),
            git_commit=data.get("git_commit", ""),
            git_tree_state=git_tree_state,
            git_diff_hash=data.get("git_diff_hash", ""),
            python_version=data.get("python_version", ""),
            mlx_version=data.get("mlx_version", ""),
            mlx_lm_version=data.get("mlx_lm_version", ""),
            macos_version=data.get("macos_version", ""),
            chip_model=data.get("chip_model", ""),
            system_memory_gb=data.get("system_memory_gb"),
            model_repo_id=data.get("model_repo_id", ""),
            model_revision=data.get("model_revision", ""),
            tokenizer_revision=data.get("tokenizer_revision", ""),
            prompt_suite_hash=data.get("prompt_suite_hash", ""),
            token_fixtures_hash=data.get("token_fixtures_hash", ""),
            turbopolar_config_hash=data.get("turbopolar_config_hash", ""),
            turbopolar_config=dict(data.get("turbopolar_config", {})),
            benchmark_command=data.get("benchmark_command", ""),
            warmup_count=int(data.get("warmup_count", 0)),
            trial_count=int(data.get("trial_count", 0)),
            context_lengths=list(data.get("context_lengths", [])),
            decode_token_count=int(data.get("decode_token_count", 0)),
            qjl_enabled=bool(data.get("qjl_enabled", False)),
            metal_kernel_source_hash=data.get("metal_kernel_source_hash", ""),
            kernel_binding_hash=data.get("kernel_binding_hash", ""),
            execution_mode=data.get("execution_mode", ""),
            evidence_kind=data.get("evidence_kind", "experimental"),
            # P1-31: Separate workload hashes
            speed_workload_hash=data.get("speed_workload_hash", ""),
            memory_workload_hash=data.get("memory_workload_hash", ""),
            fused_decode_workload_hash=data.get("fused_decode_workload_hash", ""),
            cartesian_workload_hash=data.get("cartesian_workload_hash", ""),
            teacher_forced_workload_hash=data.get("teacher_forced_workload_hash", ""),
        )


@dataclass
class PromotionEvidence:
    kernel_report: Optional[KernelReport] = None
    teacher_forced_report: Optional[TeacherForcedReport] = None
    fused_decode_report: Optional[FusedDecodeReport] = None
    speed_report: Optional[SpeedReport] = None
    memory_report: Optional[MemoryReport] = None
    baseline_comparison_report: Optional[BaselineComparisonReport] = None
    provenance: Optional[BenchmarkProvenance] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromotionEvidence":
        """Reconstruct a PromotionEvidence tree from nested dictionaries."""
        return cls(
            kernel_report=_optional(KernelReport.from_dict, data.get("kernel_report")),
            teacher_forced_report=_optional(
                TeacherForcedReport.from_dict, data.get("teacher_forced_report")
            ),
            fused_decode_report=_optional(
                FusedDecodeReport.from_dict, data.get("fused_decode_report")
            ),
            speed_report=_optional(SpeedReport.from_dict, data.get("speed_report")),
            memory_report=_optional(MemoryReport.from_dict, data.get("memory_report")),
            baseline_comparison_report=_optional(
                BaselineComparisonReport.from_dict,
                data.get("baseline_comparison_report"),
            ),
            provenance=_optional(BenchmarkProvenance.from_dict, data.get("provenance")),
        )


def _optional(factory, value):
    if value is None:
        return None
    return factory(value)


@dataclass
class PromotionDecision:
    state: PromotionState = PromotionState.INCOMPLETE
    reasons: List[str] = field(default_factory=list)
    evidence: Optional[PromotionEvidence] = None
