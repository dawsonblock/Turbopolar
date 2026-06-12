"""Schema for promotion evidence and reports.

This module defines dataclasses only. Promotion decisions belong to gate.py.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PromotionState(str, Enum):
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    PROMOTED_EXPERIMENTAL = "PROMOTED_EXPERIMENTAL"


@dataclass
class KernelReport:
    all_unit_tests_passed: bool = False
    all_kernel_tests_passed: bool = False
    all_integration_tests_passed: bool = False
    cpu_metal_agreement_verified: bool = False
    qjl_scaled_correctly: Optional[bool] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class TeacherForcedReport:
    model: str = ""
    mean_logit_cosine: Optional[float] = None
    p05_logit_cosine: Optional[float] = None
    min_logit_cosine: Optional[float] = None
    mean_top5_overlap: Optional[float] = None
    mean_top10_overlap: Optional[float] = None
    argmax_agreement: Optional[float] = None
    mean_perplexity_delta: Optional[float] = None
    max_perplexity_delta: Optional[float] = None
    any_nans_or_infs: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class FusedDecodeReport:
    model: str = ""
    contexts_evaluated: List[int] = field(default_factory=list)
    mean_logit_cosine: Optional[float] = None
    p05_logit_cosine: Optional[float] = None
    min_logit_cosine: Optional[float] = None
    mean_top5_overlap: Optional[float] = None
    mean_top10_overlap: Optional[float] = None
    argmax_agreement: Optional[float] = None
    mean_perplexity_delta: Optional[float] = None
    max_perplexity_delta: Optional[float] = None
    any_nans_or_infs: bool = True
    first_argmax_divergence_step: Optional[int] = None
    notes: List[str] = field(default_factory=list)


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
    notes: List[str] = field(default_factory=list)


@dataclass
class MemoryReport:
    model: str = ""
    contexts_evaluated: List[int] = field(default_factory=list)
    logical_kv_ratio: Optional[float] = None
    persistent_storage_ratio: Optional[float] = None
    peak_device_memory_ratio_at_8192_plus: Optional[float] = None
    hidden_dense_cache_detected: bool = True
    notes: List[str] = field(default_factory=list)


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


@dataclass
class BenchmarkProvenance:
    run_id: str = ""
    timestamp_utc: str = ""
    git_commit: str = ""
    git_dirty: bool = True
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
    turbopolar_config_hash: str = ""
    turbopolar_config: Dict[str, Any] = field(default_factory=dict)
    benchmark_command: str = ""
    warmup_count: int = 0
    trial_count: int = 0
    context_lengths: List[int] = field(default_factory=list)
    decode_token_count: int = 0
    qjl_enabled: bool = False
    metal_kernel_source_hash: str = ""


@dataclass
class PromotionEvidence:
    kernel_report: Optional[KernelReport] = None
    teacher_forced_report: Optional[TeacherForcedReport] = None
    fused_decode_report: Optional[FusedDecodeReport] = None
    speed_report: Optional[SpeedReport] = None
    memory_report: Optional[MemoryReport] = None
    baseline_comparison_report: Optional[BaselineComparisonReport] = None
    provenance: Optional[BenchmarkProvenance] = None


@dataclass
class PromotionDecision:
    state: PromotionState = PromotionState.INCOMPLETE
    reasons: List[str] = field(default_factory=list)
    evidence: Optional[PromotionEvidence] = None
