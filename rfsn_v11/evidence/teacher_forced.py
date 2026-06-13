"""Typed teacher-forced quality evidence schema for TurboPolar promotion."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PositionLevelMetrics:
    """Per-token-position quality metrics."""

    logit_cosine: float = 0.0
    argmax_agreement: float = 0.0
    top5_overlap: float = 0.0
    top10_overlap: float = 0.0
    kl_divergence: float = 0.0
    any_nans_or_infs: bool = False


@dataclass
class ContextQualityResult:
    """Quality results for one context length."""

    context_length: int = 0
    positions: List[PositionLevelMetrics] = field(default_factory=list)
    mean_logit_cosine: float = 0.0
    p05_logit_cosine: float = 0.0
    min_logit_cosine: float = 0.0
    argmax_agreement: float = 0.0
    mean_top5_overlap: float = 0.0
    mean_top10_overlap: float = 0.0
    mean_perplexity_delta: float = 0.0
    any_nans_or_infs: bool = False


@dataclass
class TeacherForcedEvidence:
    """Complete teacher-forced quality evidence artifact."""

    model_id: str = ""
    tokenizer_revision: str = ""
    execution_mode: str = ""
    evaluated_contexts: List[int] = field(default_factory=list)
    context_results: Dict[int, ContextQualityResult] = field(default_factory=dict)
    total_positions: int = 0
    mean_logit_cosine: float = 0.0
    p05_logit_cosine: float = 0.0
    min_logit_cosine: float = 0.0
    argmax_agreement: float = 0.0
    mean_top5_overlap: float = 0.0
    mean_top10_overlap: float = 0.0
    mean_perplexity_delta: float = 0.0
    any_nans_or_infs: bool = True
    raw_metrics_path: str = ""
    raw_metrics_hash: str = ""
    notes: List[str] = field(default_factory=list)
