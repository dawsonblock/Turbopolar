"""Typed Cartesian baseline comparison evidence schema for TurboPolar promotion."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BaselineContextResult:
    """Per-context comparison between TurboPolar and Cartesian int8."""

    context_length: int = 0
    dense_turbo_mean_cosine: float = 0.0
    dense_turbo_p05_cosine: float = 0.0
    dense_turbo_min_cosine: float = 0.0
    dense_turbo_argmax_agreement: float = 0.0
    dense_cartesian_mean_cosine: float = 0.0
    dense_cartesian_p05_cosine: float = 0.0
    dense_cartesian_min_cosine: float = 0.0
    dense_cartesian_argmax_agreement: float = 0.0
    turbo_wins_quality: bool = False
    turbo_wins_memory: bool = False
    turbo_wins_speed: bool = False


@dataclass
class BaselineEvidence:
    """Complete baseline comparison evidence artifact."""

    model_id: str = ""
    evaluated_contexts: List[int] = field(default_factory=list)
    context_results: Dict[int, BaselineContextResult] = field(default_factory=dict)
    cartesian_int8_baseline_implemented: bool = False
    turbo_polar_wins_on_quality: Optional[bool] = None
    turbo_polar_wins_on_memory: Optional[bool] = None
    turbo_polar_wins_on_speed: Optional[bool] = None
    recommendation: str = ""
    notes: List[str] = field(default_factory=list)
