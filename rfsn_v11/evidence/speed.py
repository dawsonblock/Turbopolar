"""Typed speed benchmark evidence schema for TurboPolar promotion."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SpeedTrialResult:
    """One speed trial for a specific context and mode."""

    context_length: int = 0
    mode: str = ""
    trial: int = 0
    prefill_seconds: float = 0.0
    first_token_ms: float = 0.0
    per_token_ms: List[float] = field(default_factory=list)
    throughput_tps: float = 0.0
    block_boundary_tokens: List[int] = field(default_factory=list)
    page_boundary_tokens: List[int] = field(default_factory=list)
    page_dispatches: int = 0
    tail_dispatches: int = 0
    fallbacks: int = 0


@dataclass
class SpeedEvidence:
    """Complete speed benchmark evidence artifact."""

    model_id: str = ""
    execution_mode: str = ""
    evaluated_contexts: List[int] = field(default_factory=list)
    diagnostic_contexts: List[int] = field(default_factory=list)
    trials_per_context: int = 0
    trial_results: List[SpeedTrialResult] = field(default_factory=list)
    dense_decode_tok_s: Dict[int, List[float]] = field(default_factory=dict)
    turbo_decode_tok_s: Dict[int, List[float]] = field(default_factory=dict)
    median_ratio: Optional[float] = None
    min_ratio_at_4096_plus: Optional[float] = None
    max_ratio_at_4096_plus: Optional[float] = None
    median_ratio_at_8192_plus: Optional[float] = None
    notes: List[str] = field(default_factory=list)
