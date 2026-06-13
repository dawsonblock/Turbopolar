"""Typed fused-decode evidence schema for TurboPolar promotion."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FusedDecodeContextResult:
    """Per-context fused decode execution evidence."""

    context_length: int = 0
    requested_positions: int = 0
    actual_positions: int = 0
    failed_positions: int = 0
    compressed_page_dispatches: int = 0
    dense_tail_dispatches: int = 0
    fallback_count: int = 0
    mean_logit_cosine: float = 0.0
    any_nans_or_infs: bool = False


@dataclass
class FusedDecodeEvidence:
    """Complete fused-decode evidence artifact."""

    model_id: str = ""
    tokenizer_revision: str = ""
    execution_mode: str = ""
    evaluated_contexts: List[int] = field(default_factory=list)
    context_results: Dict[int, FusedDecodeContextResult] = field(default_factory=dict)
    # Global quality summaries.
    mean_logit_cosine: Optional[float] = None
    p05_logit_cosine: Optional[float] = None
    min_logit_cosine: Optional[float] = None
    mean_top5_overlap: Optional[float] = None
    mean_top10_overlap: Optional[float] = None
    argmax_agreement: Optional[float] = None
    mean_perplexity_delta: Optional[float] = None
    any_nans_or_infs: bool = True
    trace_artifact_path: str = ""
    trace_artifact_hash: str = ""
    notes: List[str] = field(default_factory=list)
