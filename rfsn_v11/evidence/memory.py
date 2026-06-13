"""Typed memory benchmark evidence schema for TurboPolar promotion."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryContextResult:
    """Per-context memory measurement result."""

    context_length: int = 0
    mode: str = ""
    model_loaded_bytes: int = 0
    post_prefill_bytes: int = 0
    post_decode_bytes: int = 0
    peak_device_bytes: int = 0
    peak_delta_bytes: int = 0
    logical_cache_bytes: int = 0
    allocated_cache_bytes: int = 0
    dense_tail_bytes: int = 0
    temporary_peak_estimate: int = 0
    retained_dense_k_history: bool = True
    retained_dense_v_history: bool = True
    fallback_count: int = 0


@dataclass
class MemoryEvidence:
    """Complete memory benchmark evidence artifact."""

    model_id: str = ""
    evaluated_contexts: List[int] = field(default_factory=list)
    diagnostic_contexts: List[int] = field(default_factory=list)
    context_results: List[MemoryContextResult] = field(default_factory=list)
    logical_kv_ratio: Optional[float] = None
    persistent_storage_ratio: Optional[float] = None
    peak_device_memory_ratio_at_8192_plus: Optional[float] = None
    hidden_dense_cache_detected: bool = True
    notes: List[str] = field(default_factory=list)
