"""Telemetry dataclasses for TurboPolar MLX-LM integration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KernelExecutionStats:
    """Process-level Metal kernel execution counters for TurboPolar attention.

    All counters are process-global because MetalKernelBridge is a singleton.
    Do not sum per-cache stats; read once from the bridge after an experiment.
    """

    attention_invocations: int = 0
    compressed_page_dispatches: int = 0
    compressed_page_failures: int = 0
    compressed_page_fallbacks: int = 0
    dense_tail_dispatches: int = 0
    dense_tail_failures: int = 0
    dense_tail_fallbacks: int = 0
    full_attention_fallbacks: int = 0
    # Deprecated legacy fields kept for backward compatibility.
    fused_qk_calls: int = 0
    online_attention_calls: int = 0
    dense_tail_calls: int = 0
    fallback_calls: int = 0


@dataclass
class FallbackReasonRecord:
    """Immutable record of one fallback event."""

    layer: int
    decode_step: int
    operation: str
    page_index: Optional[int]
    reason_type: str
    reason: str
    mode: str


@dataclass
class FallbackReasonArtifact:
    """Collection of fallback reasons for a benchmark run."""

    reasons: List[FallbackReasonRecord] = field(default_factory=list)

    def add(self, record: FallbackReasonRecord) -> None:
        self.reasons.append(record)

    def clear(self) -> None:
        self.reasons.clear()
