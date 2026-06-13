"""Operation-level execution trace collector for TurboPolar strict evidence.

Traces record every kernel dispatch with enough detail to prove:
- which layer ran
- which page ran
- which decode step ran
- which kernel was used
- whether any page was skipped
- why a fallback happened
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional


@dataclass(frozen=True)
class KernelOperationTrace:
    """Immutable record of one kernel dispatch."""

    experiment_id: str
    decode_step: int
    layer_index: int
    operation: str  # "compressed_page" | "dense_tail" | "merge" | "finalize"
    page_index: Optional[int]
    kernel_name: str
    execution_mode: str
    metal_requested: bool
    metal_executed: bool
    fallback_used: bool
    fallback_reason: Optional[str]
    expected_tokens: int
    processed_tokens: int
    output_evaluated: bool = False


@dataclass
class AttentionStepTrace:
    """Traces for one attention step (one decode position in one layer)."""

    experiment_id: str
    decode_step: int
    layer_index: int
    expected_page_count: int
    page_operations: List[KernelOperationTrace] = field(default_factory=list)
    dense_tail_operation: Optional[KernelOperationTrace] = None

    @property
    def fallback_count(self) -> int:
        traces = [
            *self.page_operations,
            self.dense_tail_operation,
        ]
        return sum(
            t is not None and t.fallback_used
            for t in traces
        )

    @property
    def all_outputs_evaluated(self) -> bool:
        traces = [
            *self.page_operations,
            self.dense_tail_operation,
        ]
        return all(
            t is not None and t.output_evaluated
            for t in traces
        )


@dataclass
class ExecutionTraceCollector:
    """Collects AttentionStepTrace records for strict evidence validation."""

    _traces: List[AttentionStepTrace] = field(default_factory=list)

    def record(self, trace: AttentionStepTrace) -> None:
        self._traces.append(trace)

    def clear(self) -> None:
        self._traces.clear()

    def snapshot(self) -> List[AttentionStepTrace]:
        return list(self._traces)

    def by_layer_and_step(self, layer_index: int, decode_step: int) -> Optional[AttentionStepTrace]:
        for t in self._traces:
            if t.layer_index == layer_index and t.decode_step == decode_step:
                return t
        return None
