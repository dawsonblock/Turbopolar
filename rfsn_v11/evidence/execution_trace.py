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
    layer_index: int
    decode_step: int
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
class AttentionExecutionTrace:
    """Traces for one attention step (one decode position in one layer)."""

    layer_index: int
    decode_step: int
    expected_page_count: int
    page_traces: List[KernelOperationTrace] = field(default_factory=list)
    dense_tail_trace: Optional[KernelOperationTrace] = None

    @property
    def fallback_count(self) -> int:
        traces = list(self.page_traces)
        if self.dense_tail_trace is not None:
            traces.append(self.dense_tail_trace)
        return sum(t.fallback_used for t in traces)

    @property
    def all_outputs_evaluated(self) -> bool:
        traces = list(self.page_traces)
        if self.dense_tail_trace is not None:
            traces.append(self.dense_tail_trace)
        return all(t.output_evaluated for t in traces)


# Backward-compatible alias
AttentionStepTrace = AttentionExecutionTrace


@dataclass
class ExecutionTraceCollector:
    """Collects AttentionExecutionTrace records for strict evidence validation.

    Supports provisional recording in ASYNC_PERFORMANCE mode: traces are held
    in a provisional buffer until the caller evaluates the final output and
    commits them. If evaluation fails, provisional traces can be cleared so
    unverified operations never enter the evidence artifact.
    """

    _traces: List[AttentionExecutionTrace] = field(default_factory=list)
    _provisional: List[AttentionExecutionTrace] = field(default_factory=list)
    experiment_id: str = ""

    def __post_init__(self):
        # Initialize experiment_id if not provided
        if not hasattr(self, 'experiment_id'):
            self.experiment_id = ""

    def record(self, trace: AttentionExecutionTrace) -> None:
        self._traces.append(trace)

    def record_provisional(self, trace: AttentionExecutionTrace) -> None:
        self._provisional.append(trace)

    def commit_provisional(self, output_evaluated: bool = True) -> None:
        """Move all provisional traces into the permanent record.

        Updates every nested KernelOperationTrace with the supplied
        ``output_evaluated`` value before committing.
        """
        committed: List[AttentionExecutionTrace] = []
        for step_trace in self._provisional:
            new_page_traces = [
                replace(pt, output_evaluated=output_evaluated)
                for pt in step_trace.page_traces
            ]
            new_dense_tail = None
            if step_trace.dense_tail_trace is not None:
                new_dense_tail = replace(
                    step_trace.dense_tail_trace, output_evaluated=output_evaluated
                )
            committed.append(
                AttentionExecutionTrace(
                    layer_index=step_trace.layer_index,
                    decode_step=step_trace.decode_step,
                    expected_page_count=step_trace.expected_page_count,
                    page_traces=new_page_traces,
                    dense_tail_trace=new_dense_tail,
                )
            )
        self._traces.extend(committed)
        self._provisional.clear()

    def clear_provisional(self) -> None:
        self._provisional.clear()

    def clear(self) -> None:
        self._traces.clear()
        self._provisional.clear()

    def snapshot(self) -> List[AttentionExecutionTrace]:
        return list(self._traces)

    def provisional_snapshot(self) -> List[AttentionExecutionTrace]:
        return list(self._provisional)

    def by_layer_and_step(self, layer_index: int, decode_step: int) -> Optional[AttentionExecutionTrace]:
        for t in self._traces:
            if t.layer_index == layer_index and t.decode_step == decode_step:
                return t
        return None
