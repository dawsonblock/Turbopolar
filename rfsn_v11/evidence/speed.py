"""Typed speed benchmark evidence schema for TurboPolar promotion."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


REQUIRED_CONTEXTS = (512, 2048, 4096, 8192, 16384)
REQUIRED_TRIALS_PER_CONTEXT = 5
REQUIRED_TOKEN_LATENCIES = 128


@dataclass(frozen=True)
class RawSpeedTrial:
    """Canonical raw speed trial schema for promotion evidence.

    Both the benchmark producer and promotion gate must use this exact schema
    to ensure reproducibility and correct validation.
    """

    context_length: int
    method: str
    trial_index: int
    execution_order: Tuple[str, str]
    execution_mode: str
    prefill_seconds: float
    token_latencies_ms: Tuple[float, ...]
    first_token_ms: float
    throughput_tps: float
    compressed_page_dispatches: int
    dense_tail_dispatches: int
    fallback_calls: int


@dataclass(frozen=True)
class RawSpeedArtifact:
    """Canonical raw speed artifact schema for promotion evidence.

    Contains the schema version and all trials for validation.
    """

    schema_version: int
    trials: Tuple[RawSpeedTrial, ...]

    @classmethod
    def from_dict(cls, data: dict) -> "RawSpeedArtifact":
        """Convert dictionary to canonical schema.

        Handles three formats:
        1. Canonical: {"schema_version": 1, "trials": [...]}
        2. Benchmark: {"trial_results": [...]}
        3. Unified: {"schema_version": 1, "speed_evidence": {"trial_results": [...]}}
        """
        schema_version = data.get("schema_version", 1)

        # Handle all three formats
        trials_data = data.get("trials", [])
        if not trials_data and "trial_results" in data:
            trials_data = data["trial_results"]
        if not trials_data and "speed_evidence" in data:
            speed_evidence = data.get("speed_evidence", {})
            if isinstance(speed_evidence, dict):
                trials_data = speed_evidence.get("trial_results", [])

        trials = []
        for trial_data in trials_data:
            # Support both canonical and legacy field names for round-trip
            method = trial_data.get("method", trial_data.get("mode", "unknown"))
            trial_index = trial_data.get("trial_index", trial_data.get("trial", 0))
            latencies = trial_data.get(
                "token_latencies_ms",
                trial_data.get("per_token_ms", [])
            )
            fallback_calls = trial_data.get(
                "fallback_calls",
                trial_data.get("fallbacks", 0)
            )
            compressed_page_dispatches = trial_data.get(
                "compressed_page_dispatches",
                trial_data.get("page_dispatches", 0)
            )
            dense_tail_dispatches = trial_data.get(
                "dense_tail_dispatches",
                trial_data.get("tail_dispatches", 0)
            )
            execution_mode = trial_data.get("execution_mode", "unknown")

            trial = RawSpeedTrial(
                context_length=trial_data.get("context_length", 0),
                method=method,
                trial_index=trial_index,
                execution_order=(
                    f"{trial_data.get('context_length', 0)}_{method}",
                    f"trial_{trial_index}"
                ),
                execution_mode=execution_mode,
                prefill_seconds=trial_data.get("prefill_seconds", 0.0),
                token_latencies_ms=tuple(latencies),
                first_token_ms=trial_data.get("first_token_ms", 0.0),
                throughput_tps=trial_data.get("throughput_tps", 0.0),
                compressed_page_dispatches=compressed_page_dispatches,
                dense_tail_dispatches=dense_tail_dispatches,
                fallback_calls=fallback_calls,
            )
            trials.append(trial)

        return cls(schema_version=schema_version, trials=tuple(trials))


def validate_speed_trials(artifact: RawSpeedArtifact) -> List[str]:
    """Validate speed trial artifact against promotion requirements.

    Returns a list of validation error messages. Empty list means validation passed.
    """
    errors = []
    
    # Check schema version
    if artifact.schema_version != 1:
        errors.append(f"Unsupported schema version: {artifact.schema_version}")
    
    # Group trials by context and mode
    context_trials: Dict[int, Dict[str, List[RawSpeedTrial]]] = {}
    for trial in artifact.trials:
        context = trial.context_length
        mode = trial.method
        
        if context not in context_trials:
            context_trials[context] = {"dense": [], "turbo": []}
        context_trials[context][mode].append(trial)
    
    # Validate each required context
    for context in REQUIRED_CONTEXTS:
        if context not in context_trials:
            errors.append(f"Missing required context: {context}")
            continue
        
        modes = context_trials[context]
        
        # Check dense trials
        dense_trials = modes.get("dense", [])
        if len(dense_trials) < REQUIRED_TRIALS_PER_CONTEXT:
            errors.append(
                f"Context {context}: has {len(dense_trials)} dense trials, "
                f"required {REQUIRED_TRIALS_PER_CONTEXT}"
            )
        
        # Check turbo trials
        turbo_trials = modes.get("turbo", [])
        if len(turbo_trials) < REQUIRED_TRIALS_PER_CONTEXT:
            errors.append(
                f"Context {context}: has {len(turbo_trials)} turbo trials, "
                f"required {REQUIRED_TRIALS_PER_CONTEXT}"
            )
        
        # Validate turbo trial details
        for trial in turbo_trials:
            # Check execution mode
            if trial.execution_mode != "metal_strict":
                errors.append(
                    f"Context {context} turbo trial: execution_mode={trial.execution_mode}, "
                    f"required 'metal_strict'"
                )
            
            # Check fallback calls
            if trial.fallback_calls != 0:
                errors.append(
                    f"Context {context} turbo trial: has {trial.fallback_calls} fallback calls, "
                    f"required 0"
                )
            
            # Check token latencies count
            if len(trial.token_latencies_ms) != REQUIRED_TOKEN_LATENCIES:
                errors.append(
                    f"Context {context} turbo trial: has {len(trial.token_latencies_ms)} latencies, "
                    f"required {REQUIRED_TOKEN_LATENCIES}"
                )
            
            # Check for valid timing values
            if trial.prefill_seconds <= 0:
                errors.append(f"Context {context} turbo trial: prefill_seconds <= 0")
            if trial.first_token_ms <= 0:
                errors.append(f"Context {context} turbo trial: first_token_ms <= 0")
            if trial.throughput_tps <= 0:
                errors.append(f"Context {context} turbo trial: throughput_tps <= 0")
            
            # Check for NaN or infinity
            if any(not isinstance(x, (int, float)) or x != x for x in trial.token_latencies_ms):
                errors.append(f"Context {context} turbo trial: contains NaN in token_latencies_ms")
            if any(abs(x) == float('inf') for x in trial.token_latencies_ms):
                errors.append(f"Context {context} turbo trial: contains infinity in token_latencies_ms")
    
    return errors


@dataclass
class SpeedTrialResult:
    """One speed trial for a specific context and mode."""

    context_length: int = 0
    mode: str = ""
    trial: int = 0
    execution_mode: str = ""
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
