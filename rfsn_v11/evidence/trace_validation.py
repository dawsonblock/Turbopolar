"""Typed trace validation for TurboPolar strict evidence.

This module provides fail-safe parsing and validation of execution trace artifacts.
All fields are validated explicitly - no .get() defaults that hide missing data.
"""

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TraceArtifactError(ValueError):
    """Raised when trace artifact validation fails."""

    pass


class EvidenceValidationError(TraceArtifactError):
    """Raised when evidence artifact validation fails."""

    pass


def generate_deterministic_experiment_id(
    run_id: str,
    fixture_id: str,
    context_length: int,
    context_hash: str,
    continuation_hash: str,
    model_revision: str,
    config_hash: str,
) -> str:
    """Generate a deterministic experiment ID from immutable inputs.

    Args:
        run_id: Unique identifier for this benchmark run
        fixture_id: Identifier for the fixture used
        context_length: Length of the context in tokens
        context_hash: SHA-256 hash of the context tokens
        continuation_hash: SHA-256 hash of the continuation tokens
        model_revision: Git revision of the model
        config_hash: Hash of the TurboPolar configuration

    Returns:
        Deterministic SHA-256 hex digest as experiment ID
    """
    # Create canonical JSON representation of inputs
    canonical = json.dumps({
        "run_id": run_id,
        "fixture_id": fixture_id,
        "context_length": context_length,
        "context_hash": context_hash,
        "continuation_hash": continuation_hash,
        "model_revision": model_revision,
        "config_hash": config_hash,
    }, sort_keys=True, separators=(',', ':'))
    
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class FixtureEntry:
    """Exact fixture entry with all required metadata."""
    fixture_id: str
    category: str
    context_tokens_path: str
    continuation_tokens_path: str
    context_sha256: str
    continuation_sha256: str


@dataclass(frozen=True)
class ExactFixtureManifest:
    """Exact fixture manifest for all required contexts."""
    model_id: str
    model_revision: str
    tokenizer_revision: str
    contexts: dict[int, list[FixtureEntry]]

    def get_fixture_for_context(
        self,
        context_length: int,
        category: str | None = None,
    ) -> FixtureEntry | None:
        """Get a specific fixture for a context length and optional category.

        Args:
            context_length: Required context length
            category: Optional category filter

        Returns:
            Fixture entry or None if not found
        """
        if context_length not in self.contexts:
            return None
        
        fixtures = self.contexts[context_length]
        if category:
            fixtures = [f for f in fixtures if f.category == category]
        
        return fixtures[0] if fixtures else None


def validate_artifact_file(
    path: str,
    expected_sha256: str,
    *,
    artifact_name: str,
) -> bytes:
    """Validate an artifact file exists and matches the expected SHA-256 hash.

    Args:
        path: Path to the artifact file
        expected_sha256: Expected SHA-256 hash (64 hex characters)
        artifact_name: Name of the artifact for error messages

    Returns:
        The file contents as bytes

    Raises:
        EvidenceValidationError: If validation fails
    """
    if not path:
        raise EvidenceValidationError(f"{artifact_name} path is missing")
    
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise EvidenceValidationError(
            f"{artifact_name} does not exist: {path}"
        )
    
    try:
        content = artifact_path.read_bytes()
    except OSError as e:
        raise EvidenceValidationError(
            f"{artifact_name} could not be read: {e}"
        ) from e
    
    if len(expected_sha256) != 64:
        raise EvidenceValidationError(
            f"{artifact_name} expected hash must be 64 characters, got {len(expected_sha256)}"
        )
    
    actual_hash = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(actual_hash, expected_sha256):
        raise EvidenceValidationError(
            f"{artifact_name} hash mismatch: expected {expected_sha256[:16]}..., got {actual_hash[:16]}..."
        )
    
    return content


@dataclass(frozen=True)
class ParsedOperationTrace:
    """Immutable record of one kernel operation with explicit validation."""

    experiment_id: str
    context_length: int
    fixture_id: str
    layer_index: int
    decode_step: int
    operation: str
    page_index: int | None
    kernel_name: str
    execution_mode: str
    metal_requested: bool
    metal_executed: bool
    fallback_used: bool
    fallback_reason: str | None
    output_evaluated: bool
    expected_tokens: int
    processed_tokens: int
    decode_ordinal: int = 0
    cache_offset_before: int = 0


@dataclass(frozen=True)
class ParsedAttentionTrace:
    """Immutable record of one attention step with topology validation."""

    experiment_id: str
    context_length: int
    fixture_id: str
    layer_index: int
    decode_step: int
    expected_page_count: int
    page_operations: tuple[ParsedOperationTrace, ...]
    dense_tail_operation: ParsedOperationTrace | None
    decode_ordinal: int = 0
    cache_offset_before: int = 0
    cache_tokens_before: int = 0
    cache_tokens_after: int = 0
    partial_tail_length: int = 0


def _validate_string_field(
    operation: dict[str, Any],
    field_name: str,
    entry_index: int,
) -> str:
    """Validate a required string field."""
    if field_name not in operation:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: operation missing required field '{field_name}'"
        )
    value = operation[field_name]
    if not isinstance(value, str):
        raise TraceArtifactError(
            f"Trace entry {entry_index}: field '{field_name}' must be a string, got {type(value).__name__}"
        )
    if not value:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: field '{field_name}' cannot be empty"
        )
    return value


def _validate_int_field(
    operation: dict[str, Any],
    field_name: str,
    entry_index: int,
    *,
    min_value: int | None = None,
    allow_negative: bool = False,
) -> int:
    """Validate a required integer field."""
    if field_name not in operation:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: operation missing required field '{field_name}'"
        )
    value = operation[field_name]
    if not isinstance(value, int):
        raise TraceArtifactError(
            f"Trace entry {entry_index}: field '{field_name}' must be an integer, got {type(value).__name__}"
        )
    if not allow_negative and value < 0:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: field '{field_name}' cannot be negative, got {value}"
        )
    if min_value is not None and value < min_value:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: field '{field_name}' must be >= {min_value}, got {value}"
        )
    return value


def _validate_bool_field(
    operation: dict[str, Any],
    field_name: str,
    entry_index: int,
) -> bool:
    """Validate a required boolean field."""
    if field_name not in operation:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: operation missing required field '{field_name}'"
        )
    value = operation[field_name]
    if not isinstance(value, bool):
        raise TraceArtifactError(
            f"Trace entry {entry_index}: field '{field_name}' must be a boolean, got {type(value).__name__}"
        )
    return value


def _validate_optional_string_field(
    operation: dict[str, Any],
    field_name: str,
    entry_index: int,
) -> str | None:
    """Validate an optional string field."""
    if field_name not in operation:
        return None
    value = operation[field_name]
    if value is None:
        return None
    if not isinstance(value, str):
        raise TraceArtifactError(
            f"Trace entry {entry_index}: field '{field_name}' must be a string or null, got {type(value).__name__}"
        )
    return value


def _validate_optional_int_field(
    operation: dict[str, Any],
    field_name: str,
    entry_index: int,
    *,
    default: int = 0,
) -> int:
    """Validate an optional integer field with a default value."""
    if field_name not in operation:
        return default
    value = operation[field_name]
    if value is None:
        return default
    if not isinstance(value, int):
        raise TraceArtifactError(
            f"Trace entry {entry_index}: field '{field_name}' must be an integer or null, got {type(value).__name__}"
        )
    return value


def _validate_nullable_int_field(
    operation: dict[str, Any],
    field_name: str,
    entry_index: int,
) -> int | None:
    """Validate an integer field that can be null (no default)."""
    if field_name not in operation:
        return None
    value = operation[field_name]
    if value is None:
        return None
    if not isinstance(value, int):
        raise TraceArtifactError(
            f"Trace entry {entry_index}: field '{field_name}' must be an integer or null, got {type(value).__name__}"
        )
    return value


def _parse_operation_trace(
    operation: dict[str, Any],
    entry_index: int,
    context_length: int,
    fixture_id: str,
) -> ParsedOperationTrace:
    """Parse and validate a single operation trace."""
    experiment_id = _validate_string_field(operation, "experiment_id", entry_index)
    context_length_field = _validate_int_field(operation, "context_length", entry_index, min_value=1)
    if context_length_field != context_length:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: operation context_length {context_length_field} "
            f"does not match trace context_length {context_length}"
        )
    fixture_id_field = _validate_string_field(operation, "fixture_id", entry_index)
    if fixture_id_field != fixture_id:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: operation fixture_id '{fixture_id_field}' "
            f"does not match trace fixture_id '{fixture_id}'"
        )
    decode_step = _validate_int_field(operation, "decode_step", entry_index, min_value=0)
    decode_ordinal = _validate_optional_int_field(operation, "decode_ordinal", entry_index, default=0)
    cache_offset_before = _validate_optional_int_field(operation, "cache_offset_before", entry_index, default=decode_step)
    layer_index = _validate_int_field(operation, "layer_index", entry_index, min_value=0)
    operation_name = _validate_string_field(operation, "operation", entry_index)
    
    # For compressed_page, page_index is required (not optional)
    if operation_name == "compressed_page":
        if "page_index" not in operation:
            raise TraceArtifactError(
                f"Trace entry {entry_index}: compressed_page operation must have page_index"
            )
        page_index = _validate_int_field(operation, "page_index", entry_index, min_value=0)
    elif operation_name == "dense_tail":
        # dense_tail must have page_index = null
        page_index = _validate_nullable_int_field(operation, "page_index", entry_index)
        if page_index is not None:
            raise TraceArtifactError(
                f"Trace entry {entry_index}: dense_tail operation must have page_index=null, got {page_index}"
            )
    else:
        # Other operations (merge, finalize) have optional page_index
        page_index = _validate_optional_int_field(operation, "page_index", entry_index)
    
    kernel_name = _validate_string_field(operation, "kernel_name", entry_index)
    execution_mode = _validate_string_field(operation, "execution_mode", entry_index)
    metal_requested = _validate_bool_field(operation, "metal_requested", entry_index)
    metal_executed = _validate_bool_field(operation, "metal_executed", entry_index)
    fallback_used = _validate_bool_field(operation, "fallback_used", entry_index)
    fallback_reason = _validate_optional_string_field(operation, "fallback_reason", entry_index)
    output_evaluated = _validate_bool_field(operation, "output_evaluated", entry_index)
    expected_tokens = _validate_int_field(operation, "expected_tokens", entry_index, min_value=1)
    processed_tokens = _validate_int_field(operation, "processed_tokens", entry_index, min_value=0)

    # Validate operation-specific constraints
    valid_operations = {"compressed_page", "dense_tail", "merge", "finalize"}
    if operation_name not in valid_operations:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: invalid operation '{operation_name}', "
            f"must be one of {valid_operations}"
        )
    
    # For compressed_page, validate page_index is non-negative
    if operation_name == "compressed_page" and page_index < 0:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: page_index cannot be negative, got {page_index}"
        )

    if fallback_used and not fallback_reason:
        raise TraceArtifactError(
            f"Trace entry {entry_index}: fallback_used=True requires fallback_reason"
        )

    return ParsedOperationTrace(
        experiment_id=experiment_id,
        context_length=context_length,
        fixture_id=fixture_id,
        decode_step=decode_step,
        decode_ordinal=decode_ordinal,
        cache_offset_before=cache_offset_before,
        layer_index=layer_index,
        operation=operation_name,
        page_index=page_index,
        kernel_name=kernel_name,
        execution_mode=execution_mode,
        metal_requested=metal_requested,
        metal_executed=metal_executed,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        output_evaluated=output_evaluated,
        expected_tokens=expected_tokens,
        processed_tokens=processed_tokens,
    )


def _parse_attention_trace(entry: dict[str, Any], index: int) -> ParsedAttentionTrace:
    """Parse and validate a single attention trace entry."""
    if not isinstance(entry, dict):
        raise TraceArtifactError(
            f"Trace entry {index} must be an object, got {type(entry).__name__}"
        )

    experiment_id = _validate_string_field(entry, "experiment_id", index)
    context_length = _validate_int_field(entry, "context_length", index, min_value=1)
    fixture_id = _validate_string_field(entry, "fixture_id", index)
    decode_step = _validate_int_field(entry, "decode_step", index, min_value=0)
    # New fields with defaults for backward compatibility
    decode_ordinal = _validate_optional_int_field(entry, "decode_ordinal", index, default=0)
    cache_offset_before = _validate_optional_int_field(entry, "cache_offset_before", index, default=decode_step)
    layer_index = _validate_int_field(entry, "layer_index", index, min_value=0)
    cache_tokens_before = _validate_optional_int_field(entry, "cache_tokens_before", index, default=0)
    cache_tokens_after = _validate_optional_int_field(entry, "cache_tokens_after", index, default=0)
    partial_tail_length = _validate_optional_int_field(entry, "partial_tail_length", index, default=0)
    expected_page_count = _validate_int_field(entry, "expected_page_count", index, min_value=0)

    # Validate page_traces
    if "page_traces" not in entry:
        raise TraceArtifactError(
            f"Trace entry {index}: missing required field 'page_traces'"
        )
    page_traces_raw = entry["page_traces"]
    if not isinstance(page_traces_raw, list):
        raise TraceArtifactError(
            f"Trace entry {index}: page_traces must be a list, got {type(page_traces_raw).__name__}"
        )

    page_operations = []
    for i, page_trace in enumerate(page_traces_raw):
        if not isinstance(page_trace, dict):
            raise TraceArtifactError(
                f"Trace entry {index}: page_traces[{i}] must be an object, got {type(page_trace).__name__}"
            )
        try:
            parsed_op = _parse_operation_trace(page_trace, index, context_length, fixture_id)
            page_operations.append(parsed_op)
        except TraceArtifactError as e:
            raise TraceArtifactError(
                f"Trace entry {index}: page_traces[{i}] validation failed: {e}"
            ) from e

    # Validate dense_tail_trace
    if "dense_tail_trace" not in entry:
        raise TraceArtifactError(
            f"Trace entry {index}: missing required field 'dense_tail_trace'"
        )
    dense_tail_raw = entry["dense_tail_trace"]
    dense_tail_operation = None
    if dense_tail_raw is not None:
        if not isinstance(dense_tail_raw, dict):
            raise TraceArtifactError(
                f"Trace entry {index}: dense_tail_trace must be an object or null, got {type(dense_tail_raw).__name__}"
            )
        try:
            dense_tail_operation = _parse_operation_trace(dense_tail_raw, index, context_length, fixture_id)
        except TraceArtifactError as e:
            raise TraceArtifactError(
                f"Trace entry {index}: dense_tail_trace validation failed: {e}"
            ) from e

    return ParsedAttentionTrace(
        experiment_id=experiment_id,
        context_length=context_length,
        fixture_id=fixture_id,
        decode_step=decode_step,
        decode_ordinal=decode_ordinal,
        cache_offset_before=cache_offset_before,
        cache_tokens_before=cache_tokens_before,
        cache_tokens_after=cache_tokens_after,
        partial_tail_length=partial_tail_length,
        layer_index=layer_index,
        expected_page_count=expected_page_count,
        page_operations=tuple(page_operations),
        dense_tail_operation=dense_tail_operation,
    )


def parse_trace_artifact(raw: Any) -> list[ParsedAttentionTrace]:
    """Parse and validate a complete trace artifact.

    Args:
        raw: Raw JSON-decoded trace artifact (must be a list)

    Returns:
        List of validated ParsedAttentionTrace objects

    Raises:
        TraceArtifactError: If any validation fails
    """
    if not isinstance(raw, list):
        raise TraceArtifactError("Trace artifact must contain a JSON array")
    if not raw:
        raise TraceArtifactError("Trace artifact contains no traces")

    parsed = []
    for index, entry in enumerate(raw):
        try:
            parsed.append(_parse_attention_trace(entry, index))
        except TraceArtifactError as e:
            raise TraceArtifactError(
                f"Trace entry {index} validation failed: {e}"
            ) from e

    return parsed


class TraceTopologyError(TraceArtifactError):
    """Raised when trace topology validation fails."""

    pass


def validate_trace_topology(
    traces: list[ParsedAttentionTrace],
    model_layer_count: int,
    required_contexts: set[int],
    requested_positions_per_context: int,
) -> dict[str, Any]:
    """Validate complete trace topology for all required contexts and positions.

    Args:
        traces: Parsed trace artifacts
        model_layer_count: Number of layers in the model
        required_contexts: Set of required context lengths
        requested_positions_per_context: Number of decode positions per context

    Returns:
        Dictionary with validation results and any failures

    Raises:
        TraceTopologyError: If topology validation fails
    """
    failures: list[str] = []
    topology_stats = {
        "total_traces": len(traces),
        "contexts_found": set(),
        "traces_by_context": {},
        "page_operations_by_context": {},
        "tail_operations_by_context": {},
        "fallback_operations_by_context": {},
        "unevaluated_operations_by_context": {},
    }

    # Group traces by context
    for trace in traces:
        context = trace.context_length
        topology_stats["contexts_found"].add(context)
        if context not in topology_stats["traces_by_context"]:
            topology_stats["traces_by_context"][context] = []
        topology_stats["traces_by_context"][context].append(trace)

    # Check all required contexts are present
    missing_contexts = required_contexts - topology_stats["contexts_found"]
    if missing_contexts:
        failures.append(f"Missing required contexts: {sorted(missing_contexts)}")

    # Validate each context
    for context in required_contexts:
        if context not in topology_stats["traces_by_context"]:
            continue

        context_traces = topology_stats["traces_by_context"][context]
        
        # Check for duplicate (context, decode_ordinal, layer_index) traces
        trace_keys = [(t.context_length, t.decode_ordinal, t.layer_index) for t in context_traces]
        if len(trace_keys) != len(set(trace_keys)):
            duplicates = [k for k in trace_keys if trace_keys.count(k) > 1]
            failures.append(
                f"Context {context}: duplicate attention traces for same identity: {set(duplicates)}"
            )
        
        # Build trace index after duplicate check
        trace_index = {
            (t.decode_ordinal, t.layer_index): t for t in context_traces
        }

        # Check all required decode steps and layers are present
        for decode_ordinal in range(requested_positions_per_context):
            for layer_index in range(model_layer_count):
                key = (decode_ordinal, layer_index)
                if key not in trace_index:
                    failures.append(
                        f"Context {context}: missing trace for decode_ordinal={decode_ordinal}, layer_index={layer_index}"
                    )

        # Require exact trace identity set equality - reject out-of-range traces
        actual_keys = set(trace_index.keys())
        expected_keys = {
            (d, l)
            for d in range(requested_positions_per_context)
            for l in range(model_layer_count)
        }
        if actual_keys != expected_keys:
            extra_keys = actual_keys - expected_keys
            missing_keys = expected_keys - actual_keys
            if extra_keys:
                failures.append(
                    f"Context {context}: out-of-range traces: {extra_keys} "
                    f"(expected 0-{requested_positions_per_context-1} decode_ordinal, 0-{model_layer_count-1} layer_index)"
                )
            if missing_keys:
                failures.append(
                    f"Context {context}: missing traces: {missing_keys}"
                )

        # Recompute totals from traces for reconciliation (P0-16-21)
        trace_page_ops = 0
        trace_tail_ops = 0
        trace_fallback_ops = 0
        trace_unevaluated_ops = 0
        
        for trace in context_traces:
            trace_page_ops += len(trace.page_operations)
            if trace.dense_tail_operation:
                trace_tail_ops += 1
            
            for op in trace.page_operations:
                if op.fallback_used:
                    trace_fallback_ops += 1
                if not op.output_evaluated:
                    trace_unevaluated_ops += 1
            
            if trace.dense_tail_operation:
                op = trace.dense_tail_operation
                if op.fallback_used:
                    trace_fallback_ops += 1
                if not op.output_evaluated:
                    trace_unevaluated_ops += 1
        
        # Store trace-computed totals for later comparison
        topology_stats["trace_computed_page_ops"] = topology_stats.get("trace_computed_page_ops", {})
        topology_stats["trace_computed_page_ops"][context] = trace_page_ops
        topology_stats["trace_computed_tail_ops"] = topology_stats.get("trace_computed_tail_ops", {})
        topology_stats["trace_computed_tail_ops"][context] = trace_tail_ops
        topology_stats["trace_computed_fallback_ops"] = topology_stats.get("trace_computed_fallback_ops", {})
        topology_stats["trace_computed_fallback_ops"][context] = trace_fallback_ops
        topology_stats["trace_computed_unevaluated_ops"] = topology_stats.get("trace_computed_unevaluated_ops", {})
        topology_stats["trace_computed_unevaluated_ops"][context] = trace_unevaluated_ops

        # Validate page indices and counts for each trace
        for trace in context_traces:
            page_indices = [op.page_index for op in trace.page_operations if op.page_index is not None]
            
            # Derive expected page count from cache state, not artifact claim
            BLOCK_SIZE = 64
            PAGE_CAPACITY_BLOCKS = 16  # 16 blocks per storage page = 1024 tokens
            # Use cache_tokens_after (post-append) since attention executes after append
            completed_blocks = trace.cache_tokens_after // BLOCK_SIZE
            expected_pages_from_cache = (completed_blocks + PAGE_CAPACITY_BLOCKS - 1) // PAGE_CAPACITY_BLOCKS
            if len(page_indices) != expected_pages_from_cache:
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"page count {len(page_indices)} != expected from cache state {expected_pages_from_cache} "
                    f"(cache_tokens_after={trace.cache_tokens_after}, completed_blocks={completed_blocks}, artifact claimed {trace.expected_page_count})"
                )

            # Check for duplicate page indices
            if len(page_indices) != len(set(page_indices)):
                duplicates = [idx for idx in page_indices if page_indices.count(idx) > 1]
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"duplicate page indices: {set(duplicates)}"
                )

            # Check page indices are complete and sequential
            if page_indices:
                expected_indices = list(range(trace.expected_page_count))
                if sorted(page_indices) != expected_indices:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                        f"page indices {sorted(page_indices)} != expected {expected_indices}"
                    )

            # Check dense tail presence matches expectation
            has_tail = trace.dense_tail_operation is not None
            expected_tail = trace.partial_tail_length > 0
            if has_tail != expected_tail:
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"dense tail mismatch: has_tail={has_tail}, expected_tail={expected_tail} "
                    f"(partial_tail_length={trace.partial_tail_length})"
                )

            # Collect statistics
            if context not in topology_stats["page_operations_by_context"]:
                topology_stats["page_operations_by_context"][context] = 0
            topology_stats["page_operations_by_context"][context] += len(trace.page_operations)

            if trace.dense_tail_operation:
                if context not in topology_stats["tail_operations_by_context"]:
                    topology_stats["tail_operations_by_context"][context] = 0
                topology_stats["tail_operations_by_context"][context] += 1

            # Check for fallbacks
            for op in trace.page_operations:
                if op.fallback_used:
                    if context not in topology_stats["fallback_operations_by_context"]:
                        topology_stats["fallback_operations_by_context"][context] = 0
                    topology_stats["fallback_operations_by_context"][context] += 1
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"fallback used: {op.fallback_reason}"
                    )

                # Cross-check: operation context_length matches parent
                if op.context_length != trace.context_length:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation context_length {op.context_length} != parent {trace.context_length}"
                    )

                # Cross-check: operation fixture_id matches parent
                if op.fixture_id != trace.fixture_id:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation fixture_id {op.fixture_id} != parent {trace.fixture_id}"
                    )

                # Cross-check: operation layer_index matches parent
                if op.layer_index != trace.layer_index:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation layer_index {op.layer_index} != parent {trace.layer_index}"
                    )

                # Cross-check: operation decode_step matches parent
                if op.decode_step != trace.decode_step:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation decode_step {op.decode_step} != parent {trace.decode_step}"
                    )

                # Cross-check: operation decode_ordinal matches parent
                if op.decode_ordinal != trace.decode_ordinal:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation decode_ordinal {op.decode_ordinal} != parent {trace.decode_ordinal}"
                    )

                # Cross-check: operation cache_offset_before matches parent
                if op.cache_offset_before != trace.cache_offset_before:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation cache_offset_before {op.cache_offset_before} != parent {trace.cache_offset_before}"
                    )
                
                # P1: Require operation-level strict mode
                if op.execution_mode != "metal_strict":
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation execution_mode must be 'metal_strict', got '{op.execution_mode}'"
                    )
                
                # P1: Require metal_requested=True for all operations
                if not op.metal_requested:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation metal_requested must be True, got {op.metal_requested}"
                    )
                
                # P1: Require processed tokens equal expected tokens
                if op.processed_tokens != op.expected_tokens:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation processed_tokens {op.processed_tokens} != expected_tokens {op.expected_tokens}"
                    )
                
                # P1: Require complete parent/nested identity equality
                if op.experiment_id != trace.experiment_id:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, page={op.page_index}: "
                        f"operation experiment_id {op.experiment_id} != parent {trace.experiment_id}"
                    )

            # Cross-check dense tail operation if present
            if trace.dense_tail_operation:
                op = trace.dense_tail_operation
                if op.context_length != trace.context_length:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation context_length {op.context_length} != parent {trace.context_length}"
                    )
                if op.fixture_id != trace.fixture_id:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation fixture_id {op.fixture_id} != parent {trace.fixture_id}"
                    )
                if op.layer_index != trace.layer_index:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation layer_index {op.layer_index} != parent {trace.layer_index}"
                    )
                if op.decode_step != trace.decode_step:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation decode_step {op.decode_step} != parent {trace.decode_step}"
                    )
                if op.decode_ordinal != trace.decode_ordinal:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation decode_ordinal {op.decode_ordinal} != parent {trace.decode_ordinal}"
                    )
                if op.cache_offset_before != trace.cache_offset_before:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation cache_offset_before {op.cache_offset_before} != parent {trace.cache_offset_before}"
                    )
                
                # P1: Require operation-level strict mode for dense_tail
                if op.execution_mode != "metal_strict":
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation execution_mode must be 'metal_strict', got '{op.execution_mode}'"
                    )
                
                # P1: Require metal_requested=True for dense_tail
                if not op.metal_requested:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation metal_requested must be True, got {op.metal_requested}"
                    )
                
                # P1: Require processed tokens equal expected tokens for dense_tail
                if op.processed_tokens != op.expected_tokens:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation processed_tokens {op.processed_tokens} != expected_tokens {op.expected_tokens}"
                    )
                
                # P1: Require complete parent/nested identity equality for dense_tail
                if op.experiment_id != trace.experiment_id:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, dense_tail: "
                        f"operation experiment_id {op.experiment_id} != parent {trace.experiment_id}"
                    )

            # Cache state consistency validation
            # P1: Validate cache before/after equations
            if trace.cache_offset_before != trace.cache_tokens_before:
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"cache_offset_before {trace.cache_offset_before} != cache_tokens_before {trace.cache_tokens_before}"
                )
            
            if trace.decode_step != trace.cache_offset_before:
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"decode_step {trace.decode_step} != cache_offset_before {trace.cache_offset_before}"
                )
            
            expected_tail_length = trace.cache_tokens_after % 64
            if trace.partial_tail_length != expected_tail_length:
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"partial_tail_length {trace.partial_tail_length} != expected {expected_tail_length} "
                    f"(cache_tokens_after % 64)"
                )
            
            # P1: Validate processed totals against cache size
            total_processed = sum(op.processed_tokens for op in trace.page_operations)
            if trace.dense_tail_operation:
                total_processed += trace.dense_tail_operation.processed_tokens
            if total_processed != trace.cache_tokens_after:
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"sum(processed_tokens) {total_processed} != cache_tokens_after {trace.cache_tokens_after}"
                )

            if trace.dense_tail_operation and trace.dense_tail_operation.fallback_used:
                if context not in topology_stats["fallback_operations_by_context"]:
                    topology_stats["fallback_operations_by_context"][context] = 0
                topology_stats["fallback_operations_by_context"][context] += 1
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"dense tail fallback used: {trace.dense_tail_operation.fallback_reason}"
                )

            # Check for Metal execution
            for op in trace.page_operations:
                if not op.metal_executed:
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, "
                        f"page={op.page_index}: Metal not executed"
                    )

            if trace.dense_tail_operation and not trace.dense_tail_operation.metal_executed:
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"dense tail Metal not executed"
                )

            # Check for output evaluation
            for op in trace.page_operations:
                if not op.output_evaluated:
                    if context not in topology_stats["unevaluated_operations_by_context"]:
                        topology_stats["unevaluated_operations_by_context"][context] = 0
                    topology_stats["unevaluated_operations_by_context"][context] += 1
                    failures.append(
                        f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}, "
                        f"page={op.page_index}: output not evaluated"
                    )

            if trace.dense_tail_operation and not trace.dense_tail_operation.output_evaluated:
                if context not in topology_stats["unevaluated_operations_by_context"]:
                    topology_stats["unevaluated_operations_by_context"][context] = 0
                topology_stats["unevaluated_operations_by_context"][context] += 1
                failures.append(
                    f"Context {context}, step={trace.decode_ordinal}, layer={trace.layer_index}: "
                    f"dense tail output not evaluated"
                )

    if failures:
        raise TraceTopologyError(
            f"Trace topology validation failed with {len(failures)} errors: " + "; ".join(failures[:10])
        )

    return topology_stats
