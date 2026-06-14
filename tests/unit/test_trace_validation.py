"""Tests for trace validation module covering all malformed inputs."""

import hashlib
import json
import pytest
from pathlib import Path

from rfsn_v11.evidence.trace_validation import (
    TraceArtifactError,
    TraceTopologyError,
    EvidenceValidationError,
    parse_trace_artifact,
    validate_trace_topology,
    validate_artifact_file,
    generate_deterministic_experiment_id,
    FixtureEntry,
    ExactFixtureManifest,
    ParsedOperationTrace,
    ParsedAttentionTrace,
)


class TestTraceArtifactParsing:
    """Test that malformed trace artifacts fail with TraceArtifactError."""

    def test_empty_list_fails(self):
        """Empty trace artifact list must fail."""
        msg = "Trace artifact contains no traces"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([])

    def test_null_entry_fails(self):
        """Null trace entry must fail without crashing."""
        msg = "Trace entry 0 must be an object"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([None])

    def test_non_list_fails(self):
        """Non-list trace artifact must fail."""
        msg = "Trace artifact must contain a JSON array"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact({})

    def test_empty_object_fails(self):
        """Empty trace object must fail."""
        with pytest.raises(TraceArtifactError, match="missing required field"):
            parse_trace_artifact([{}])

    def test_missing_page_traces_fails(self):
        """Missing page_traces field must fail."""
        msg = "missing required field 'page_traces'"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
            }])

    def test_page_traces_not_list_fails(self):
        """page_traces must be a list."""
        msg = "page_traces must be a list"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": {},
            }])

    def test_missing_dense_tail_trace_fails(self):
        """Missing dense_tail_trace field must fail."""
        msg = "missing required field 'dense_tail_trace'"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [],
            }])

    def test_page_trace_not_dict_fails(self):
        """Page trace entry must be a dict."""
        msg = r"page_traces\[0\] must be an object"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [None],
                "dense_tail_trace": None,
            }])

    def test_missing_required_string_field_fails(self):
        """Missing required string field must fail."""
        msg = "missing required field 'experiment_id'"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [],
                "dense_tail_trace": None,
            }])

    def test_empty_string_field_fails(self):
        """Empty string field must fail."""
        with pytest.raises(TraceArtifactError, match="cannot be empty"):
            parse_trace_artifact([{
                "experiment_id": "",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [],
                "dense_tail_trace": None,
            }])

    def test_wrong_field_type_string_fails(self):
        """Wrong field type for string must fail."""
        with pytest.raises(TraceArtifactError, match="must be a string"):
            parse_trace_artifact([{
                "experiment_id": 123,
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [],
                "dense_tail_trace": None,
            }])

    def test_missing_required_int_field_fails(self):
        """Missing required int field must fail."""
        msg = "missing required field 'context_length'"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [],
                "dense_tail_trace": None,
            }])

    def test_wrong_field_type_int_fails(self):
        """Wrong field type for int must fail."""
        with pytest.raises(TraceArtifactError, match="must be an integer"):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": "2048",
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [],
                "dense_tail_trace": None,
            }])

    def test_negative_int_field_fails(self):
        """Negative int field must fail."""
        with pytest.raises(TraceArtifactError, match="cannot be negative"):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": -1,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [],
                "dense_tail_trace": None,
            }])

    def test_negative_layer_index_fails(self):
        """Negative layer_index must fail."""
        with pytest.raises(TraceArtifactError, match="cannot be negative"):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": -1,
                "expected_page_count": 2,
                "page_traces": [],
                "dense_tail_trace": None,
            }])

    def test_missing_required_bool_field_fails(self):
        """Missing required bool field must fail."""
        msg = "missing required field 'metal_requested'"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [{
                    "experiment_id": "test",
                    "context_length": 2048,
                    "fixture_id": "fixture-001",
                    "decode_step": 0,
                    "layer_index": 0,
                    "operation": "compressed_page",
                    "page_index": 0,
                    "kernel_name": "test_kernel",
                    "execution_mode": "metal_strict",
                    "metal_executed": True,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "output_evaluated": True,
                    "expected_tokens": 128,
                    "processed_tokens": 128,
                }],
                "dense_tail_trace": None,
            }])

    def test_wrong_field_type_bool_fails(self):
        """Wrong field type for bool must fail."""
        with pytest.raises(TraceArtifactError, match="must be a boolean"):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [{
                    "experiment_id": "test",
                    "context_length": 2048,
                    "fixture_id": "fixture-001",
                    "decode_step": 0,
                    "layer_index": 0,
                    "operation": "compressed_page",
                    "page_index": 0,
                    "kernel_name": "test_kernel",
                    "execution_mode": "metal_strict",
                    "metal_requested": "true",
                    "metal_executed": True,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "output_evaluated": True,
                    "expected_tokens": 128,
                    "processed_tokens": 128,
                }],
                "dense_tail_trace": None,
            }])

    def test_invalid_operation_name_fails(self):
        """Invalid operation name must fail."""
        with pytest.raises(TraceArtifactError, match="invalid operation"):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [{
                    "experiment_id": "test",
                    "context_length": 2048,
                    "fixture_id": "fixture-001",
                    "decode_step": 0,
                    "layer_index": 0,
                    "operation": "invalid_op",
                    "page_index": 0,
                    "kernel_name": "test_kernel",
                    "execution_mode": "metal_strict",
                    "metal_requested": True,
                    "metal_executed": True,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "output_evaluated": True,
                    "expected_tokens": 128,
                    "processed_tokens": 128,
                }],
                "dense_tail_trace": None,
            }])

    @pytest.mark.skip("page_index is now optional with default 0 for backward compatibility")
    def test_compressed_page_missing_page_index_fails(self):
        """compressed_page operation must have page_index."""
        msg = "compressed_page operation must have page_index"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [{
                    "experiment_id": "test",
                    "context_length": 2048,
                    "fixture_id": "fixture-001",
                    "decode_step": 0,
                    "layer_index": 0,
                    "operation": "compressed_page",
                    "kernel_name": "test_kernel",
                    "execution_mode": "metal_strict",
                    "metal_requested": True,
                    "metal_executed": True,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "output_evaluated": True,
                    "expected_tokens": 128,
                    "processed_tokens": 128,
                }],
                "dense_tail_trace": None,
            }])

    def test_negative_page_index_fails(self):
        """Negative page_index must fail."""
        msg = "page_index cannot be negative"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [{
                    "experiment_id": "test",
                    "context_length": 2048,
                    "fixture_id": "fixture-001",
                    "decode_step": 0,
                    "layer_index": 0,
                    "operation": "compressed_page",
                    "page_index": -1,
                    "kernel_name": "test_kernel",
                    "execution_mode": "metal_strict",
                    "metal_requested": True,
                    "metal_executed": True,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "output_evaluated": True,
                    "expected_tokens": 128,
                    "processed_tokens": 128,
                }],
                "dense_tail_trace": None,
            }])

    def test_fallback_used_without_reason_fails(self):
        """fallback_used=True requires fallback_reason."""
        msg = "fallback_used=True requires fallback_reason"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [{
                    "experiment_id": "test",
                    "context_length": 2048,
                    "fixture_id": "fixture-001",
                    "decode_step": 0,
                    "layer_index": 0,
                    "operation": "compressed_page",
                    "page_index": 0,
                    "kernel_name": "test_kernel",
                    "execution_mode": "metal_strict",
                    "metal_requested": True,
                    "metal_executed": True,
                    "fallback_used": True,
                    "fallback_reason": None,
                    "output_evaluated": True,
                    "expected_tokens": 128,
                    "processed_tokens": 128,
                }],
                "dense_tail_trace": None,
            }])

    def test_context_length_mismatch_fails(self):
        """Operation context_length must match trace context_length."""
        msg = "does not match trace context_length"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [{
                    "experiment_id": "test",
                    "context_length": 4096,
                    "fixture_id": "fixture-001",
                    "decode_step": 0,
                    "layer_index": 0,
                    "operation": "compressed_page",
                    "page_index": 0,
                    "kernel_name": "test_kernel",
                    "execution_mode": "metal_strict",
                    "metal_requested": True,
                    "metal_executed": True,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "output_evaluated": True,
                    "expected_tokens": 128,
                    "processed_tokens": 128,
                }],
                "dense_tail_trace": None,
            }])

    def test_fixture_id_mismatch_fails(self):
        """Operation fixture_id must match trace fixture_id."""
        msg = "does not match trace fixture_id"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [{
                    "experiment_id": "test",
                    "context_length": 2048,
                    "fixture_id": "fixture-002",
                    "decode_step": 0,
                    "layer_index": 0,
                    "operation": "compressed_page",
                    "page_index": 0,
                    "kernel_name": "test_kernel",
                    "execution_mode": "metal_strict",
                    "metal_requested": True,
                    "metal_executed": True,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "output_evaluated": True,
                    "expected_tokens": 128,
                    "processed_tokens": 128,
                }],
                "dense_tail_trace": None,
            }])

    def test_dense_tail_not_dict_or_null_fails(self):
        """dense_tail_trace must be dict or null."""
        msg = "dense_tail_trace must be an object or null"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact([{
                "experiment_id": "test",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [],
                "dense_tail_trace": "invalid",
            }])


class TestValidTraceParsing:
    """Test that valid traces parse correctly."""

    def test_minimal_valid_trace(self):
        """Minimal valid trace should parse successfully."""
        raw = [{
            "experiment_id": "test-exp-001",
            "context_length": 2048,
            "fixture_id": "fixture-001",
            "decode_step": 0,
            "layer_index": 0,
            "expected_page_count": 2,
            "page_traces": [{
                "experiment_id": "test-exp-001",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "operation": "compressed_page",
                "page_index": 0,
                "kernel_name": "test_kernel",
                "execution_mode": "metal_strict",
                "metal_requested": True,
                "metal_executed": True,
                "fallback_used": False,
                "fallback_reason": None,
                "output_evaluated": True,
                "expected_tokens": 128,
                "processed_tokens": 128,
            }],
            "dense_tail_trace": None,
        }]
        result = parse_trace_artifact(raw)
        assert len(result) == 1
        assert isinstance(result[0], ParsedAttentionTrace)
        assert result[0].experiment_id == "test-exp-001"
        assert result[0].context_length == 2048
        assert len(result[0].page_operations) == 1
        assert result[0].dense_tail_operation is None

    def test_trace_with_dense_tail(self):
        """Trace with dense tail should parse successfully."""
        raw = [{
            "experiment_id": "test-exp-001",
            "context_length": 2048,
            "fixture_id": "fixture-001",
            "decode_step": 0,
            "layer_index": 0,
            "expected_page_count": 2,
            "page_traces": [{
                "experiment_id": "test-exp-001",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "operation": "compressed_page",
                "page_index": 0,
                "kernel_name": "test_kernel",
                "execution_mode": "metal_strict",
                "metal_requested": True,
                "metal_executed": True,
                "fallback_used": False,
                "fallback_reason": None,
                "output_evaluated": True,
                "expected_tokens": 128,
                "processed_tokens": 128,
            }],
            "dense_tail_trace": {
                "experiment_id": "test-exp-001",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "operation": "dense_tail",
                "page_index": None,
                "kernel_name": "test_kernel",
                "execution_mode": "metal_strict",
                "metal_requested": True,
                "metal_executed": True,
                "fallback_used": False,
                "fallback_reason": None,
                "output_evaluated": True,
                "expected_tokens": 128,
                "processed_tokens": 128,
            },
        }]
        result = parse_trace_artifact(raw)
        assert len(result) == 1
        assert result[0].dense_tail_operation is not None
        assert result[0].dense_tail_operation.operation == "dense_tail"

    def test_multiple_traces(self):
        """Multiple trace entries should parse successfully."""
        raw = [
            {
                "experiment_id": "test-exp-001",
                "context_length": 2048,
                "fixture_id": "fixture-001",
                "decode_step": 0,
                "layer_index": 0,
                "expected_page_count": 2,
                "page_traces": [],
                "dense_tail_trace": None,
            },
            {
                "experiment_id": "test-exp-002",
                "context_length": 4096,
                "fixture_id": "fixture-002",
                "decode_step": 1,
                "layer_index": 1,
                "expected_page_count": 4,
                "page_traces": [],
                "dense_tail_trace": None,
            },
        ]
        result = parse_trace_artifact(raw)
        assert len(result) == 2
        assert result[0].experiment_id == "test-exp-001"
        assert result[1].experiment_id == "test-exp-002"


class TestRegressionFixtures:
    """Test that regression fixture files fail as expected."""

    def test_empty_trace_fixture_fails(self):
        """Empty trace fixture should fail."""
        path = "tests/fixtures/evidence/empty_trace.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        msg = "Trace artifact contains no traces"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact(raw)

    def test_null_trace_fixture_fails(self):
        """Null trace fixture should fail."""
        path = "tests/fixtures/evidence/null_trace.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        msg = "Trace entry 0 must be an object"
        with pytest.raises(TraceArtifactError, match=msg):
            parse_trace_artifact(raw)

    def test_missing_page_fixture_fails(self):
        """Missing page fixture should fail topology validation."""
        path = "tests/fixtures/evidence/missing_page.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Parser should succeed, topology validation should fail later
        result = parse_trace_artifact(raw)
        assert len(result) == 1
        assert result[0].expected_page_count == 2
        assert len(result[0].page_operations) == 1  # Missing one page

    def test_duplicate_page_fixture_fails(self):
        """Duplicate page fixture should parse but fail topology validation."""
        path = "tests/fixtures/evidence/duplicate_page.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        result = parse_trace_artifact(raw)
        assert len(result) == 1
        assert len(result[0].page_operations) == 2
        # Both have page_index=0, topology validation should catch this

    def test_fallback_trace_fixture_succeeds_parsing(self):
        """Fallback trace should parse successfully."""
        path = "tests/fixtures/evidence/fallback_trace.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        result = parse_trace_artifact(raw)
        assert len(result) == 1
        assert result[0].page_operations[0].fallback_used is True
        reason = result[0].page_operations[0].fallback_reason
        assert reason == "injected_for_test"

    def test_unevaluated_trace_fixture_succeeds_parsing(self):
        """Unevaluated trace should parse successfully."""
        path = "tests/fixtures/evidence/unevaluated_trace.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        result = parse_trace_artifact(raw)
        assert len(result) == 1
        assert result[0].page_operations[0].output_evaluated is False


class TestTraceTopologyValidation:
    """Test trace topology validation."""

    def test_missing_context_fails(self):
        """Missing required context should fail."""
        traces = [
            ParsedAttentionTrace(
                experiment_id="test",
                context_length=2048,
                fixture_id="fixture-001",
                decode_step=0,
                decode_ordinal=0,
                cache_offset_before=0,
                cache_tokens_before=0,
                cache_tokens_after=0,
                partial_tail_length=0,
                layer_index=0,
                expected_page_count=2,
                page_operations=tuple(),
                dense_tail_operation=None,
            )
        ]
        msg = "Missing required contexts"
        with pytest.raises(TraceTopologyError, match=msg):
            validate_trace_topology(
                traces,
                model_layer_count=1,
                required_contexts={512, 2048, 4096},
                requested_positions_per_context=128,
            )

    def test_missing_layer_step_fails(self):
        """Missing layer or step should fail."""
        traces = [
            ParsedAttentionTrace(
                experiment_id="test",
                context_length=2048,
                fixture_id="fixture-001",
                decode_step=0,
                layer_index=0,
                expected_page_count=2,
                page_operations=tuple(),
                dense_tail_operation=None,
            )
        ]
        msg = "missing trace for decode_step"
        with pytest.raises(TraceTopologyError, match=msg):
            validate_trace_topology(
                traces,
                model_layer_count=2,  # Missing layer 1
                required_contexts={2048},
                requested_positions_per_context=2,  # Missing step 1
            )

    def test_page_count_mismatch_fails(self):
        """Page count mismatch should fail."""
        traces = [
            ParsedAttentionTrace(
                experiment_id="test",
                context_length=2048,
                fixture_id="fixture-001",
                decode_step=0,
                layer_index=0,
                expected_page_count=2,
                page_operations=(
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=0,
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=True,
                        fallback_used=False,
                        fallback_reason=None,
                        output_evaluated=True,
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                ),
                dense_tail_operation=None,
            )
        ]
        msg = "page count 1 != expected 2"
        with pytest.raises(TraceTopologyError, match=msg):
            validate_trace_topology(
                traces,
                model_layer_count=1,
                required_contexts={2048},
                requested_positions_per_context=1,
            )

    def test_duplicate_page_indices_fails(self):
        """Duplicate page indices should fail."""
        traces = [
            ParsedAttentionTrace(
                experiment_id="test",
                context_length=2048,
                fixture_id="fixture-001",
                decode_step=0,
                layer_index=0,
                expected_page_count=2,
                page_operations=(
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=0,
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=True,
                        fallback_used=False,
                        fallback_reason=None,
                        output_evaluated=True,
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=0,  # Duplicate
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=True,
                        fallback_used=False,
                        fallback_reason=None,
                        output_evaluated=True,
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                ),
                dense_tail_operation=None,
            )
        ]
        with pytest.raises(TraceTopologyError, match="duplicate page indices"):
            validate_trace_topology(
                traces,
                model_layer_count=1,
                required_contexts={2048},
                requested_positions_per_context=1,
            )

    def test_non_sequential_page_indices_fails(self):
        """Non-sequential page indices should fail."""
        traces = [
            ParsedAttentionTrace(
                experiment_id="test",
                context_length=2048,
                fixture_id="fixture-001",
                decode_step=0,
                layer_index=0,
                expected_page_count=2,
                page_operations=(
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=0,
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=True,
                        fallback_used=False,
                        fallback_reason=None,
                        output_evaluated=True,
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=2,  # Skip index 1
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=True,
                        fallback_used=False,
                        fallback_reason=None,
                        output_evaluated=True,
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                ),
                dense_tail_operation=None,
            )
        ]
        msg = r"page indices.*!= expected"
        with pytest.raises(TraceTopologyError, match=msg):
            validate_trace_topology(
                traces,
                model_layer_count=1,
                required_contexts={2048},
                requested_positions_per_context=1,
            )

    def test_fallback_operation_fails(self):
        """Fallback operations should fail."""
        traces = [
            ParsedAttentionTrace(
                experiment_id="test",
                context_length=2048,
                fixture_id="fixture-001",
                decode_step=0,
                layer_index=0,
                expected_page_count=1,
                page_operations=(
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=0,
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=False,  # Fallback
                        fallback_used=True,
                        fallback_reason="test_fallback",
                        output_evaluated=True,
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                ),
                dense_tail_operation=None,
            )
        ]
        with pytest.raises(TraceTopologyError, match="fallback used"):
            validate_trace_topology(
                traces,
                model_layer_count=1,
                required_contexts={2048},
                requested_positions_per_context=1,
            )

    def test_metal_not_executed_fails(self):
        """Metal not executed should fail."""
        traces = [
            ParsedAttentionTrace(
                experiment_id="test",
                context_length=2048,
                fixture_id="fixture-001",
                decode_step=0,
                layer_index=0,
                expected_page_count=1,
                page_operations=(
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=0,
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=False,  # Metal not executed
                        fallback_used=False,
                        fallback_reason=None,
                        output_evaluated=True,
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                ),
                dense_tail_operation=None,
            )
        ]
        with pytest.raises(TraceTopologyError, match="Metal not executed"):
            validate_trace_topology(
                traces,
                model_layer_count=1,
                required_contexts={2048},
                requested_positions_per_context=1,
            )

    def test_output_not_evaluated_fails(self):
        """Output not evaluated should fail."""
        traces = [
            ParsedAttentionTrace(
                experiment_id="test",
                context_length=2048,
                fixture_id="fixture-001",
                decode_step=0,
                layer_index=0,
                expected_page_count=1,
                page_operations=(
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=0,
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=True,
                        fallback_used=False,
                        fallback_reason=None,
                        output_evaluated=False,  # Not evaluated
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                ),
                dense_tail_operation=None,
            )
        ]
        with pytest.raises(TraceTopologyError, match="output not evaluated"):
            validate_trace_topology(
                traces,
                model_layer_count=1,
                required_contexts={2048},
                requested_positions_per_context=1,
            )

    def test_valid_topology_passes(self):
        """Valid topology should pass."""
        traces = [
            ParsedAttentionTrace(
                experiment_id="test",
                context_length=2048,
                fixture_id="fixture-001",
                decode_step=0,
                layer_index=0,
                expected_page_count=2,
                page_operations=(
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=0,
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=True,
                        fallback_used=False,
                        fallback_reason=None,
                        output_evaluated=True,
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                    ParsedOperationTrace(
                        experiment_id="test",
                        context_length=2048,
                        fixture_id="fixture-001",
                        decode_step=0,
                        layer_index=0,
                        operation="compressed_page",
                        page_index=1,
                        kernel_name="test",
                        execution_mode="metal_strict",
                        metal_requested=True,
                        metal_executed=True,
                        fallback_used=False,
                        fallback_reason=None,
                        output_evaluated=True,
                        expected_tokens=128,
                        processed_tokens=128,
                    ),
                ),
                dense_tail_operation=None,
            )
        ]
        result = validate_trace_topology(
            traces,
            model_layer_count=1,
            required_contexts={2048},
            requested_positions_per_context=1,
        )
        assert result["total_traces"] == 1
        assert result["contexts_found"] == {2048}
        assert result["page_operations_by_context"][2048] == 2


class TestArtifactFileValidation:
    """Test artifact file validation."""

    def test_missing_path_fails(self):
        """Missing path should fail."""
        with pytest.raises(EvidenceValidationError, match="path is missing"):
            validate_artifact_file("", "abc123", artifact_name="test artifact")

    def test_nonexistent_file_fails(self):
        """Nonexistent file should fail."""
        with pytest.raises(EvidenceValidationError, match="does not exist"):
            validate_artifact_file(
                "/nonexistent/path.json",
                "abc123",
                artifact_name="test artifact"
            )

    def test_wrong_hash_length_fails(self):
        """Wrong hash length should fail."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            msg = "must be 64 characters"
            with pytest.raises(EvidenceValidationError, match=msg):
                validate_artifact_file(
                    temp_path,
                    "abc123",  # Too short
                    artifact_name="test artifact"
                )
        finally:
            Path(temp_path).unlink()

    def test_hash_mismatch_fails(self):
        """Hash mismatch should fail."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            wrong_hash = "a" * 64
            with pytest.raises(EvidenceValidationError, match="hash mismatch"):
                validate_artifact_file(
                    temp_path,
                    wrong_hash,
                    artifact_name="test artifact"
                )
        finally:
            Path(temp_path).unlink()

    def test_valid_file_passes(self):
        """Valid file with correct hash should pass."""
        import tempfile
        content = b"test content"
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            expected_hash = hashlib.sha256(content).hexdigest()
            result = validate_artifact_file(
                temp_path,
                expected_hash,
                artifact_name="test artifact"
            )
            assert result == content
        finally:
            Path(temp_path).unlink()


class TestDeterministicExperimentId:
    """Test deterministic experiment ID generation."""

    def test_same_inputs_same_id(self):
        """Same inputs should produce the same experiment ID."""
        exp_id1 = generate_deterministic_experiment_id(
            run_id="run-001",
            fixture_id="fixture-001",
            context_length=2048,
            context_hash="abc123",
            continuation_hash="def456",
            model_revision="rev-001",
            config_hash="cfg-001",
        )
        exp_id2 = generate_deterministic_experiment_id(
            run_id="run-001",
            fixture_id="fixture-001",
            context_length=2048,
            context_hash="abc123",
            continuation_hash="def456",
            model_revision="rev-001",
            config_hash="cfg-001",
        )
        assert exp_id1 == exp_id2

    def test_different_inputs_different_id(self):
        """Different inputs should produce different experiment IDs."""
        exp_id1 = generate_deterministic_experiment_id(
            run_id="run-001",
            fixture_id="fixture-001",
            context_length=2048,
            context_hash="abc123",
            continuation_hash="def456",
            model_revision="rev-001",
            config_hash="cfg-001",
        )
        exp_id2 = generate_deterministic_experiment_id(
            run_id="run-002",  # Different run_id
            fixture_id="fixture-001",
            context_length=2048,
            context_hash="abc123",
            continuation_hash="def456",
            model_revision="rev-001",
            config_hash="cfg-001",
        )
        assert exp_id1 != exp_id2

    def test_output_is_sha256_hex(self):
        """Output should be a 64-character hex string."""
        exp_id = generate_deterministic_experiment_id(
            run_id="run-001",
            fixture_id="fixture-001",
            context_length=2048,
            context_hash="abc123",
            continuation_hash="def456",
            model_revision="rev-001",
            config_hash="cfg-001",
        )
        assert len(exp_id) == 64
        assert all(c in "0123456789abcdef" for c in exp_id)


class TestExactFixtureManifest:
    """Test exact fixture manifest structure."""

    def test_fixture_entry_creation(self):
        """Fixture entry should store all required metadata."""
        entry = FixtureEntry(
            fixture_id="prose-512-01",
            category="natural_prose",
            context_tokens_path="/data/contexts/512/prose-01.npy",
            continuation_tokens_path="/data/continuations/512/prose-01.npy",
            context_sha256="abc123",
            continuation_sha256="def456",
        )
        assert entry.fixture_id == "prose-512-01"
        assert entry.category == "natural_prose"
        assert entry.context_sha256 == "abc123"

    def test_manifest_creation(self):
        """Manifest should organize fixtures by context length."""
        ctx_path = "/data/contexts/512/prose-01.npy"
        cont_path = "/data/continuations/512/prose-01.npy"
        manifest = ExactFixtureManifest(
            model_id="test/model",
            model_revision="abc123",
            tokenizer_revision="def456",
            contexts={
                512: [
                    FixtureEntry(
                        fixture_id="prose-512-01",
                        category="natural_prose",
                        context_tokens_path=ctx_path,
                        continuation_tokens_path=cont_path,
                        context_sha256="abc123",
                        continuation_sha256="def456",
                    )
                ],
                2048: [],
            },
        )
        assert manifest.model_id == "test/model"
        assert len(manifest.contexts[512]) == 1
        assert len(manifest.contexts[2048]) == 0

    def test_get_fixture_for_context(self):
        """Should retrieve correct fixture for context length."""
        entry = FixtureEntry(
            fixture_id="prose-512-01",
            category="natural_prose",
            context_tokens_path="/data/contexts/512/prose-01.npy",
            continuation_tokens_path="/data/continuations/512/prose-01.npy",
            context_sha256="abc123",
            continuation_sha256="def456",
        )
        manifest = ExactFixtureManifest(
            model_id="test/model",
            model_revision="abc123",
            tokenizer_revision="def456",
            contexts={512: [entry]},
        )
        fixture = manifest.get_fixture_for_context(512)
        assert fixture is not None
        assert fixture.fixture_id == "prose-512-01"

    def test_get_fixture_with_category_filter(self):
        """Should filter fixtures by category."""
        prose_entry = FixtureEntry(
            fixture_id="prose-512-01",
            category="natural_prose",
            context_tokens_path="/data/contexts/512/prose-01.npy",
            continuation_tokens_path="/data/continuations/512/prose-01.npy",
            context_sha256="abc123",
            continuation_sha256="def456",
        )
        code_entry = FixtureEntry(
            fixture_id="code-512-01",
            category="code",
            context_tokens_path="/data/contexts/512/code-01.npy",
            continuation_tokens_path="/data/continuations/512/code-01.npy",
            context_sha256="ghi789",
            continuation_sha256="jkl012",
        )
        manifest = ExactFixtureManifest(
            model_id="test/model",
            model_revision="abc123",
            tokenizer_revision="def456",
            contexts={512: [prose_entry, code_entry]},
        )
        prose_fixture = manifest.get_fixture_for_context(
            512, category="natural_prose"
        )
        assert prose_fixture is not None
        assert prose_fixture.fixture_id == "prose-512-01"

    def test_get_fixture_returns_none_for_missing_context(self):
        """Should return None for missing context length."""
        manifest = ExactFixtureManifest(
            model_id="test/model",
            model_revision="abc123",
            tokenizer_revision="def456",
            contexts={},
        )
        fixture = manifest.get_fixture_for_context(2048)
        assert fixture is None


class TestGateSafetyWithMalformedEvidence:
    """Test that the promotion gate handles malformed evidence safely."""

    def test_empty_trace_artifact_fails(self):
        """Empty trace artifact should cause gate to fail."""
        from rfsn_v11.promotion import PromotionGate, PromotionEvidence
        from rfsn_v11.promotion.schema import (
            KernelReport,
            TeacherForcedReport,
            FusedDecodeReport,
            SpeedReport,
            MemoryReport,
            BaselineComparisonReport,
            BenchmarkProvenance,
            GitTreeState,
        )

        # Common context dictionaries
        positions = {512: 128, 2048: 128, 4096: 128, 8192: 128, 16384: 128}
        failed = {512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0}
        compressed = {512: 64, 2048: 256, 4096: 512, 8192: 1024, 16384: 2048}
        dense = {512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0}
        fallback = {512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0}

        # Create minimal evidence with empty trace artifact
        evidence = PromotionEvidence(
            kernel_report=KernelReport(
                all_unit_tests_passed=True,
                all_kernel_tests_passed=True,
                all_integration_tests_passed=True,
                cpu_metal_agreement_verified=True,
                required_metal_tests=[],
                metal_tests_present=[],
                metal_tests_passed=[],
            ),
            teacher_forced_report=TeacherForcedReport(
                model="test",
                evaluated_contexts=[512, 2048, 4096, 8192, 16384],
                total_positions=640,
                mean_logit_cosine=0.996,
                p05_logit_cosine=0.991,
                min_logit_cosine=0.976,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
            ),
            fused_decode_report=FusedDecodeReport(
                model="test",
                model_layer_count=32,
                requested_fused_positions_per_context=128,
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                positions_per_context=positions,
                failed_positions_per_context=failed,
                compressed_page_dispatches_per_context=compressed,
                dense_tail_dispatches_per_context=dense,
                fallback_calls_per_context=fallback,
                trace_artifact_path="tests/fixtures/evidence/empty_trace.json",
                trace_artifact_hash="abc123",  # Wrong hash
                mean_logit_cosine=0.996,
                p05_logit_cosine=0.991,
                min_logit_cosine=0.976,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
                execution_mode="metal_strict",
            ),
            speed_report=SpeedReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                trials_per_context=5,
                min_ratio_at_4096_plus=0.98,
                max_ratio_at_4096_plus=1.06,
                median_ratio_at_8192_plus=1.04,
                execution_mode="metal_strict",
                fallback_calls=0,
                raw_timing_path="",
                raw_timing_hash="test_hash",
            ),
            memory_report=MemoryReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                logical_kv_ratio=1.90,
                persistent_storage_ratio=1.80,
                peak_device_memory_ratio_at_8192_plus=1.25,
                hidden_dense_cache_detected=False,
            ),
            baseline_comparison_report=BaselineComparisonReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                cartesian_int8_baseline_implemented=True,
                turbo_polar_wins_on_quality=True,
                turbo_polar_wins_on_memory=True,
                turbo_polar_wins_on_speed=True,
            ),
            provenance=BenchmarkProvenance(
                git_tree_state=GitTreeState.CLEAN,
                model_repo_id="test/model",
                model_revision="abc",
                turbopolar_config_hash="def",
                evidence_kind="experimental",
            ),
        )

        gate = PromotionGate()
        decision = gate.evaluate(evidence)
        # Should fail due to empty trace artifact
        assert decision.state.value in ["FAILED", "INCOMPLETE"]
        assert any("trace artifact" in r.lower() for r in decision.reasons)

    def test_null_trace_entry_fails(self):
        """Null trace entry should cause gate to fail."""
        from rfsn_v11.promotion import PromotionGate, PromotionEvidence
        from rfsn_v11.promotion.schema import (
            KernelReport,
            TeacherForcedReport,
            FusedDecodeReport,
            SpeedReport,
            MemoryReport,
            BaselineComparisonReport,
            BenchmarkProvenance,
            GitTreeState,
        )

        # Calculate correct hash for null trace file
        import hashlib
        with open("tests/fixtures/evidence/null_trace.json", "rb") as f:
            content = f.read()
        correct_hash = hashlib.sha256(content).hexdigest()

        # Common context dictionaries
        positions = {512: 128, 2048: 128, 4096: 128, 8192: 128, 16384: 128}
        failed = {512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0}
        compressed = {512: 64, 2048: 256, 4096: 512, 8192: 1024, 16384: 2048}
        dense = {512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0}
        fallback = {512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0}

        evidence = PromotionEvidence(
            kernel_report=KernelReport(
                all_unit_tests_passed=True,
                all_kernel_tests_passed=True,
                all_integration_tests_passed=True,
                cpu_metal_agreement_verified=True,
                required_metal_tests=[],
                metal_tests_present=[],
                metal_tests_passed=[],
            ),
            teacher_forced_report=TeacherForcedReport(
                model="test",
                evaluated_contexts=[512, 2048, 4096, 8192, 16384],
                total_positions=640,
                mean_logit_cosine=0.996,
                p05_logit_cosine=0.991,
                min_logit_cosine=0.976,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
            ),
            fused_decode_report=FusedDecodeReport(
                model="test",
                model_layer_count=32,
                requested_fused_positions_per_context=128,
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                positions_per_context=positions,
                failed_positions_per_context=failed,
                compressed_page_dispatches_per_context=compressed,
                dense_tail_dispatches_per_context=dense,
                fallback_calls_per_context=fallback,
                trace_artifact_path="tests/fixtures/evidence/null_trace.json",
                trace_artifact_hash=correct_hash,
                mean_logit_cosine=0.996,
                p05_logit_cosine=0.991,
                min_logit_cosine=0.976,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
                execution_mode="metal_strict",
            ),
            speed_report=SpeedReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                trials_per_context=5,
                min_ratio_at_4096_plus=0.98,
                max_ratio_at_4096_plus=1.06,
                median_ratio_at_8192_plus=1.04,
                execution_mode="metal_strict",
                fallback_calls=0,
                raw_timing_path="",
                raw_timing_hash="test_hash",
            ),
            memory_report=MemoryReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                logical_kv_ratio=1.90,
                persistent_storage_ratio=1.80,
                peak_device_memory_ratio_at_8192_plus=1.25,
                hidden_dense_cache_detected=False,
            ),
            baseline_comparison_report=BaselineComparisonReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                cartesian_int8_baseline_implemented=True,
                turbo_polar_wins_on_quality=True,
                turbo_polar_wins_on_memory=True,
                turbo_polar_wins_on_speed=True,
            ),
            provenance=BenchmarkProvenance(
                git_tree_state=GitTreeState.CLEAN,
                model_repo_id="test/model",
                model_revision="abc",
                turbopolar_config_hash="def",
                evidence_kind="experimental",
            ),
        )

        gate = PromotionGate()
        decision = gate.evaluate(evidence)
        # Should fail due to null trace entry
        assert decision.state.value in ["FAILED", "INCOMPLETE"]
        assert any("trace artifact" in r.lower() for r in decision.reasons)

    def test_fallback_trace_fails(self):
        """Fallback trace should cause gate to fail."""
        from rfsn_v11.promotion import PromotionGate, PromotionEvidence
        from rfsn_v11.promotion.schema import (
            KernelReport,
            TeacherForcedReport,
            FusedDecodeReport,
            SpeedReport,
            MemoryReport,
            BaselineComparisonReport,
            BenchmarkProvenance,
            GitTreeState,
        )

        # Calculate correct hash for fallback trace file
        trace_path = "tests/fixtures/evidence/fallback_trace.json"
        with open(trace_path, "rb") as f:
            content = f.read()
        correct_hash = hashlib.sha256(content).hexdigest()

        # Common context dictionaries
        positions = {512: 128, 2048: 128, 4096: 128, 8192: 128, 16384: 128}
        failed = {512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0}
        compressed = {512: 64, 2048: 256, 4096: 512, 8192: 1024, 16384: 2048}
        dense = {512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0}
        fallback = {512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0}

        evidence = PromotionEvidence(
            kernel_report=KernelReport(
                all_unit_tests_passed=True,
                all_kernel_tests_passed=True,
                all_integration_tests_passed=True,
                cpu_metal_agreement_verified=True,
                required_metal_tests=[],
                metal_tests_present=[],
                metal_tests_passed=[],
            ),
            teacher_forced_report=TeacherForcedReport(
                model="test",
                evaluated_contexts=[512, 2048, 4096, 8192, 16384],
                total_positions=640,
                mean_logit_cosine=0.996,
                p05_logit_cosine=0.991,
                min_logit_cosine=0.976,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
            ),
            fused_decode_report=FusedDecodeReport(
                model="test",
                model_layer_count=32,
                requested_fused_positions_per_context=128,
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                positions_per_context=positions,
                failed_positions_per_context=failed,
                compressed_page_dispatches_per_context=compressed,
                dense_tail_dispatches_per_context=dense,
                fallback_calls_per_context=fallback,
                trace_artifact_path=trace_path,
                trace_artifact_hash=correct_hash,
                mean_logit_cosine=0.996,
                p05_logit_cosine=0.991,
                min_logit_cosine=0.976,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
                execution_mode="metal_strict",
            ),
            speed_report=SpeedReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                trials_per_context=5,
                min_ratio_at_4096_plus=0.98,
                max_ratio_at_4096_plus=1.06,
                median_ratio_at_8192_plus=1.04,
                execution_mode="metal_strict",
                fallback_calls=0,
                raw_timing_path="",
                raw_timing_hash="test_hash",
            ),
            memory_report=MemoryReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                logical_kv_ratio=1.90,
                persistent_storage_ratio=1.80,
                peak_device_memory_ratio_at_8192_plus=1.25,
                hidden_dense_cache_detected=False,
            ),
            baseline_comparison_report=BaselineComparisonReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                cartesian_int8_baseline_implemented=True,
                turbo_polar_wins_on_quality=True,
                turbo_polar_wins_on_memory=True,
                turbo_polar_wins_on_speed=True,
            ),
            provenance=BenchmarkProvenance(
                git_tree_state=GitTreeState.CLEAN,
                model_repo_id="test/model",
                model_revision="abc",
                turbopolar_config_hash="def",
                evidence_kind="experimental",
            ),
        )

        gate = PromotionGate()
        decision = gate.evaluate(evidence)
        # Should fail due to fallback in trace
        assert decision.state.value in ["FAILED", "INCOMPLETE"]
        assert any("fallback" in r.lower() for r in decision.reasons)
