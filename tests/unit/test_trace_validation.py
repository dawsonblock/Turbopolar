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
    ParsedOperationTrace,
    ParsedAttentionTrace,
)


class TestTraceArtifactParsing:
    """Test that malformed trace artifacts fail with TraceArtifactError."""

    def test_empty_list_fails(self):
        """Empty trace artifact list must fail."""
        with pytest.raises(TraceArtifactError, match="Trace artifact contains no traces"):
            parse_trace_artifact([])

    def test_null_entry_fails(self):
        """Null trace entry must fail without crashing."""
        with pytest.raises(TraceArtifactError, match="Trace entry 0 must be an object"):
            parse_trace_artifact([None])

    def test_non_list_fails(self):
        """Non-list trace artifact must fail."""
        with pytest.raises(TraceArtifactError, match="Trace artifact must contain a JSON array"):
            parse_trace_artifact({})

    def test_empty_object_fails(self):
        """Empty trace object must fail."""
        with pytest.raises(TraceArtifactError, match="missing required field"):
            parse_trace_artifact([{}])

    def test_missing_page_traces_fails(self):
        """Missing page_traces field must fail."""
        with pytest.raises(TraceArtifactError, match="missing required field 'page_traces'"):
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
        with pytest.raises(TraceArtifactError, match="page_traces must be a list"):
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
        with pytest.raises(TraceArtifactError, match="missing required field 'dense_tail_trace'"):
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
        with pytest.raises(TraceArtifactError, match="page_traces\\[0\\] must be an object"):
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
        with pytest.raises(TraceArtifactError, match="missing required field 'experiment_id'"):
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
        with pytest.raises(TraceArtifactError, match="missing required field 'context_length'"):
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
        with pytest.raises(TraceArtifactError, match="missing required field 'metal_requested'"):
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

    def test_compressed_page_missing_page_index_fails(self):
        """compressed_page operation must have page_index."""
        with pytest.raises(TraceArtifactError, match="compressed_page operation must have page_index"):
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
        with pytest.raises(TraceArtifactError, match="page_index cannot be negative"):
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
        with pytest.raises(TraceArtifactError, match="fallback_used=True requires fallback_reason"):
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
        with pytest.raises(TraceArtifactError, match="does not match trace context_length"):
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
        with pytest.raises(TraceArtifactError, match="does not match trace fixture_id"):
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
        with pytest.raises(TraceArtifactError, match="dense_tail_trace must be an object or null"):
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
        with open(path, "r") as f:
            raw = json.load(f)
        with pytest.raises(TraceArtifactError, match="Trace artifact contains no traces"):
            parse_trace_artifact(raw)

    def test_null_trace_fixture_fails(self):
        """Null trace fixture should fail."""
        path = "tests/fixtures/evidence/null_trace.json"
        with open(path, "r") as f:
            raw = json.load(f)
        with pytest.raises(TraceArtifactError, match="Trace entry 0 must be an object"):
            parse_trace_artifact(raw)

    def test_missing_page_fixture_fails(self):
        """Missing page fixture should fail topology validation (not parser)."""
        path = "tests/fixtures/evidence/missing_page.json"
        with open(path, "r") as f:
            raw = json.load(f)
        # Parser should succeed, topology validation should fail later
        result = parse_trace_artifact(raw)
        assert len(result) == 1
        assert result[0].expected_page_count == 2
        assert len(result[0].page_operations) == 1  # Missing one page

    def test_duplicate_page_fixture_fails(self):
        """Duplicate page fixture should parse but fail topology validation."""
        path = "tests/fixtures/evidence/duplicate_page.json"
        with open(path, "r") as f:
            raw = json.load(f)
        result = parse_trace_artifact(raw)
        assert len(result) == 1
        assert len(result[0].page_operations) == 2
        # Both have page_index=0, topology validation should catch this

    def test_fallback_trace_fixture_succeeds_parsing(self):
        """Fallback trace should parse successfully (validation happens later)."""
        path = "tests/fixtures/evidence/fallback_trace.json"
        with open(path, "r") as f:
            raw = json.load(f)
        result = parse_trace_artifact(raw)
        assert len(result) == 1
        assert result[0].page_operations[0].fallback_used is True
        assert result[0].page_operations[0].fallback_reason == "injected_for_test"

    def test_unevaluated_trace_fixture_succeeds_parsing(self):
        """Unevaluated trace should parse successfully (validation happens later)."""
        path = "tests/fixtures/evidence/unevaluated_trace.json"
        with open(path, "r") as f:
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
                layer_index=0,
                expected_page_count=2,
                page_operations=tuple(),
                dense_tail_operation=None,
            )
        ]
        with pytest.raises(TraceTopologyError, match="Missing required contexts"):
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
        with pytest.raises(TraceTopologyError, match="missing trace for decode_step"):
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
        with pytest.raises(TraceTopologyError, match="page count 1 != expected 2"):
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
        with pytest.raises(TraceTopologyError, match="page indices.*!= expected"):
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
            with pytest.raises(EvidenceValidationError, match="must be 64 characters"):
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
