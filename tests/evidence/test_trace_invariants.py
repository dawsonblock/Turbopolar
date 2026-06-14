"""Tests for trace invariant validation."""

import unittest

from rfsn_v11.evidence.trace_validation import (
    ParsedOperationTrace,
    ParsedAttentionTrace,
    validate_trace_invariants,
    validate_single_trace_invariants,
)


class TestTraceInvariants(unittest.TestCase):
    """Test trace invariant validation logic."""

    def test_validate_trace_invariants_empty(self):
        """Empty trace list should pass validation."""
        errors = validate_trace_invariants([])
        self.assertEqual(errors, [])

    def test_validate_trace_invariants_consistent_experiment_id(self):
        """Traces with consistent experiment ID should pass."""
        traces = [
            ParsedOperationTrace(
                experiment_id="exp123",
                context_length=512,
                fixture_id="short_16",
                layer_index=0,
                decode_step=0,
                operation="attention",
                page_index=0,
                kernel_name="metal_attention",
                execution_mode="metal_strict",
                metal_requested=True,
                metal_executed=True,
                fallback_used=False,
                fallback_reason=None,
                output_evaluated=True,
                expected_tokens=128,
                processed_tokens=128,
                decode_ordinal=0,
            ),
            ParsedOperationTrace(
                experiment_id="exp123",
                context_length=512,
                fixture_id="short_16",
                layer_index=0,
                decode_step=1,
                operation="attention",
                page_index=1,
                kernel_name="metal_attention",
                execution_mode="metal_strict",
                metal_requested=True,
                metal_executed=True,
                fallback_used=False,
                fallback_reason=None,
                output_evaluated=True,
                expected_tokens=128,
                processed_tokens=128,
                decode_ordinal=1,
            ),
        ]
        errors = validate_trace_invariants(traces)
        self.assertEqual(errors, [])

    def test_validate_trace_invariants_inconsistent_experiment_id(self):
        """Traces with inconsistent experiment ID should fail."""
        traces = [
            ParsedOperationTrace(
                experiment_id="exp123",
                context_length=512,
                fixture_id="short_16",
                layer_index=0,
                decode_step=0,
                operation="attention",
                page_index=0,
                kernel_name="metal_attention",
                execution_mode="metal_strict",
                metal_requested=True,
                metal_executed=True,
                fallback_used=False,
                fallback_reason=None,
                output_evaluated=True,
                expected_tokens=128,
                processed_tokens=128,
                decode_ordinal=0,
            ),
            ParsedOperationTrace(
                experiment_id="exp456",
                context_length=512,
                fixture_id="short_16",
                layer_index=0,
                decode_step=1,
                operation="attention",
                page_index=1,
                kernel_name="metal_attention",
                execution_mode="metal_strict",
                metal_requested=True,
                metal_executed=True,
                fallback_used=False,
                fallback_reason=None,
                output_evaluated=True,
                expected_tokens=128,
                processed_tokens=128,
                decode_ordinal=1,
            ),
        ]
        errors = validate_trace_invariants(traces)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("Inconsistent experiment IDs" in errors[0])

    def test_validate_trace_invariants_monotonic_ordinals(self):
        """Traces with monotonic decode ordinals should pass."""
        traces = [
            ParsedOperationTrace(
                experiment_id="exp123",
                context_length=512,
                fixture_id="short_16",
                layer_index=0,
                decode_step=0,
                operation="attention",
                page_index=0,
                kernel_name="metal_attention",
                execution_mode="metal_strict",
                metal_requested=True,
                metal_executed=True,
                fallback_used=False,
                fallback_reason=None,
                output_evaluated=True,
                expected_tokens=128,
                processed_tokens=128,
                decode_ordinal=0,
            ),
            ParsedOperationTrace(
                experiment_id="exp123",
                context_length=512,
                fixture_id="short_16",
                layer_index=0,
                decode_step=1,
                operation="attention",
                page_index=1,
                kernel_name="metal_attention",
                execution_mode="metal_strict",
                metal_requested=True,
                metal_executed=True,
                fallback_used=False,
                fallback_reason=None,
                output_evaluated=True,
                expected_tokens=128,
                processed_tokens=128,
                decode_ordinal=1,
            ),
        ]
        errors = validate_trace_invariants(traces)
        self.assertEqual(errors, [])

    def test_validate_trace_invariants_non_monotonic_ordinals(self):
        """Traces with non-monotonic decode ordinals should fail."""
        traces = [
            ParsedOperationTrace(
                experiment_id="exp123",
                context_length=512,
                fixture_id="short_16",
                layer_index=0,
                decode_step=0,
                operation="attention",
                page_index=0,
                kernel_name="metal_attention",
                execution_mode="metal_strict",
                metal_requested=True,
                metal_executed=True,
                fallback_used=False,
                fallback_reason=None,
                output_evaluated=True,
                expected_tokens=128,
                processed_tokens=128,
                decode_ordinal=1,
            ),
            ParsedOperationTrace(
                experiment_id="exp123",
                context_length=512,
                fixture_id="short_16",
                layer_index=0,
                decode_step=1,
                operation="attention",
                page_index=1,
                kernel_name="metal_attention",
                execution_mode="metal_strict",
                metal_requested=True,
                metal_executed=True,
                fallback_used=False,
                fallback_reason=None,
                output_evaluated=True,
                expected_tokens=128,
                processed_tokens=128,
                decode_ordinal=0,
            ),
        ]
        errors = validate_trace_invariants(traces)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("monotonically" in errors[0])

    def test_validate_single_trace_positive_context_length(self):
        """Trace with positive context length should pass."""
        trace = ParsedOperationTrace(
            experiment_id="exp123",
            context_length=512,
            fixture_id="short_16",
            layer_index=0,
            decode_step=0,
            operation="attention",
            page_index=0,
            kernel_name="metal_attention",
            execution_mode="metal_strict",
            metal_requested=True,
            metal_executed=True,
            fallback_used=False,
            fallback_reason=None,
            output_evaluated=True,
            expected_tokens=128,
            processed_tokens=128,
        )
        errors = validate_single_trace_invariants(trace)
        self.assertEqual(errors, [])

    def test_validate_single_trace_negative_context_length(self):
        """Trace with negative context length should fail."""
        trace = ParsedOperationTrace(
            experiment_id="exp123",
            context_length=-1,
            fixture_id="short_16",
            layer_index=0,
            decode_step=0,
            operation="attention",
            page_index=0,
            kernel_name="metal_attention",
            execution_mode="metal_strict",
            metal_requested=True,
            metal_executed=True,
            fallback_used=False,
            fallback_reason=None,
            output_evaluated=True,
            expected_tokens=128,
            processed_tokens=128,
        )
        errors = validate_single_trace_invariants(trace)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("Context length" in errors[0])

    def test_validate_single_trace_metal_without_fallback(self):
        """Metal requested but not executed without fallback should fail."""
        trace = ParsedOperationTrace(
            experiment_id="exp123",
            context_length=512,
            fixture_id="short_16",
            layer_index=0,
            decode_step=0,
            operation="attention",
            page_index=0,
            kernel_name="metal_attention",
            execution_mode="metal_strict",
            metal_requested=True,
            metal_executed=False,
            fallback_used=False,
            fallback_reason=None,
            output_evaluated=True,
            expected_tokens=128,
            processed_tokens=128,
        )
        errors = validate_single_trace_invariants(trace)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("Metal requested but not executed" in errors[0])

    def test_validate_single_trace_fallback_without_reason(self):
        """Fallback used without reason should fail."""
        trace = ParsedOperationTrace(
            experiment_id="exp123",
            context_length=512,
            fixture_id="short_16",
            layer_index=0,
            decode_step=0,
            operation="attention",
            page_index=0,
            kernel_name="metal_attention",
            execution_mode="metal_strict",
            metal_requested=True,
            metal_executed=False,
            fallback_used=True,
            fallback_reason=None,
            output_evaluated=True,
            expected_tokens=128,
            processed_tokens=128,
        )
        errors = validate_single_trace_invariants(trace)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("no reason provided" in errors[0])

    def test_validate_attention_trace_cache_tokens_decrease(self):
        """Attention trace with decreasing cache tokens should fail."""
        trace = ParsedAttentionTrace(
            experiment_id="exp123",
            context_length=512,
            fixture_id="short_16",
            layer_index=0,
            decode_step=0,
            expected_page_count=4,
            page_operations=(),
            dense_tail_operation=None,
            cache_tokens_before=100,
            cache_tokens_after=50,
        )
        errors = validate_single_trace_invariants(trace)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("decreased" in errors[0])

    def test_validate_attention_trace_page_count_mismatch(self):
        """Attention trace with page count mismatch should fail."""
        trace = ParsedAttentionTrace(
            experiment_id="exp123",
            context_length=512,
            fixture_id="short_16",
            layer_index=0,
            decode_step=0,
            expected_page_count=4,
            page_operations=(),  # Empty, but expected 4
            dense_tail_operation=None,
        )
        errors = validate_single_trace_invariants(trace)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("Page count mismatch" in errors[0])


if __name__ == "__main__":
    unittest.main()