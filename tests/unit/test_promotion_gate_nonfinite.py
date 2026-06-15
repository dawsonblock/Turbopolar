"""Regression tests for summary-level non-finite and boolean rejection.

Revision 9 fixed raw JSON NaN/Infinity rejection, but dataclass summary
fields still bypassed threshold checks because ``NaN < x`` is ``False``.
These tests verify that every numeric summary field is finite-checked
before any threshold or reconciliation comparison.
"""

import unittest

from rfsn_v11.promotion.schema import (
    BaselineComparisonReport,
    BenchmarkProvenance,
    FusedDecodeReport,
    GitTreeState,
    KernelReport,
    MemoryReport,
    PromotionEvidence,
    PromotionState,
    SpeedReport,
    TeacherForcedReport,
)
from rfsn_v11.promotion.gate import PromotionGate


class TestPromotionGateNonFiniteSummary(unittest.TestCase):
    """Verify that non-finite or boolean summary fields fail the gate."""

    def _valid_evidence(self) -> PromotionEvidence:
        """Return structurally valid evidence that would pass without NaN."""
        required_contexts = [512, 2048, 4096, 8192, 16384]
        per_context = {
            512: 128, 2048: 128, 4096: 128, 8192: 128, 16384: 128
        }
        # Unique workload hashes to avoid duplicate-hash failure
        hashes = [
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "e" * 64,
        ]
        return PromotionEvidence(
            kernel_report=KernelReport(
                all_unit_tests_passed=True,
                all_kernel_tests_passed=True,
                all_integration_tests_passed=True,
                cpu_metal_agreement_verified=True,
                required_metal_tests=[],
                metal_tests_present=[
                    "tests.kernels.test_paged_online_attention",
                    "tests.kernels.test_qjl_scaled_fused_qk",
                    "tests.kernels.test_qjl_scaled_online_attention",
                    "tests.kernels.test_metal_strict",
                    "tests.kernels.test_fallback_injection",
                    "tests.benchmarks.test_turbopolar_fast_attention",
                    "tests.benchmarks.test_turbo_polar_online_attention",
                ],
                metal_tests_passed=[
                    "tests.kernels.test_paged_online_attention",
                    "tests.kernels.test_qjl_scaled_fused_qk",
                    "tests.kernels.test_qjl_scaled_online_attention",
                    "tests.kernels.test_metal_strict",
                    "tests.kernels.test_fallback_injection",
                    "tests.benchmarks.test_turbopolar_fast_attention",
                    "tests.benchmarks.test_turbo_polar_online_attention",
                ],
                metal_tests_skipped=[],
            ),
            teacher_forced_report=TeacherForcedReport(
                model="test",
                evaluated_contexts=required_contexts,
                total_positions=640,
                mean_logit_cosine=0.999,
                p05_logit_cosine=0.995,
                min_logit_cosine=0.990,
                argmax_agreement=0.98,
                mean_top5_overlap=0.97,
                mean_top10_overlap=0.98,
                mean_perplexity_delta=0.005,
                any_nans_or_infs=False,
                raw_metrics_path="/tmp/dummy_teacher.json",
                raw_metrics_hash="f" * 64,
            ),
            fused_decode_report=FusedDecodeReport(
                model="test",
                model_layer_count=32,
                requested_fused_positions_per_context=128,
                contexts_evaluated=required_contexts,
                positions_per_context=per_context,
                failed_positions_per_context={
                    512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0
                },
                compressed_page_dispatches_per_context={
                    512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0
                },
                dense_tail_dispatches_per_context={
                    512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0
                },
                fallback_calls_per_context={
                    512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0
                },
                trace_artifact_path="/tmp/dummy_trace.json",
                trace_artifact_hash="g" * 64,
                mean_logit_cosine=0.999,
                p05_logit_cosine=0.995,
                min_logit_cosine=0.990,
                mean_top5_overlap=0.97,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.005,
                any_nans_or_infs=False,
                execution_mode="metal_strict",
                compressed_page_metal_calls=0,
                dense_tail_metal_calls=0,
                compressed_page_fallback_calls=0,
                dense_tail_fallback_calls=0,
                full_attention_fallback_calls=0,
                fallback_reasons=[],
                fallback_calls=0,
                first_argmax_divergence_step=None,
                actual_fused_positions=128,
            ),
            speed_report=SpeedReport(
                model="test",
                contexts_evaluated=required_contexts,
                trials_per_context=5,
                median_ratio=1.1,
                min_ratio_at_4096_plus=0.98,
                max_ratio_at_4096_plus=1.10,
                median_ratio_at_8192_plus=1.05,
                execution_mode="metal_strict",
                fallback_calls=0,
                raw_timing_path="/tmp/dummy_speed.json",
                raw_timing_hash="h" * 64,
            ),
            memory_report=MemoryReport(
                model="test",
                contexts_evaluated=required_contexts,
                logical_kv_ratio=1.90,
                persistent_storage_ratio=1.80,
                peak_device_memory_ratio_at_8192_plus=1.25,
                hidden_dense_cache_detected=False,
                raw_memory_path="/tmp/dummy_memory.json",
                raw_memory_hash="i" * 64,
            ),
            baseline_comparison_report=BaselineComparisonReport(
                model="test",
                contexts_evaluated=required_contexts,
                cartesian_int8_baseline_implemented=True,
                turbo_polar_wins_on_quality=True,
                turbo_polar_wins_on_memory=True,
                turbo_polar_wins_on_speed=True,
            ),
            provenance=BenchmarkProvenance(
                run_id="run-001",
                timestamp_utc="2024-01-01T00:00:00Z",
                git_commit="a" * 40,
                git_tree_state=GitTreeState.CLEAN,
                git_diff_hash="",
                python_version="3.12.0",
                mlx_version="0.15.0",
                mlx_lm_version="0.10.0",
                macos_version="14.0",
                chip_model="Apple M1",
                system_memory_gb=16.0,
                model_repo_id="test/model",
                model_revision="abc123def456",
                tokenizer_revision="ghi789jkl012",
                token_fixtures_hash="mno345pqr678" * 5 + "mno3",
                turbopolar_config_hash="a" * 64,
                benchmark_command="python run.py",
                warmup_count=1,
                trial_count=5,
                context_lengths=required_contexts,
                decode_token_count=128,
                qjl_enabled=False,
                metal_kernel_source_hash="b" * 64,
                kernel_binding_hash="c" * 64,
                execution_mode="metal_strict",
                evidence_kind="experimental",
                speed_workload_hash=hashes[0],
                memory_workload_hash=hashes[1],
                fused_decode_workload_hash=hashes[2],
                cartesian_workload_hash=hashes[3],
                teacher_forced_workload_hash=hashes[4],
            ),
        )

    def _assert_fails(
        self, evidence: PromotionEvidence, field_name: str
    ) -> None:
        decision = PromotionGate().evaluate(evidence)
        self.assertEqual(
            decision.state,
            PromotionState.FAILED,
            f"Expected FAILED when {field_name} is invalid, "
            f"got {decision.state}",
        )
        reasons = "\n".join(decision.reasons)
        self.assertTrue(
            (
                "non-finite" in reasons
                or "boolean" in reasons
                or "missing" in reasons
            ),
            f"Expected non-finite/boolean/missing reason for "
            f"{field_name}; got:\n{reasons}",
        )

    # Teacher forced ------------------------------------------------------

    def test_teacher_mean_cosine_nan_fails(self):
        ev = self._valid_evidence()
        ev.teacher_forced_report.mean_logit_cosine = float("nan")
        self._assert_fails(ev, "teacher mean_logit_cosine=NaN")

    def test_teacher_mean_cosine_inf_fails(self):
        ev = self._valid_evidence()
        ev.teacher_forced_report.mean_logit_cosine = float("inf")
        self._assert_fails(ev, "teacher mean_logit_cosine=inf")

    def test_teacher_mean_cosine_bool_fails(self):
        ev = self._valid_evidence()
        ev.teacher_forced_report.mean_logit_cosine = True
        self._assert_fails(ev, "teacher mean_logit_cosine=True")

    def test_teacher_perplexity_delta_nan_fails(self):
        ev = self._valid_evidence()
        ev.teacher_forced_report.mean_perplexity_delta = float("nan")
        self._assert_fails(ev, "teacher mean_perplexity_delta=NaN")

    def test_teacher_perplexity_delta_inf_fails(self):
        ev = self._valid_evidence()
        ev.teacher_forced_report.mean_perplexity_delta = float("inf")
        self._assert_fails(ev, "teacher mean_perplexity_delta=inf")

    # Fused decode --------------------------------------------------------

    def test_fused_mean_cosine_nan_fails(self):
        ev = self._valid_evidence()
        ev.fused_decode_report.mean_logit_cosine = float("nan")
        self._assert_fails(ev, "fused mean_logit_cosine=NaN")

    def test_fused_mean_cosine_inf_fails(self):
        ev = self._valid_evidence()
        ev.fused_decode_report.mean_logit_cosine = float("inf")
        self._assert_fails(ev, "fused mean_logit_cosine=inf")

    def test_fused_mean_cosine_bool_fails(self):
        ev = self._valid_evidence()
        ev.fused_decode_report.mean_logit_cosine = True
        self._assert_fails(ev, "fused mean_logit_cosine=True")

    # Speed ---------------------------------------------------------------

    def test_speed_min_ratio_nan_fails(self):
        ev = self._valid_evidence()
        ev.speed_report.min_ratio_at_4096_plus = float("nan")
        self._assert_fails(ev, "speed min_ratio_at_4096_plus=NaN")

    def test_speed_min_ratio_inf_fails(self):
        ev = self._valid_evidence()
        ev.speed_report.min_ratio_at_4096_plus = float("inf")
        self._assert_fails(ev, "speed min_ratio_at_4096_plus=inf")

    def test_speed_min_ratio_bool_fails(self):
        ev = self._valid_evidence()
        ev.speed_report.min_ratio_at_4096_plus = True
        self._assert_fails(ev, "speed min_ratio_at_4096_plus=True")

    def test_speed_max_ratio_nan_fails(self):
        ev = self._valid_evidence()
        ev.speed_report.max_ratio_at_4096_plus = float("nan")
        self._assert_fails(ev, "speed max_ratio_at_4096_plus=NaN")

    def test_speed_median_ratio_8192_nan_fails(self):
        ev = self._valid_evidence()
        ev.speed_report.median_ratio_at_8192_plus = float("nan")
        self._assert_fails(ev, "speed median_ratio_at_8192_plus=NaN")

    # Memory --------------------------------------------------------------

    def test_memory_logical_kv_ratio_nan_fails(self):
        ev = self._valid_evidence()
        ev.memory_report.logical_kv_ratio = float("nan")
        self._assert_fails(ev, "memory logical_kv_ratio=NaN")

    def test_memory_logical_kv_ratio_inf_fails(self):
        ev = self._valid_evidence()
        ev.memory_report.logical_kv_ratio = float("inf")
        self._assert_fails(ev, "memory logical_kv_ratio=inf")

    def test_memory_logical_kv_ratio_bool_fails(self):
        ev = self._valid_evidence()
        ev.memory_report.logical_kv_ratio = True
        self._assert_fails(ev, "memory logical_kv_ratio=True")

    def test_memory_persistent_ratio_nan_fails(self):
        ev = self._valid_evidence()
        ev.memory_report.persistent_storage_ratio = float("nan")
        self._assert_fails(ev, "memory persistent_storage_ratio=NaN")

    def test_memory_peak_ratio_nan_fails(self):
        ev = self._valid_evidence()
        ev.memory_report.peak_device_memory_ratio_at_8192_plus = float("nan")
        self._assert_fails(
            ev, "memory peak_device_memory_ratio_at_8192_plus=NaN"
        )

    # Schema from_dict boolean coercion -----------------------------------

    def test_schema_rejects_bool_for_teacher_mean_cosine(self):
        tf = TeacherForcedReport.from_dict({"mean_logit_cosine": True})
        self.assertIsNone(tf.mean_logit_cosine)

    def test_schema_rejects_bool_for_fused_mean_cosine(self):
        fd = FusedDecodeReport.from_dict({"mean_logit_cosine": True})
        self.assertIsNone(fd.mean_logit_cosine)

    def test_schema_rejects_bool_for_speed_min_ratio(self):
        sr = SpeedReport.from_dict({"min_ratio_at_4096_plus": True})
        self.assertIsNone(sr.min_ratio_at_4096_plus)

    def test_schema_rejects_bool_for_memory_logical_ratio(self):
        mr = MemoryReport.from_dict({"logical_kv_ratio": True})
        self.assertIsNone(mr.logical_kv_ratio)

    def test_schema_rejects_bool_for_actual_fused_positions(self):
        fd = FusedDecodeReport.from_dict({"actual_fused_positions": True})
        self.assertIsNone(fd.actual_fused_positions)

    def test_schema_rejects_bool_for_fallback_calls(self):
        sr = SpeedReport.from_dict({"fallback_calls": True})
        self.assertEqual(sr.fallback_calls, 0)

    # Git tree state UNKNOWN ordering fix ---------------------------------

    def test_git_unknown_not_added_as_hard_failure(self):
        """UNKNOWN git handling must come after the FAILED return, not before
        it (regression test for early-exit bug)."""
        import inspect

        source_lines, _ = inspect.getsourcelines(PromotionGate.evaluate)
        failed_line = None
        unknown_line = None
        for i, line in enumerate(source_lines):
            if "state=PromotionState.FAILED" in line:
                failed_line = i
                break
        for i, line in enumerate(source_lines):
            if "GitTreeState.UNKNOWN" in line:
                unknown_line = i
                break
        self.assertIsNotNone(
            failed_line, "FAILED return must exist in evaluate"
        )
        self.assertIsNotNone(
            unknown_line, "UNKNOWN check must exist in evaluate"
        )
        self.assertGreater(
            unknown_line,
            failed_line,
            "UNKNOWN check must come after the FAILED return, not before it",
        )


if __name__ == "__main__":
    unittest.main()
