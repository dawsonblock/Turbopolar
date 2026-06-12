"""Tests that long-context speed gates block promotion when failed."""

import unittest

from rfsn_v11.promotion import (
    BaselineComparisonReport,
    BenchmarkProvenance,
    FusedDecodeReport,
    GitTreeState,
    KernelReport,
    MemoryReport,
    PromotionEvidence,
    PromotionGate,
    PromotionState,
    SpeedReport,
    TeacherForcedReport,
)


class TestLongContextGates(unittest.TestCase):
    """One failed long-context tier must block promotion."""

    def _evidence_with_speed(self, **speed_overrides) -> PromotionEvidence:
        speed_fields = {
            "contexts_evaluated": [512, 2048, 4096, 8192],
            "min_ratio_at_4096_plus": 0.98,
            "max_ratio_at_4096_plus": 1.06,
            "median_ratio_at_8192_plus": 1.04,
        }
        speed_fields.update(speed_overrides)
        speed = SpeedReport(**speed_fields)
        return PromotionEvidence(
            kernel_report=KernelReport(
                all_unit_tests_passed=True,
                all_kernel_tests_passed=True,
                all_integration_tests_passed=True,
            ),
            teacher_forced_report=TeacherForcedReport(
                mean_logit_cosine=0.999,
                p05_logit_cosine=0.995,
                min_logit_cosine=0.980,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
            ),
            fused_decode_report=FusedDecodeReport(
                mean_logit_cosine=0.999,
                p05_logit_cosine=0.995,
                min_logit_cosine=0.980,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
            ),
            speed_report=speed,
            memory_report=MemoryReport(
                logical_kv_ratio=1.90,
                persistent_storage_ratio=1.80,
                peak_device_memory_ratio_at_8192_plus=1.25,
                hidden_dense_cache_detected=False,
            ),
            baseline_comparison_report=BaselineComparisonReport(
                cartesian_int8_baseline_implemented=True,
                turbo_polar_wins_on_speed=True,
            ),
            provenance=BenchmarkProvenance(
                git_tree_state=GitTreeState.CLEAN,
                model_repo_id="test/model",
                model_revision="abc123",
                turbopolar_config_hash="def456",
            ),
        )

    def test_3pct_regression_at_4096_blocks(self):
        evidence = self._evidence_with_speed(min_ratio_at_4096_plus=0.95)
        decision = PromotionGate().evaluate(evidence)
        self.assertEqual(decision.state, PromotionState.FAILED)
        self.assertTrue(any("4096+ minimum" in r for r in decision.reasons))

    def test_no_improvement_blocks(self):
        evidence = self._evidence_with_speed(max_ratio_at_4096_plus=1.01)
        decision = PromotionGate().evaluate(evidence)
        self.assertEqual(decision.state, PromotionState.FAILED)
        self.assertTrue(
            any("No long-context tier improved" in r for r in decision.reasons)
        )

    def test_low_median_at_8192_blocks(self):
        evidence = self._evidence_with_speed(median_ratio_at_8192_plus=1.01)
        decision = PromotionGate().evaluate(evidence)
        self.assertEqual(decision.state, PromotionState.FAILED)
        self.assertTrue(any("8192+" in r for r in decision.reasons))


if __name__ == "__main__":
    unittest.main()
