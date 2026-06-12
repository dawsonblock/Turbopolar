"""Tests that missing promotion evidence results in failure, not success."""

import unittest

from rfsn_v11.promotion import PromotionEvidence, PromotionGate, PromotionState


class TestMissingEvidenceFails(unittest.TestCase):
    """PromotionGate.evaluate must return INCOMPLETE or FAILED for missing reports."""

    def setUp(self):
        self.gate = PromotionGate()

    def test_none_evidence_is_incomplete(self):
        decision = self.gate.evaluate(None)
        self.assertEqual(decision.state, PromotionState.INCOMPLETE)
        self.assertIn("No promotion evidence provided", decision.reasons[0])

    def test_empty_evidence_is_incomplete(self):
        decision = self.gate.evaluate(PromotionEvidence())
        self.assertEqual(decision.state, PromotionState.INCOMPLETE)
        self.assertEqual(len(decision.reasons), 7)
        for reason in decision.reasons:
            self.assertIn("Missing required report", reason)

    def test_partial_evidence_is_incomplete(self):
        from rfsn_v11.promotion import KernelReport

        evidence = PromotionEvidence(kernel_report=KernelReport())
        decision = self.gate.evaluate(evidence)
        self.assertEqual(decision.state, PromotionState.INCOMPLETE)

    def test_speed_default_true_pattern_fails(self):
        """The old wrong pattern (speed=True if missing) must not occur here."""
        from rfsn_v11.promotion import (
            BaselineComparisonReport,
            BenchmarkProvenance,
            FusedDecodeReport,
            GitTreeState,
            KernelReport,
            MemoryReport,
            SpeedReport,
            TeacherForcedReport,
        )

        evidence = PromotionEvidence(
            kernel_report=KernelReport(all_unit_tests_passed=True),
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
            speed_report=SpeedReport(
                min_ratio_at_4096_plus=1.0,
                max_ratio_at_4096_plus=1.06,
                median_ratio_at_8192_plus=1.04,
            ),
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
        decision = self.gate.evaluate(evidence)
        # Kernel report is incomplete (kernel tests not passed), so it should fail.
        self.assertEqual(decision.state, PromotionState.FAILED)
        self.assertTrue(len(decision.reasons) > 0)
        self.assertTrue(any("Kernel tests" in r for r in decision.reasons))


if __name__ == "__main__":
    unittest.main()
