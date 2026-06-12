"""Tests for promotion schema nested constructors and tri-state git."""

import unittest

from rfsn_v11.promotion import (
    BaselineComparisonReport,
    BenchmarkProvenance,
    FusedDecodeReport,
    GitTreeState,
    KernelReport,
    MemoryReport,
    PromotionDecision,
    PromotionEvidence,
    PromotionGate,
    PromotionState,
    SpeedReport,
    TeacherForcedReport,
)


class TestPromotionSchema(unittest.TestCase):
    def test_kernel_report_from_dict(self):
        kr = KernelReport.from_dict({"all_unit_tests_passed": True, "notes": ["ok"]})
        self.assertTrue(kr.all_unit_tests_passed)
        self.assertEqual(kr.notes, ["ok"])

    def test_teacher_forced_report_from_dict(self):
        data = {
            "model": "m",
            "mean_logit_cosine": 0.99,
            "mean_top5_overlap": 0.95,
            "any_nans_or_infs": False,
        }
        tf = TeacherForcedReport.from_dict(data)
        self.assertEqual(tf.model, "m")
        self.assertEqual(tf.mean_logit_cosine, 0.99)
        self.assertIsNone(tf.p05_logit_cosine)
        self.assertFalse(tf.any_nans_or_infs)

    def test_promotion_evidence_roundtrip(self):
        evidence = PromotionEvidence(
            kernel_report=KernelReport(all_unit_tests_passed=True),
            provenance=BenchmarkProvenance(
                git_tree_state=GitTreeState.CLEAN,
                model_repo_id="test/model",
                model_revision="abc",
                turbopolar_config_hash="def",
            ),
        )
        from dataclasses import asdict

        d = asdict(evidence)
        # Enums serialize to their value string via asdict.
        restored = PromotionEvidence.from_dict(d)
        self.assertIsInstance(restored.kernel_report, KernelReport)
        self.assertTrue(restored.kernel_report.all_unit_tests_passed)
        self.assertEqual(restored.provenance.git_tree_state, GitTreeState.CLEAN)

    def test_git_tree_state_unknown_is_review_required(self):
        evidence = PromotionEvidence(
            kernel_report=KernelReport(all_unit_tests_passed=True),
            teacher_forced_report=TeacherForcedReport(mean_logit_cosine=0.999),
            fused_decode_report=FusedDecodeReport(mean_logit_cosine=0.999),
            speed_report=SpeedReport(
                min_ratio_at_4096_plus=0.98,
                max_ratio_at_4096_plus=1.06,
                median_ratio_at_8192_plus=1.04,
            ),
            memory_report=MemoryReport(logical_kv_ratio=1.90),
            baseline_comparison_report=BaselineComparisonReport(
                cartesian_int8_baseline_implemented=True,
                turbo_polar_wins_on_speed=True,
            ),
            provenance=BenchmarkProvenance(
                git_tree_state=GitTreeState.UNKNOWN,
                model_repo_id="test/model",
                model_revision="abc",
                turbopolar_config_hash="def",
            ),
        )
        decision = PromotionGate().evaluate(evidence)
        self.assertEqual(decision.state, PromotionState.REVIEW_REQUIRED)
        self.assertTrue(any("Git tree state unknown" in r for r in decision.reasons))


if __name__ == "__main__":
    unittest.main()
