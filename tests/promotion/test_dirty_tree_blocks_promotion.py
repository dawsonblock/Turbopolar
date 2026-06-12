"""Tests that a dirty git tree blocks promotion."""

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


class TestDirtyTreeBlocksPromotion(unittest.TestCase):
    """A dirty source tree must result in FAILED, not PROMOTED_EXPERIMENTAL."""

    def _full_passing_evidence(self, dirty: bool) -> PromotionEvidence:
        return PromotionEvidence(
            kernel_report=KernelReport(
                all_unit_tests_passed=True,
                all_kernel_tests_passed=True,
                all_integration_tests_passed=True,
                cpu_metal_agreement_verified=True,
                metal_tests_present=list(PromotionGate.REQUIRED_NATIVE_METAL_TESTS),
                metal_tests_passed=list(PromotionGate.REQUIRED_NATIVE_METAL_TESTS),
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
                contexts_evaluated=list(PromotionGate.REQUIRED_CONTEXTS),
            ),
            speed_report=SpeedReport(
                min_ratio_at_4096_plus=0.98,
                max_ratio_at_4096_plus=1.06,
                median_ratio_at_8192_plus=1.04,
                trials_per_context=5,
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
                git_tree_state=GitTreeState.DIRTY if dirty else GitTreeState.CLEAN,
                git_diff_hash="abcd1234" if dirty else "",
                model_repo_id="test/model",
                model_revision="abc123",
                turbopolar_config_hash="def456",
            ),
        )

    def test_clean_tree_can_promote(self):
        evidence = self._full_passing_evidence(dirty=False)
        decision = PromotionGate().evaluate(evidence)
        # Promotion is locked at REVIEW_REQUIRED until independent validation.
        self.assertEqual(decision.state, PromotionState.REVIEW_REQUIRED)

    def test_dirty_tree_blocks(self):
        evidence = self._full_passing_evidence(dirty=True)
        decision = PromotionGate().evaluate(evidence)
        self.assertEqual(decision.state, PromotionState.FAILED)
        self.assertTrue(any("dirty" in r.lower() for r in decision.reasons))


if __name__ == "__main__":
    unittest.main()
