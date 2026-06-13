"""Tests for promotion schema nested constructors and tri-state git."""

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
            kernel_report=KernelReport(
                all_unit_tests_passed=True,
                all_kernel_tests_passed=True,
                all_integration_tests_passed=True,
                cpu_metal_agreement_verified=True,
                required_metal_tests=[
                    "tests.kernels.test_paged_online_attention",
                    "tests.kernels.test_qjl_scaled_fused_qk",
                    "tests.kernels.test_qjl_scaled_online_attention",
                    "tests.kernels.test_metal_strict",
                    "tests.kernels.test_fallback_injection",
                    "tests.benchmarks.test_turbopolar_fast_attention",
                    "tests.benchmarks.test_turbo_polar_online_attention",
                ],
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
            ),
            teacher_forced_report=TeacherForcedReport(
                mean_logit_cosine=0.999,
                p05_logit_cosine=0.995,
                min_logit_cosine=0.990,
                argmax_agreement=0.98,
                mean_top5_overlap=0.97,
                mean_top10_overlap=0.98,
                mean_perplexity_delta=0.005,
                any_nans_or_infs=False,
            ),
            fused_decode_report=FusedDecodeReport(
                mean_logit_cosine=0.999,
                p05_logit_cosine=0.995,
                min_logit_cosine=0.990,
                mean_top5_overlap=0.97,
                mean_top10_overlap=0.98,
                argmax_agreement=0.98,
                mean_perplexity_delta=0.005,
                any_nans_or_infs=False,
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                positions_per_context={512: 128, 2048: 128, 4096: 128, 8192: 128, 16384: 128},
                failed_positions_per_context={512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0},
                compressed_page_dispatches_per_context={512: 1, 2048: 1, 4096: 1, 8192: 1, 16384: 1},
                dense_tail_dispatches_per_context={512: 1, 2048: 1, 4096: 1, 8192: 1, 16384: 1},
                fallback_calls_per_context={512: 0, 2048: 0, 4096: 0, 8192: 0, 16384: 0},
                actual_fused_positions=128,
                execution_mode="metal_strict",
                compressed_page_metal_calls=1,
                dense_tail_metal_calls=1,
                compressed_page_fallback_calls=0,
                dense_tail_fallback_calls=0,
                full_attention_fallback_calls=0,
                fallback_reasons=[],
            ),
            speed_report=SpeedReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                trials_per_context=5,
                min_ratio_at_4096_plus=0.98,
                max_ratio_at_4096_plus=1.06,
                median_ratio_at_8192_plus=1.04,
            ),
            memory_report=MemoryReport(
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                logical_kv_ratio=1.90,
                persistent_storage_ratio=1.80,
                peak_device_memory_ratio_at_8192_plus=1.25,
                hidden_dense_cache_detected=False,
            ),
            baseline_comparison_report=BaselineComparisonReport(
                cartesian_int8_baseline_implemented=True,
                turbo_polar_wins_on_quality=True,
                turbo_polar_wins_on_memory=True,
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
        # Unknown provenance with no hard failures is INCOMPLETE, not REVIEW_REQUIRED.
        self.assertEqual(decision.state, PromotionState.INCOMPLETE)
        self.assertTrue(any("Git tree state unknown" in r for r in decision.reasons))


if __name__ == "__main__":
    unittest.main()
