"""Tests that a dirty git tree blocks promotion."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

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

    def setUp(self):
        """Create temporary files for artifact validation."""
        self.temp_dir = tempfile.mkdtemp()

        # Create teacher-forced metrics file
        self.teacher_forced_path = Path(self.temp_dir) / "teacher_forced.json"
        teacher_forced_data = {
            "prompts": [
                {
                    "position_metrics": [
                        {
                            "logit_cosine": 0.999,
                            "top5_overlap": 0.96,
                            "top10_overlap": 0.98,
                            "argmax_agreement": True,
                            "any_nan_or_inf": False,
                        }
                        for _ in range(128)
                    ]
                }
            ]
        }
        self.teacher_forced_path.write_text(json.dumps(teacher_forced_data))
        self.teacher_forced_hash = hashlib.sha256(
            json.dumps(teacher_forced_data).encode()
        ).hexdigest()

        # Create fused decode trace file
        self.fused_trace_path = Path(self.temp_dir) / "fused_decode_trace.json"
        trace_data = {"experiments": []}
        self.fused_trace_path.write_text(json.dumps(trace_data))
        self.fused_trace_hash = hashlib.sha256(
            json.dumps(trace_data).encode()
        ).hexdigest()

        # Create speed timing file
        self.speed_timing_path = Path(self.temp_dir) / "speed_timing.json"
        speed_data = {
            "schema_version": 1,
            "speed_evidence": {"trial_results": []},
        }
        self.speed_timing_path.write_text(json.dumps(speed_data))
        encoded_speed = json.dumps(speed_data).encode()
        self.speed_timing_hash = hashlib.sha256(
            encoded_speed
        ).hexdigest()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

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
                raw_metrics_path=str(self.teacher_forced_path),
                raw_metrics_hash=self.teacher_forced_hash,
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
                execution_mode="metal_strict",
                compressed_page_metal_calls=10,
                dense_tail_metal_calls=10,
                merge_metal_calls=0,
                finalization_metal_calls=0,
                compressed_page_fallback_calls=0,
                dense_tail_fallback_calls=0,
                full_attention_fallback_calls=0,
                fallback_reasons=[],
                actual_fused_positions=128,
                requested_fused_positions_per_context=128,
                positions_per_context={ctx: 128 for ctx in PromotionGate.REQUIRED_CONTEXTS},
                failed_positions_per_context={ctx: 0 for ctx in PromotionGate.REQUIRED_CONTEXTS},
                fallback_calls_per_context={ctx: 0 for ctx in PromotionGate.REQUIRED_CONTEXTS},
                trace_artifact_path=str(self.fused_trace_path),
                trace_artifact_hash=self.fused_trace_hash,
            ),
            speed_report=SpeedReport(
                min_ratio_at_4096_plus=0.98,
                max_ratio_at_4096_plus=1.06,
                median_ratio_at_8192_plus=1.04,
                trials_per_context=5,
                contexts_evaluated=list(PromotionGate.REQUIRED_CONTEXTS),
                execution_mode="metal_strict",
                raw_timing_path=str(self.speed_timing_path),
                raw_timing_hash=self.speed_timing_hash,
            ),
            memory_report=MemoryReport(
                logical_kv_ratio=1.90,
                persistent_storage_ratio=1.80,
                peak_device_memory_ratio_at_8192_plus=1.25,
                hidden_dense_cache_detected=False,
                contexts_evaluated=list(PromotionGate.REQUIRED_CONTEXTS),
            ),
            baseline_comparison_report=BaselineComparisonReport(
                cartesian_int8_baseline_implemented=True,
                turbo_polar_wins_on_speed=True,
                contexts_evaluated=list(PromotionGate.REQUIRED_CONTEXTS),
            ),
            provenance=BenchmarkProvenance(
                git_tree_state=GitTreeState.DIRTY if dirty else GitTreeState.CLEAN,
                git_diff_hash="abcd1234" if dirty else "",
                model_repo_id="test/model",
                model_revision="abc123def",
                tokenizer_revision="abc123def",
                turbopolar_config_hash="def456",
                evidence_kind="experimental",
                token_fixtures_hash="a" * 64,
            ),
        )

    def test_clean_tree_can_promote(self):
        evidence = self._full_passing_evidence(dirty=False)
        decision = PromotionGate().evaluate(evidence)
        # This test uses minimal/malformed artifacts, so it should return FAILED
        # A valid but locked package would return REVIEW_REQUIRED
        self.assertEqual(decision.state, PromotionState.FAILED)
        self.assertTrue(PromotionGate.PROMOTION_LOCKED)

    def test_dirty_tree_blocks(self):
        evidence = self._full_passing_evidence(dirty=True)
        decision = PromotionGate().evaluate(evidence)
        # Dirty tree should fail
        self.assertEqual(decision.state, PromotionState.FAILED)
        self.assertTrue(any("dirty" in r.lower() for r in decision.reasons))


if __name__ == "__main__":
    unittest.main()
