"""Round-trip tests for evidence schema serialization.

Every evidence schema must satisfy:
    from_dict(asdict(x)) == x
or a semantically equivalent invariant.
"""

import unittest
from dataclasses import asdict

from rfsn_v11.promotion.schema import (
    BenchmarkProvenance,
    BaselineComparisonReport,
    FusedDecodeReport,
    GitTreeState,
    KernelReport,
    MemoryReport,
    PromotionEvidence,
    SpeedReport,
    TeacherForcedReport,
)


class TestEvidenceRoundTrip(unittest.TestCase):
    """Test that evidence schemas round-trip through serialization."""

    def test_benchmark_provenance_round_trip(self):
        """BenchmarkProvenance must round-trip through asdict/from_dict."""
        original = BenchmarkProvenance(
            run_id="run_123",
            timestamp_utc="2026-01-01T00:00:00Z",
            git_commit="a" * 40,
            git_tree_state=GitTreeState.CLEAN,
            git_diff_hash="",
            python_version="3.12.0",
            mlx_version="0.31.2",
            mlx_lm_version="0.31.2",
            macos_version="14.0",
            chip_model="M3",
            system_memory_gb=32.0,
            model_repo_id="test/model",
            model_revision="abc123def456",
            tokenizer_revision="def456abc123",
            prompt_suite_hash="p" * 64,
            token_fixtures_hash="t" * 64,
            turbopolar_config_hash="c" * 64,
            turbopolar_config={"key": "value"},
            benchmark_command="python run.py",
            warmup_count=1,
            trial_count=5,
            context_lengths=[512, 2048, 4096],
            decode_token_count=128,
            qjl_enabled=False,
            metal_kernel_source_hash="m" * 64,
            kernel_binding_hash="k" * 64,
            execution_mode="metal_strict",
            evidence_kind="experimental",
        )
        d = asdict(original)
        restored = BenchmarkProvenance.from_dict(d)

        self.assertEqual(restored.run_id, original.run_id)
        self.assertEqual(restored.git_commit, original.git_commit)
        self.assertEqual(restored.git_tree_state, original.git_tree_state)
        self.assertEqual(restored.macos_version, original.macos_version)
        self.assertEqual(restored.chip_model, original.chip_model)
        self.assertEqual(restored.kernel_binding_hash, original.kernel_binding_hash)
        self.assertEqual(restored.execution_mode, original.execution_mode)
        self.assertEqual(restored.turbopolar_config, original.turbopolar_config)
        self.assertEqual(restored.evidence_kind, original.evidence_kind)

    def test_kernel_report_round_trip(self):
        """KernelReport must round-trip."""
        original = KernelReport(
            all_unit_tests_passed=True,
            all_kernel_tests_passed=True,
            cpu_metal_agreement_verified=True,
            metal_tests_present=["test1", "test2"],
            metal_tests_passed=["test1"],
            notes=["note1"],
        )
        restored = KernelReport.from_dict(asdict(original))
        self.assertEqual(restored.all_unit_tests_passed, original.all_unit_tests_passed)
        self.assertEqual(restored.metal_tests_present, original.metal_tests_present)
        self.assertEqual(restored.notes, original.notes)

    def test_teacher_forced_report_round_trip(self):
        """TeacherForcedReport must round-trip."""
        original = TeacherForcedReport(
            model="test",
            evaluated_contexts=[512],
            total_positions=128,
            mean_logit_cosine=0.999,
            p05_logit_cosine=0.995,
            min_logit_cosine=0.980,
            argmax_agreement=1.0,
            mean_top5_overlap=0.96,
            mean_perplexity_delta=0.01,
            any_nans_or_infs=False,
            raw_metrics_path="/tmp/metrics.json",
            raw_metrics_hash="h" * 64,
            notes=["good"],
        )
        restored = TeacherForcedReport.from_dict(asdict(original))
        self.assertEqual(restored.mean_logit_cosine, original.mean_logit_cosine)
        self.assertEqual(restored.raw_metrics_hash, original.raw_metrics_hash)

    def test_speed_report_round_trip(self):
        """SpeedReport must round-trip."""
        original = SpeedReport(
            model="test",
            contexts_evaluated=[512, 2048],
            min_ratio_at_4096_plus=1.1,
            max_ratio_at_4096_plus=1.2,
            median_ratio_at_8192_plus=1.15,
            trials_per_context=5,
            execution_mode="metal_strict",
            fallback_calls=0,
            raw_timing_path="/tmp/timing.json",
            raw_timing_hash="h" * 64,
            notes=["fast"],
        )
        restored = SpeedReport.from_dict(asdict(original))
        self.assertEqual(restored.execution_mode, original.execution_mode)
        self.assertEqual(restored.raw_timing_hash, original.raw_timing_hash)
        self.assertEqual(restored.min_ratio_at_4096_plus, original.min_ratio_at_4096_plus)

    def test_memory_report_round_trip(self):
        """MemoryReport must round-trip."""
        original = MemoryReport(
            model="test",
            contexts_evaluated=[512, 2048],
            logical_kv_ratio=1.9,
            persistent_storage_ratio=1.8,
            hidden_dense_cache_detected=False,
            notes=["good"],
        )
        restored = MemoryReport.from_dict(asdict(original))
        self.assertEqual(restored.logical_kv_ratio, original.logical_kv_ratio)
        self.assertEqual(restored.hidden_dense_cache_detected, original.hidden_dense_cache_detected)

    def test_baseline_comparison_round_trip(self):
        """BaselineComparisonReport must round-trip."""
        original = BaselineComparisonReport(
            model="test",
            contexts_evaluated=[512],
            cartesian_int8_baseline_implemented=True,
            turbo_polar_wins_on_speed=True,
            notes=["wins"],
        )
        restored = BaselineComparisonReport.from_dict(asdict(original))
        self.assertEqual(restored.cartesian_int8_baseline_implemented, original.cartesian_int8_baseline_implemented)
        self.assertEqual(restored.turbo_polar_wins_on_speed, original.turbo_polar_wins_on_speed)

    def test_fused_decode_report_round_trip(self):
        """FusedDecodeReport must round-trip."""
        original = FusedDecodeReport(
            model="test",
            model_layer_count=32,
            contexts_evaluated=[512],
            execution_mode="metal_strict",
            compressed_page_metal_calls=10,
            dense_tail_metal_calls=10,
            actual_fused_positions=128,
            requested_fused_positions_per_context=128,
            positions_per_context={512: 128},
            failed_positions_per_context={512: 0},
            fallback_calls_per_context={512: 0},
            trace_artifact_path="/tmp/trace.json",
            trace_artifact_hash="h" * 64,
            notes=["good"],
        )
        restored = FusedDecodeReport.from_dict(asdict(original))
        self.assertEqual(restored.execution_mode, original.execution_mode)
        self.assertEqual(restored.trace_artifact_hash, original.trace_artifact_hash)
        self.assertEqual(restored.positions_per_context, original.positions_per_context)

    def test_full_promotion_evidence_round_trip(self):
        """Complete PromotionEvidence must round-trip and produce same gate decision."""
        from rfsn_v11.promotion import PromotionGate, PromotionState

        evidence = PromotionEvidence(
            kernel_report=KernelReport(
                all_unit_tests_passed=True,
                all_kernel_tests_passed=True,
                all_integration_tests_passed=True,
                cpu_metal_agreement_verified=True,
            ),
            teacher_forced_report=TeacherForcedReport(
                mean_logit_cosine=0.999,
                p05_logit_cosine=0.995,
                min_logit_cosine=0.980,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=1.0,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
                total_positions=128,
            ),
            fused_decode_report=FusedDecodeReport(
                mean_logit_cosine=0.999,
                p05_logit_cosine=0.995,
                min_logit_cosine=0.980,
                mean_top5_overlap=0.96,
                mean_top10_overlap=0.98,
                argmax_agreement=1.0,
                mean_perplexity_delta=0.01,
                any_nans_or_infs=False,
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                execution_mode="metal_strict",
                model_layer_count=32,
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
                positions_per_context={ctx: 128 for ctx in [512, 2048, 4096, 8192, 16384]},
                failed_positions_per_context={ctx: 0 for ctx in [512, 2048, 4096, 8192, 16384]},
                fallback_calls_per_context={ctx: 0 for ctx in [512, 2048, 4096, 8192, 16384]},
            ),
            speed_report=SpeedReport(
                min_ratio_at_4096_plus=1.25,
                max_ratio_at_4096_plus=1.30,
                median_ratio_at_8192_plus=1.28,
                trials_per_context=5,
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
                execution_mode="metal_strict",
                fallback_calls=0,
            ),
            memory_report=MemoryReport(
                logical_kv_ratio=1.90,
                persistent_storage_ratio=1.80,
                peak_device_memory_ratio_at_8192_plus=1.25,
                hidden_dense_cache_detected=False,
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
            ),
            baseline_comparison_report=BaselineComparisonReport(
                cartesian_int8_baseline_implemented=True,
                turbo_polar_wins_on_speed=True,
                contexts_evaluated=[512, 2048, 4096, 8192, 16384],
            ),
            provenance=BenchmarkProvenance(
                run_id="test_run",
                git_commit="a" * 40,
                git_tree_state=GitTreeState.CLEAN,
                python_version="3.12.0",
                mlx_version="0.31.2",
                macos_version="14.0",
                chip_model="M3",
                model_repo_id="test/model",
                model_revision="abc123def456",
                tokenizer_revision="def456abc123",
                turbopolar_config_hash="c" * 64,
                metal_kernel_source_hash="m" * 64,
                kernel_binding_hash="k" * 64,
                execution_mode="metal_strict",
                evidence_kind="experimental",
                token_fixtures_hash="t" * 64,
            ),
        )

        gate = PromotionGate()
        decision_before = gate.evaluate(evidence)

        # Round-trip through serialization
        d = asdict(evidence)
        restored = PromotionEvidence.from_dict(d)
        decision_after = gate.evaluate(restored)

        self.assertEqual(
            decision_before.state, decision_after.state,
            f"Gate decision changed after round-trip. Before: {decision_before.state}, "
            f"After: {decision_after.state}. Reasons before: {decision_before.reasons}, "
            f"Reasons after: {decision_after.reasons}"
        )


if __name__ == "__main__":
    unittest.main()
