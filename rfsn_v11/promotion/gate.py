"""Single promotion authority for TurboPolar.

No other component may declare promotion. All required evidence must be present
and passing; missing evidence results in INCOMPLETE/FAILED.
"""

from pathlib import Path
from typing import List

from rfsn_v11.promotion.schema import (
    GitTreeState,
    PromotionDecision,
    PromotionEvidence,
    PromotionState,
)


class PromotionGate:
    """Evaluate PromotionEvidence and render a single PromotionDecision."""

    # Locked until the strict no-fallback Metal suite passes end-to-end.
    # A correct fallback result does not prove the Metal implementation works.
    # This must remain True until all evidence systems are scientifically trustworthy.
    PROMOTION_LOCKED = True

    # Correctness thresholds
    MEAN_COSINE = 0.995
    P05_COSINE = 0.990
    MIN_COSINE = 0.975
    MEAN_TOP5 = 0.95
    MEAN_TOP10 = 0.97
    ARGMAX_AGREE = 0.97
    MAX_PPL_DELTA = 0.02

    # Memory thresholds
    LOGICAL_KV_RATIO = 1.85
    PERSISTENT_STORAGE_RATIO = 1.75
    PEAK_MEMORY_RATIO_8192 = 1.20

    # Speed thresholds
    MAX_REGRESSION_AT_4096_PLUS = 0.97  # no more than 3% regression
    MIN_IMPROVEMENT_AT_ANY_LONG_CONTEXT = 1.05
    MIN_MEDIAN_RATIO_AT_8192_PLUS = 1.03

    # Experiment completeness requirements
    REQUIRED_CONTEXTS = {512, 2048, 4096, 8192, 16384}
    REQUIRED_FORCED_DECODE_TOKENS = 128
    MIN_TRIALS_PER_CONTEXT = 5

    # Required native Metal test modules (prefix match).
    REQUIRED_NATIVE_METAL_TESTS = {
        "tests.kernels.test_paged_online_attention",
        "tests.kernels.test_qjl_scaled_fused_qk",
        "tests.kernels.test_qjl_scaled_online_attention",
        "tests.kernels.test_metal_strict",
        "tests.kernels.test_fallback_injection",
        "tests.benchmarks.test_turbopolar_fast_attention",
        "tests.benchmarks.test_turbo_polar_online_attention",
    }

    def evaluate(self, evidence: PromotionEvidence) -> PromotionDecision:
        reasons: List[str] = []

        if evidence is None:
            return PromotionDecision(
                state=PromotionState.INCOMPLETE,
                reasons=["No promotion evidence provided."],
            )

        # Missing reports are failures, not successes.
        required_reports = [
            ("kernel_report", evidence.kernel_report),
            ("teacher_forced_report", evidence.teacher_forced_report),
            ("fused_decode_report", evidence.fused_decode_report),
            ("speed_report", evidence.speed_report),
            ("memory_report", evidence.memory_report),
            ("baseline_comparison_report", evidence.baseline_comparison_report),
            ("provenance", evidence.provenance),
        ]
        for name, report in required_reports:
            if report is None:
                reasons.append(f"Missing required report: {name}")

        if reasons:
            return PromotionDecision(
                state=PromotionState.INCOMPLETE,
                reasons=reasons,
                evidence=evidence,
            )

        # Kernel correctness
        kr = evidence.kernel_report
        if not kr.all_unit_tests_passed:
            reasons.append("Unit tests did not all pass.")
        if not kr.all_kernel_tests_passed:
            reasons.append("Kernel tests did not all pass.")
        if not kr.all_integration_tests_passed:
            reasons.append("Integration tests did not all pass.")

        # Required native Metal tests
        present = set(kr.metal_tests_present)
        passed = set(kr.metal_tests_passed)
        missing = self.REQUIRED_NATIVE_METAL_TESTS - present
        if missing:
            reasons.append(
                f"Required Metal tests missing from collection: {sorted(missing)}"
            )
        failed = self.REQUIRED_NATIVE_METAL_TESTS - passed
        if failed:
            reasons.append(f"Required Metal tests did not pass: {sorted(failed)}")

        # Validate skipped status
        if kr.metal_tests_skipped:
            reasons.append(f"Native Metal tests were skipped: {sorted(kr.metal_tests_skipped)}")

        # Teacher-forced quality
        tf = evidence.teacher_forced_report
        if tf.mean_logit_cosine is None or tf.mean_logit_cosine < self.MEAN_COSINE:
            reasons.append(
                f"Teacher-forced mean cosine {tf.mean_logit_cosine} < {self.MEAN_COSINE}"
            )
        if tf.p05_logit_cosine is None or tf.p05_logit_cosine < self.P05_COSINE:
            reasons.append(
                f"Teacher-forced p05 cosine {tf.p05_logit_cosine} < {self.P05_COSINE}"
            )
        if tf.min_logit_cosine is None or tf.min_logit_cosine < self.MIN_COSINE:
            reasons.append(
                f"Teacher-forced min cosine {tf.min_logit_cosine} < {self.MIN_COSINE}"
            )
        if tf.mean_top5_overlap is None or tf.mean_top5_overlap < self.MEAN_TOP5:
            reasons.append(
                f"Teacher-forced top-5 overlap {tf.mean_top5_overlap} < {self.MEAN_TOP5}"
            )
        if tf.mean_top10_overlap is None or tf.mean_top10_overlap < self.MEAN_TOP10:
            reasons.append(
                f"Teacher-forced top-10 overlap {tf.mean_top10_overlap} < {self.MEAN_TOP10}"
            )
        if tf.argmax_agreement is None or tf.argmax_agreement < self.ARGMAX_AGREE:
            reasons.append(
                f"Teacher-forced argmax agreement {tf.argmax_agreement} < {self.ARGMAX_AGREE}"
            )
        if (
            tf.mean_perplexity_delta is None
            or tf.mean_perplexity_delta > self.MAX_PPL_DELTA
        ):
            reasons.append(
                f"Teacher-forced absolute perplexity delta {tf.mean_perplexity_delta} > {self.MAX_PPL_DELTA}"
            )
        if tf.any_nans_or_infs:
            reasons.append("Teacher-forced run contained NaNs or infinities.")

        # Fused decode quality
        fd = evidence.fused_decode_report
        if fd.mean_logit_cosine is None or fd.mean_logit_cosine < self.MEAN_COSINE:
            reasons.append(
                f"Fused decode mean cosine {fd.mean_logit_cosine} < {self.MEAN_COSINE}"
            )
        if fd.p05_logit_cosine is None or fd.p05_logit_cosine < self.P05_COSINE:
            reasons.append(
                f"Fused decode p05 cosine {fd.p05_logit_cosine} < {self.P05_COSINE}"
            )
        if fd.min_logit_cosine is None or fd.min_logit_cosine < self.MIN_COSINE:
            reasons.append(
                f"Fused decode min cosine {fd.min_logit_cosine} < {self.MIN_COSINE}"
            )
        if fd.mean_top5_overlap is None or fd.mean_top5_overlap < self.MEAN_TOP5:
            reasons.append(
                f"Fused decode top-5 overlap {fd.mean_top5_overlap} < {self.MEAN_TOP5}"
            )
        if fd.mean_top10_overlap is None or fd.mean_top10_overlap < self.MEAN_TOP10:
            reasons.append(
                f"Fused decode top-10 overlap {fd.mean_top10_overlap} < {self.MEAN_TOP10}"
            )
        if fd.argmax_agreement is None or fd.argmax_agreement < self.ARGMAX_AGREE:
            reasons.append(
                f"Fused decode argmax agreement {fd.argmax_agreement} < {self.ARGMAX_AGREE}"
            )
        if (
            fd.mean_perplexity_delta is None
            or fd.mean_perplexity_delta > self.MAX_PPL_DELTA
        ):
            reasons.append(
                f"Fused decode absolute perplexity delta {fd.mean_perplexity_delta} > {self.MAX_PPL_DELTA}"
            )
        if fd.any_nans_or_infs:
            reasons.append("Fused decode run contained NaNs or infinities.")

        # Trace artifact validation
        # Skip artifact validation for synthetic dry-run evidence
        if evidence.provenance.evidence_kind == "synthetic_dry_run":
            if fd.trace_artifact_path or fd.trace_artifact_hash:
                reasons.append("Synthetic dry-run evidence should not have trace artifacts.")
        else:
            if not fd.trace_artifact_path:
                reasons.append("Fused decode trace artifact path is missing.")
            elif not Path(fd.trace_artifact_path).exists():
                reasons.append(f"Fused decode trace artifact file does not exist: {fd.trace_artifact_path}")
            if not fd.trace_artifact_hash:
                reasons.append("Fused decode trace artifact hash is missing.")
            
            # Recompute and verify trace artifact hash
            if fd.trace_artifact_path and fd.trace_artifact_hash and Path(fd.trace_artifact_path).exists():
                import hashlib
                import json
                try:
                    with open(fd.trace_artifact_path, "r") as f:
                        trace_data = f.read()
                    computed_hash = hashlib.sha256(trace_data.encode()).hexdigest()
                    if computed_hash != fd.trace_artifact_hash:
                        reasons.append(
                            f"Trace artifact hash mismatch: computed {computed_hash[:16]} != recorded {fd.trace_artifact_hash[:16]}"
                        )
                    
                    # Parse and validate trace topology
                    traces = json.loads(trace_data)
                    if not isinstance(traces, list):
                        reasons.append("Trace artifact is not a list of traces.")
                    else:
                        # Validate trace structure
                        for trace in traces:
                            if not isinstance(trace, dict):
                                reasons.append("Trace entry is not a dictionary.")
                                continue
                            
                            # Check page traces
                            page_traces = trace.get("page_traces", [])
                            if not isinstance(page_traces, list):
                                reasons.append("page_traces is not a list.")
                                continue
                            
                            # Validate each page trace
                            for page_trace in page_traces:
                                if not isinstance(page_trace, dict):
                                    reasons.append("Page trace is not a dictionary.")
                                    continue
                                
                                # Require Metal execution
                                if not page_trace.get("metal_executed", False):
                                    reasons.append(
                                        f"Page trace layer={page_trace.get('layer_index')} "
                                        f"step={page_trace.get('decode_step')} "
                                        f"page={page_trace.get('page_index')} "
                                        "did not execute Metal kernel."
                                    )
                                
                                # Require no fallback
                                if page_trace.get("fallback_used", False):
                                    reasons.append(
                                        f"Page trace layer={page_trace.get('layer_index')} "
                                        f"step={page_trace.get('decode_step')} "
                                        f"page={page_trace.get('page_index')} "
                                        f"used fallback: {page_trace.get('fallback_reason')}"
                                    )
                                
                                # Require output evaluation
                                if not page_trace.get("output_evaluated", False):
                                    reasons.append(
                                        f"Page trace layer={page_trace.get('layer_index')} "
                                        f"step={page_trace.get('decode_step')} "
                                        f"page={page_trace.get('page_index')} "
                                        "output was not evaluated."
                                    )
                            
                            # Check dense tail trace
                            dense_tail = trace.get("dense_tail_trace")
                            if dense_tail and isinstance(dense_tail, dict):
                                if not dense_tail.get("metal_executed", False):
                                    reasons.append(
                                        f"Dense tail trace layer={dense_tail.get('layer_index')} "
                                        f"step={dense_tail.get('decode_step')} "
                                        "did not execute Metal kernel."
                                    )
                                if dense_tail.get("fallback_used", False):
                                    reasons.append(
                                        f"Dense tail trace layer={dense_tail.get('layer_index')} "
                                        f"step={dense_tail.get('decode_step')} "
                                        f"used fallback: {dense_tail.get('fallback_reason')}"
                                    )
                                if not dense_tail.get("output_evaluated", False):
                                    reasons.append(
                                        f"Dense tail trace layer={dense_tail.get('layer_index')} "
                                        f"step={dense_tail.get('decode_step')} "
                                        "output was not evaluated."
                                    )
                        
                        # Check for empty experiment IDs
                        for trace in traces:
                            page_traces = trace.get("page_traces", [])
                            for pt in page_traces:
                                if not pt.get("experiment_id"):
                                    reasons.append("Trace operation missing experiment_id.")
                                    break
                            dense_tail = trace.get("dense_tail_trace")
                            if dense_tail and not dense_tail.get("experiment_id"):
                                reasons.append("Dense tail trace missing experiment_id.")
                                break
                    
                except (json.JSONDecodeError, IOError) as e:
                    reasons.append(f"Failed to parse trace artifact: {e}")

        # Strict Metal execution verification.
        if fd.execution_mode is None:
            reasons.append("Fused decode execution_mode is missing.")
        elif fd.execution_mode != "metal_strict":
            reasons.append(
                f"Fused decode execution_mode='{fd.execution_mode}' != 'metal_strict'; "
                "only strict Metal runs are eligible for promotion."
            )

        # Required Metal dispatch counts must be present.
        # merge_metal_calls and finalization_metal_calls are optional because
        # merge/finalization currently use ordinary MLX operations, not custom
        # Metal kernels. This is a known hybrid execution policy.
        required_metal_fields = [
            "compressed_page_metal_calls",
            "dense_tail_metal_calls",
        ]
        for field in required_metal_fields:
            val = getattr(fd, field, None)
            if val is None:
                reasons.append(f"Fused decode missing required field: {field}")
            elif val == 0:
                reasons.append(f"Fused decode {field}=0; no Metal dispatches recorded.")
        if fd.merge_metal_calls is not None and fd.merge_metal_calls > 0:
            reasons.append("Fused decode reported merge_metal_calls>0, but merge uses MLX operations.")
        if fd.finalization_metal_calls is not None and fd.finalization_metal_calls > 0:
            reasons.append("Fused decode reported finalization_metal_calls>0, but finalization uses MLX operations.")

        # Fallback counts must be present and zero.
        required_fallback_fields = [
            "compressed_page_fallback_calls",
            "dense_tail_fallback_calls",
            "full_attention_fallback_calls",
        ]
        for field in required_fallback_fields:
            val = getattr(fd, field, None)
            if val is None:
                reasons.append(f"Fused decode missing required field: {field}")
            elif val != 0:
                reasons.append(
                    f"Fused decode {field}={val}; fallback occurred in strict mode."
                )

        if fd.fallback_reasons is None:
            reasons.append("Fused decode fallback_reasons missing.")
        elif fd.fallback_reasons:
            reasons.append(f"Fused decode had fallback reasons: {fd.fallback_reasons}")

        # Speed
        sr = evidence.speed_report
        if (
            sr.min_ratio_at_4096_plus is None
            or sr.min_ratio_at_4096_plus < self.MAX_REGRESSION_AT_4096_PLUS
        ):
            reasons.append(
                f"Speed ratio at 4096+ minimum {sr.min_ratio_at_4096_plus} < {self.MAX_REGRESSION_AT_4096_PLUS}"
            )
        
        # Require strict execution and zero fallbacks in speed evidence
        if sr.execution_mode is None:
            reasons.append("Speed evidence execution_mode is missing.")
        elif sr.execution_mode != "metal_strict":
            reasons.append(
                f"Speed evidence execution_mode='{sr.execution_mode}' != 'metal_strict'; "
                "only strict Metal runs are eligible for promotion."
            )
        if sr.fallback_calls != 0:
            reasons.append(
                f"Speed evidence fallback_calls={sr.fallback_calls}; fallback occurred in strict mode."
            )
        if not sr.raw_timing_hash:
            reasons.append("Speed evidence raw_timing_hash is missing.")
        
        # Require baseline contexts through 16K
        if 16384 not in sr.contexts_evaluated:
            reasons.append("Speed evidence missing required 16K context length.")
        
        # Baseline comparison must include 16K
        bc = evidence.baseline_comparison_report
        if bc and 16384 not in bc.contexts_evaluated:
            reasons.append("Baseline comparison missing required 16K context length.")
        if (
            sr.max_ratio_at_4096_plus is None
            or sr.max_ratio_at_4096_plus < self.MIN_IMPROVEMENT_AT_ANY_LONG_CONTEXT
        ):
            reasons.append(
                f"No long-context tier improved by >= {self.MIN_IMPROVEMENT_AT_ANY_LONG_CONTEXT}: "
                f"max ratio {sr.max_ratio_at_4096_plus}"
            )
        if (
            sr.median_ratio_at_8192_plus is None
            or sr.median_ratio_at_8192_plus < self.MIN_MEDIAN_RATIO_AT_8192_PLUS
        ):
            reasons.append(
                f"Median 8192+ speed ratio {sr.median_ratio_at_8192_plus} < {self.MIN_MEDIAN_RATIO_AT_8192_PLUS}"
            )

        # Memory
        mr = evidence.memory_report
        if mr.logical_kv_ratio is None or mr.logical_kv_ratio < self.LOGICAL_KV_RATIO:
            reasons.append(
                f"Logical KV ratio {mr.logical_kv_ratio} < {self.LOGICAL_KV_RATIO}"
            )
        if (
            mr.persistent_storage_ratio is None
            or mr.persistent_storage_ratio < self.PERSISTENT_STORAGE_RATIO
        ):
            reasons.append(
                f"Persistent storage ratio {mr.persistent_storage_ratio} < {self.PERSISTENT_STORAGE_RATIO}"
            )
        if (
            mr.peak_device_memory_ratio_at_8192_plus is None
            or mr.peak_device_memory_ratio_at_8192_plus < self.PEAK_MEMORY_RATIO_8192
        ):
            reasons.append(
                f"Peak memory ratio at 8192+ {mr.peak_device_memory_ratio_at_8192_plus} < {self.PEAK_MEMORY_RATIO_8192}"
            )
        if mr.hidden_dense_cache_detected:
            reasons.append("Hidden dense full-history cache detected.")

        # Baseline comparison
        br = evidence.baseline_comparison_report
        if not br.cartesian_int8_baseline_implemented:
            reasons.append("Cartesian int8 baseline not implemented.")
        elif not (
            br.turbo_polar_wins_on_quality
            or br.turbo_polar_wins_on_memory
            or br.turbo_polar_wins_on_speed
        ):
            reasons.append(
                "TurboPolar does not differentiate from Cartesian int8 on quality, memory, or speed."
            )

        # Experiment completeness: per-context fused decode evidence.
        # Use required-set inclusion so reports with diagnostic contexts still pass.
        fd = evidence.fused_decode_report
        contexts_set = (
            set(fd.contexts_evaluated) if fd and fd.contexts_evaluated else set()
        )
        if not self.REQUIRED_CONTEXTS.issubset(contexts_set):
            reasons.append(
                f"Fused decode contexts incomplete: required {self.REQUIRED_CONTEXTS} not in {contexts_set}"
            )

        # Per-context completeness gates.
        for context in self.REQUIRED_CONTEXTS:
            if context not in fd.positions_per_context:
                reasons.append(f"Missing fused decode positions for context {context}")
            elif fd.positions_per_context.get(context, 0) < self.REQUIRED_FORCED_DECODE_TOKENS:
                reasons.append(
                    f"Context {context}: fused positions {fd.positions_per_context.get(context, 0)} "
                    f"< required {self.REQUIRED_FORCED_DECODE_TOKENS}"
                )
            if fd.failed_positions_per_context.get(context, 0) != 0:
                reasons.append(f"Context {context}: has failed fused positions")
            if fd.fallback_calls_per_context.get(context, 0) != 0:
                reasons.append(f"Context {context}: has fallback calls")

        # Legacy fallback check (global).
        if fd and fd.actual_fused_positions is not None:
            if fd.actual_fused_positions < self.REQUIRED_FORCED_DECODE_TOKENS:
                reasons.append(
                    f"Fused decode actual positions {fd.actual_fused_positions} < "
                    f"required {self.REQUIRED_FORCED_DECODE_TOKENS}"
                )
        else:
            reasons.append("Fused decode actual_fused_positions missing.")

        sr = evidence.speed_report
        if sr and sr.trials_per_context < self.MIN_TRIALS_PER_CONTEXT:
            reasons.append(
                f"Speed trials per context {sr.trials_per_context} < {self.MIN_TRIALS_PER_CONTEXT}"
            )
        speed_contexts = set(sr.contexts_evaluated) if sr and sr.contexts_evaluated else set()
        if not self.REQUIRED_CONTEXTS.issubset(speed_contexts):
            reasons.append(
                f"Speed contexts incomplete: required {self.REQUIRED_CONTEXTS} not in {speed_contexts}"
            )
        mr = evidence.memory_report
        memory_contexts = set(mr.contexts_evaluated) if mr and mr.contexts_evaluated else set()
        if not self.REQUIRED_CONTEXTS.issubset(memory_contexts):
            reasons.append(
                f"Memory contexts incomplete: required {self.REQUIRED_CONTEXTS} not in {memory_contexts}"
            )

        # Provenance
        pv = evidence.provenance
        if pv.git_tree_state == GitTreeState.UNKNOWN:
            reasons.append("Git tree state unknown; cannot verify reproducibility.")
            # Do not soften hard quantitative failures into REVIEW_REQUIRED.
            if reasons[:-1]:
                pass  # Already has failures; continue to FAILED below.
            else:
                return PromotionDecision(
                    state=PromotionState.INCOMPLETE,
                    reasons=reasons,
                    evidence=evidence,
                )
        if pv.git_tree_state == GitTreeState.DIRTY:
            reasons.append(
                f"Source tree was dirty (diff hash {pv.git_diff_hash}); promotion requires a clean tree."
            )
        if not pv.model_repo_id or not pv.model_revision:
            reasons.append("Model provenance incomplete.")
        if not pv.turbopolar_config_hash:
            reasons.append("TurboPolar config hash missing.")

        if reasons:
            return PromotionDecision(
                state=PromotionState.FAILED,
                reasons=reasons,
                evidence=evidence,
            )

        if self.PROMOTION_LOCKED:
            return PromotionDecision(
                state=PromotionState.REVIEW_REQUIRED,
                reasons=[
                    "All quantitative thresholds pass, but promotion is capped at REVIEW_REQUIRED "
                    "until the full evidence suite (fused decode, isolated memory, fair baseline, "
                    "installed-wheel test) has been independently validated on native Apple Silicon."
                ],
                evidence=evidence,
            )

        return PromotionDecision(
            state=PromotionState.PROMOTED_EXPERIMENTAL,
            reasons=["All required evidence present and passing."],
            evidence=evidence,
        )
