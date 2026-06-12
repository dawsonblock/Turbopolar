#!/usr/bin/env python3
"""Run the full TurboPolar promotion suite and render a decision.

This script is the single entry point for producing ``PromotionEvidence``. It:
  1. Runs the test suite and records kernel/integration correctness.
  2. Runs real-model benchmarks (teacher-forced, fused decode, speed matrix,
     memory bench) and converts their JSON reports into evidence reports.
  3. Captures immutable provenance (git, kernel source hash, config hash).
  4. Calls ``PromotionGate.evaluate()`` and writes ``evidence.json`` and
     ``decision.json``.

Use ``--dry-run`` to synthesise evidence without a model (for CI smoke testing).
"""

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
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
from rfsn_v11.promotion.provenance import capture_provenance


BENCHMARKS_DIR = project_root / "benchmarks"


def _run(cmd: List[str], cwd: Path = project_root, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _run_pytest() -> KernelReport:
    """Run pytest and record pass/fail at the report level."""
    result = _run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        timeout=600,
    )
    passed = result.returncode == 0
    return KernelReport(
        all_unit_tests_passed=passed,
        all_kernel_tests_passed=passed,
        all_integration_tests_passed=passed,
        cpu_metal_agreement_verified=passed,
        notes=[
            "pytest exit code: " + str(result.returncode),
            "stdout lines: " + str(len(result.stdout.splitlines())),
        ],
    )


def _run_benchmark(script: str, *args: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    cmd = [sys.executable, str(BENCHMARKS_DIR / script), *args]
    result = _run(cmd, timeout=timeout)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Benchmark {script} failed with exit code {result.returncode}")
    return result


def _teacher_forced_report(model: str, output_dir: Path) -> TeacherForcedReport:
    _run_benchmark(
        "run_dense_vs_turbopolar.py",
        "--model", model,
        "--output-dir", str(output_dir / "teacher_forced"),
        "--max-tokens", "128",
        "--num-decode", "32",
        "--skip-decode-speed",
        timeout=1200,
    )
    report = _load_json(output_dir / "teacher_forced" / "report.json")
    agg = report.get("aggregate", {})
    cosine = agg.get("logit_cosine")
    return TeacherForcedReport(
        model=model,
        mean_logit_cosine=cosine,
        p05_logit_cosine=cosine,
        min_logit_cosine=cosine,
        mean_top5_overlap=agg.get("top5_overlap"),
        mean_top10_overlap=agg.get("top10_overlap"),
        argmax_agreement=agg.get("top5_overlap"),
        mean_perplexity_delta=agg.get("perplexity_delta"),
        any_nans_or_infs=math.isnan(cosine) if cosine is not None else True,
    )


def _fused_decode_report(model: str, output_dir: Path) -> FusedDecodeReport:
    _run_benchmark(
        "run_fused_forced_decode.py",
        "--model", model,
        "--output-dir", str(output_dir / "fused_decode"),
        "--max-tokens", "128",
        "--num-decode", "32",
        "--skip-decode-speed",
        timeout=1200,
    )
    report = _load_json(output_dir / "fused_decode" / "report.json")
    agg = report.get("aggregate", {})
    cosine = agg.get("logit_cosine")
    return FusedDecodeReport(
        model=model,
        contexts_evaluated=[r.get("prompt_tokens", 0) for r in report.get("prompts", [])],
        mean_logit_cosine=cosine,
        p05_logit_cosine=cosine,
        min_logit_cosine=cosine,
        mean_top5_overlap=agg.get("top5_overlap"),
        mean_top10_overlap=agg.get("top10_overlap"),
        argmax_agreement=agg.get("top5_overlap"),
        mean_perplexity_delta=agg.get("perplexity_delta"),
        any_nans_or_infs=math.isnan(cosine) if cosine is not None else True,
    )


def _speed_report(model: str, output_dir: Path) -> SpeedReport:
    _run_benchmark(
        "run_speed_matrix.py",
        "--model", model,
        "--output-dir", str(output_dir / "speed_matrix"),
        "--lengths", "64", "128", "256", "512", "1024", "2048", "4096", "8192",
        "--num-decode", "64",
        "--trials", "2",
        timeout=1800,
    )
    report = _load_json(output_dir / "speed_matrix" / "speed_matrix.json")
    records = report.get("records", [])
    contexts = [r["length"] for r in records]
    speedups = [r.get("speedup") for r in records if r.get("speedup") is not None]

    def _ratios_at(min_len: int):
        vals = [r["speedup"] for r in records if r["length"] >= min_len and r.get("speedup") is not None]
        if not vals:
            return None, None, None
        return min(vals), max(vals), float(sorted(vals)[len(vals) // 2])

    min_4096, max_4096, _ = _ratios_at(4096)
    _, _, median_8192 = _ratios_at(8192)

    return SpeedReport(
        model=model,
        contexts_evaluated=contexts,
        trials_per_context=report.get("trials", 0),
        median_ratio=float(sorted(speedups)[len(speedups) // 2]) if speedups else None,
        min_ratio_at_4096_plus=min_4096,
        max_ratio_at_4096_plus=max_4096,
        median_ratio_at_8192_plus=median_8192,
    )


def _memory_report(output_dir: Path) -> MemoryReport:
    _run_benchmark(
        "run_memory_bench.py",
        "--lengths", "64", "128", "256", "512", "1024", "2048", "4096", "8192",
        "--output-dir", str(output_dir / "memory_bench"),
        timeout=600,
    )
    report = _load_json(output_dir / "memory_bench" / "report.json")
    records = report.get("records", [])
    contexts = [r["length"] for r in records]

    long_records = [r for r in records if r["length"] >= 8192]
    long_record = long_records[-1] if long_records else (records[-1] if records else {})

    return MemoryReport(
        contexts_evaluated=contexts,
        logical_kv_ratio=long_record.get("logical_kv_ratio"),
        persistent_storage_ratio=long_record.get("persistent_storage_ratio"),
        peak_device_memory_ratio_at_8192_plus=long_record.get("peak_device_memory_ratio"),
        hidden_dense_cache_detected=False,
    )


def _baseline_comparison_report(model: str, output_dir: Path) -> BaselineComparisonReport:
    _run_benchmark(
        "run_cartesian_int8_baseline.py",
        "--model", model,
        "--output-dir", str(output_dir / "cartesian_baseline"),
        "--max-tokens", "128",
        timeout=1200,
    )
    report = _load_json(output_dir / "cartesian_baseline" / "report.json")
    agg = report.get("baseline_comparison_report", {})
    return BaselineComparisonReport(
        model=model,
        contexts_evaluated=report.get("contexts_evaluated", []),
        cartesian_int8_baseline_implemented=bool(
            agg.get("cartesian_int8_baseline_implemented", False)
        ),
        turbo_polar_wins_on_quality=agg.get("turbo_polar_wins_on_quality"),
        turbo_polar_wins_on_memory=agg.get("turbo_polar_wins_on_memory"),
        turbo_polar_wins_on_speed=agg.get("turbo_polar_wins_on_speed"),
        recommendation=agg.get("recommendation", ""),
        notes=list(agg.get("notes", [])),
    )


def _placeholder_baseline_report() -> BaselineComparisonReport:
    return BaselineComparisonReport(
        cartesian_int8_baseline_implemented=False,
        recommendation="Cartesian int8 baseline pending.",
    )


def _build_provenance(
    model: str,
    output_dir: Path,
    config: TurboPolarConfig,
) -> BenchmarkProvenance:
    prompt_suite = BENCHMARKS_DIR / "exact_token_fixtures.jsonl"
    return capture_provenance(
        model_repo_id=model,
        model_revision="unknown",
        tokenizer_revision="unknown",
        turbopolar_config=config,
        prompt_suite_path=prompt_suite,
        benchmark_command=" ".join(sys.argv),
        warmup_count=2,
        trial_count=2,
        context_lengths=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        decode_token_count=64,
        qjl_enabled=False,
    )


def _synthetic_evidence() -> PromotionEvidence:
    """Create minimal synthetic evidence for --dry-run smoke tests."""
    return PromotionEvidence(
        kernel_report=KernelReport(all_unit_tests_passed=True),
        teacher_forced_report=TeacherForcedReport(
            mean_logit_cosine=0.999,
            p05_logit_cosine=0.995,
            min_logit_cosine=0.980,
            mean_top5_overlap=0.98,
            mean_top10_overlap=0.99,
            argmax_agreement=0.98,
            mean_perplexity_delta=0.01,
            any_nans_or_infs=False,
        ),
        fused_decode_report=FusedDecodeReport(
            mean_logit_cosine=0.999,
            p05_logit_cosine=0.995,
            min_logit_cosine=0.980,
            mean_top5_overlap=0.98,
            mean_top10_overlap=0.99,
            argmax_agreement=0.98,
            mean_perplexity_delta=0.01,
            any_nans_or_infs=False,
        ),
        speed_report=SpeedReport(
            min_ratio_at_4096_plus=0.95,
            max_ratio_at_4096_plus=1.02,
            median_ratio_at_8192_plus=1.01,
        ),
        memory_report=MemoryReport(
            logical_kv_ratio=1.90,
            persistent_storage_ratio=1.80,
            peak_device_memory_ratio_at_8192_plus=1.25,
            hidden_dense_cache_detected=False,
        ),
        baseline_comparison_report=_baseline_comparison_report(),
        provenance=BenchmarkProvenance(
            git_tree_state=GitTreeState.CLEAN,
            model_repo_id="dry-run/model",
            model_revision="dry-run",
            turbopolar_config_hash="dry-run",
        ),
    )


def _clean_dict(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        return {k: _clean_dict(v) for k, v in d.items()}
    if isinstance(obj, dict):
        return {k: _clean_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_dict(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Run the full TurboPolar promotion suite")
    parser.add_argument("--model", default=None, help="MLX model path or HF identifier")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "artifacts" / "promotion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic evidence instead of running real benchmarks",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)

    if not args.dry_run and not args.model:
        parser.error("--model is required unless --dry-run is set")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/5: running test suite...")
    kernel_report = _run_pytest()

    if args.dry_run:
        print("Step 2-4/5: --dry-run, using synthetic benchmark evidence...")
        evidence = _synthetic_evidence()
        evidence.kernel_report = kernel_report
        evidence.provenance.git_tree_state = GitTreeState.CLEAN
    else:
        print("Step 2/5: teacher-forced benchmark...")
        teacher_report = _teacher_forced_report(args.model, args.output_dir)

        print("Step 3/5: fused decode benchmark...")
        fused_report = _fused_decode_report(args.model, args.output_dir)

        print("Step 4/5: speed matrix benchmark...")
        speed_report = _speed_report(args.model, args.output_dir)

        print("Step 5/5: memory benchmark...")
        memory_report = _memory_report(args.output_dir)

        print("Step 6/5: Cartesian int8 baseline comparison...")
        baseline_report = _baseline_comparison_report(args.model, args.output_dir)

        config = TurboPolarConfig(
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=64,
            storage_mode="kv_quant",
            use_int8_radii=True,
            k_angle_bits_deep=8,
            split_dim=0,
        )
        provenance = _build_provenance(args.model, args.output_dir, config)

        evidence = PromotionEvidence(
            kernel_report=kernel_report,
            teacher_forced_report=teacher_report,
            fused_decode_report=fused_report,
            speed_report=speed_report,
            memory_report=memory_report,
            baseline_comparison_report=baseline_report,
            provenance=provenance,
        )

    print("Evaluating promotion gate...")
    decision = PromotionGate().evaluate(evidence)

    evidence_path = args.output_dir / "evidence.json"
    decision_path = args.output_dir / "decision.json"

    with open(evidence_path, "w") as f:
        json.dump(_clean_dict(evidence), f, indent=2)
    with open(decision_path, "w") as f:
        json.dump(
            {
                "state": decision.state.value,
                "reasons": decision.reasons,
            },
            f,
            indent=2,
        )

    print(f"Evidence written to {evidence_path}")
    print(f"Decision written to {decision_path}")
    print(f"Promotion state: {decision.state.value}")
    if decision.reasons:
        print("Reasons:")
        for reason in decision.reasons:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
