#!/usr/bin/env python3
"""Cartesian int8 baseline comparison against dense and TurboPolar.

Produces a ``BaselineComparisonReport`` that the promotion suite can use to
decide whether TurboPolar differentiates from a simple int8 Cartesian baseline.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from benchmarks.cartesian_int8_cache import CartesianInt8Cache
from benchmarks.report_schema import BaselineComparisonReport
from benchmarks.report_writer import write_json_report, write_markdown_report
from benchmarks.turbopolar_mlxlm_cache import TurboPolarMLXLMCache


def _first_param_dtype(params: Dict[str, Any]) -> str:
    for v in params.values():
        if isinstance(v, dict):
            dtype = _first_param_dtype(v)
            if dtype is not None:
                return dtype
        elif hasattr(v, "dtype"):
            if "float" in str(v.dtype):
                return str(v.dtype)
    for v in params.values():
        if hasattr(v, "dtype"):
            return str(v.dtype)
    return "unknown"


def _model_cache_config(model: Any) -> Tuple[int, int, int]:
    n_heads = getattr(model, "n_heads", None)
    n_kv_heads = getattr(model, "n_kv_heads", None)
    hidden_size = getattr(model, "hidden_size", None)
    if n_heads is None or n_kv_heads is None or hidden_size is None:
        attn = None
        for module in model.modules():
            if type(module).__name__ == "Attention":
                attn = module
                break
        if attn is None:
            raise ValueError("Could not infer attention config from model")
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads
        hidden_size = attn.q_proj.weight.shape[0]
    return int(n_heads), int(n_kv_heads), int(hidden_size // n_heads)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _logit_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if np.isnan(a).any() or np.isnan(b).any():
        return 0.0
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12
    return float(np.dot(a_flat, b_flat) / denom)


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if np.isnan(a).any() or np.isnan(b).any():
        return 0.0
    if a.ndim == 2:
        a = a[None, ...]
        b = b[None, ...]
    matches = 0
    total = 0
    for batch in range(a.shape[0]):
        for t in range(a.shape[1]):
            top_a = set(np.argsort(a[batch, t])[-k:].tolist())
            top_b = set(np.argsort(b[batch, t])[-k:].tolist())
            matches += len(top_a & top_b)
            total += k
    return matches / total if total > 0 else 0.0


def _perplexity(logits: np.ndarray, tokens: List[int]) -> float:
    if np.isnan(logits).any():
        return float("inf")
    if logits.ndim == 2:
        logits = logits[None, ...]
    log_probs = _softmax(logits, axis=-1)
    token_log_probs = []
    for t in range(logits.shape[1] - 1):
        token_id = tokens[t + 1]
        token_log_probs.append(-np.log(log_probs[0, t, token_id] + 1e-12))
    return float(np.exp(np.mean(token_log_probs))) if token_log_probs else float("inf")


def _teacher_forced_logits(model, tokenizer, prompt_text: str, cache: List, max_tokens: int):
    tokens = tokenizer.encode(prompt_text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    tokens_mx = mx.array(tokens)[None, :]
    logits = model(tokens_mx, cache=cache)
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    return np.array(logits), tokens


def _make_dense_cache(num_layers: int) -> List[KVCache]:
    return [KVCache() for _ in range(num_layers)]


def _make_cartesian_cache(num_layers: int) -> List[CartesianInt8Cache]:
    return [CartesianInt8Cache() for _ in range(num_layers)]


def _make_turbo_cache(num_layers: int, num_q_heads: int, num_kv_heads: int, head_dim: int):
    config = TurboPolarConfig(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=64,
        qjl_proj_dim=64,
        use_qjl=False,
        storage_mode="kv_quant",
        use_int8_radii=True,
        k_angle_bits_deep=8,
        split_dim=0,
    )
    return [TurboPolarMLXLMCache(config) for _ in range(num_layers)]


def _load_prompts(path: Path) -> List[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                prompts.append(obj.get("prompt", obj.get("text", "")))
            elif isinstance(obj, str):
                prompts.append(obj)
    return prompts


def benchmark_prompt(
    model,
    tokenizer,
    prompt_text: str,
    max_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Dict[str, Any]:
    num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)

    dense_cache = _make_dense_cache(num_layers)
    dense_logits, tokens = _teacher_forced_logits(
        model, tokenizer, prompt_text, dense_cache, max_tokens
    )

    cart_cache = _make_cartesian_cache(num_layers)
    cart_logits, _ = _teacher_forced_logits(
        model, tokenizer, prompt_text, cart_cache, max_tokens
    )

    turbo_cache = _make_turbo_cache(num_layers, num_q_heads, num_kv_heads, head_dim)
    turbo_logits, _ = _teacher_forced_logits(
        model, tokenizer, prompt_text, turbo_cache, max_tokens
    )

    return {
        "prompt_tokens": len(tokens),
        "dense_peak_bytes": sum(c.nbytes for c in dense_cache),
        "cartesian_peak_bytes": sum(c.nbytes for c in cart_cache),
        "turbo_peak_bytes": sum(c.nbytes for c in turbo_cache),
        "cartesian_cosine": _logit_cosine(dense_logits, cart_logits),
        "turbo_cosine": _logit_cosine(dense_logits, turbo_logits),
        "cartesian_top5": _topk_overlap(dense_logits, cart_logits, k=5),
        "turbo_top5": _topk_overlap(dense_logits, turbo_logits, k=5),
        "cartesian_ppl_delta": abs(
            _perplexity(dense_logits, tokens) - _perplexity(cart_logits, tokens)
        ),
        "turbo_ppl_delta": abs(
            _perplexity(dense_logits, tokens) - _perplexity(turbo_logits, tokens)
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cartesian int8 baseline comparison"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-suite",
        type=Path,
        default=Path(__file__).parent / "prompt_suite.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "cartesian_baseline",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer = load(str(args.model))

    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
    prompts = _load_prompts(args.prompt_suite)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_suite}")

    records = []
    for i, prompt in enumerate(prompts):
        prompt_len = len(tokenizer.encode(prompt))
        print(f"Benchmarking prompt {i + 1}/{len(prompts)} ({prompt_len} tokens)")
        record = benchmark_prompt(
            model, tokenizer, prompt, args.max_tokens, num_q_heads, num_kv_heads, head_dim
        )
        records.append(record)
        print(
            f"  cartesian cos={record['cartesian_cosine']:.4f} "
            f"turbo cos={record['turbo_cosine']:.4f}"
        )

    # Wins are evaluated on gate-eligible prompts (>= 64 tokens) to avoid short-
    # prompt tail-only distortions.
    eligible = [r for r in records if r["prompt_tokens"] >= 64] or records

    turbo_wins_quality = (
        np.mean([r["turbo_cosine"] for r in eligible])
        >= np.mean([r["cartesian_cosine"] for r in eligible])
    )
    turbo_wins_memory = (
        np.mean([r["turbo_peak_bytes"] for r in eligible])
        < np.mean([r["cartesian_peak_bytes"] for r in eligible])
    )

    report = BaselineComparisonReport(
        model=str(args.model),
        contexts_evaluated=[r["prompt_tokens"] for r in records],
        cartesian_int8_baseline_implemented=True,
        turbo_polar_wins_on_quality=bool(turbo_wins_quality),
        turbo_polar_wins_on_memory=bool(turbo_wins_memory),
        turbo_polar_wins_on_speed=None,
        recommendation=(
            "TurboPolar differentiates from Cartesian int8 on "
            + ", ".join(
                label
                for flag, label in [
                    (turbo_wins_quality, "quality"),
                    (turbo_wins_memory, "memory"),
                ]
                if flag
            )
            or "no dimension"
        )
        + ".",
        notes=[
            f"mean cartesian cosine: {np.mean([r['cartesian_cosine'] for r in eligible]):.4f}",
            f"mean turbo cosine: {np.mean([r['turbo_cosine'] for r in eligible]):.4f}",
            f"mean cartesian bytes: {int(np.mean([r['cartesian_peak_bytes'] for r in eligible])):,}",
            f"mean turbo bytes: {int(np.mean([r['turbo_peak_bytes'] for r in eligible])):,}",
        ],
    )

    output = {
        "model": str(args.model),
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "dtype": _first_param_dtype(model.parameters()),
        "seed": args.seed,
        "records": records,
        "baseline_comparison_report": {
            k: v.value if hasattr(v, "value") else v
            for k, v in vars(report).items()
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "report.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Report written to {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
