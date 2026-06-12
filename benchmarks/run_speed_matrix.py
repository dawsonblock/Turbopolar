#!/usr/bin/env python3
"""Speed matrix benchmark for dense vs fused TurboPolar decode.

For each requested sequence length the script:
  1. Prefills the cache with that many tokens.
  2. Runs ``num_decode`` greedy decode steps.
  3. Selects the next token on-device to avoid CPU/GPU synchronization overhead.

Trials alternate which method runs first so that thermal throttling or
background scheduler noise are not systematically biased toward one path.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.integrations.mlx_lm.llama_adapter import TurboPolarLlamaAdapter
from benchmarks.prompt_fixtures import normalize_prompts
from benchmarks.report_writer import write_json_report
from benchmarks.turbopolar_fast_attention import make_turbo_caches


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


def _make_turbo_config(num_q_heads: int, num_kv_heads: int, head_dim: int) -> TurboPolarConfig:
    return TurboPolarConfig(
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


def _make_dense_cache(num_layers: int) -> List[KVCache]:
    return [KVCache() for _ in range(num_layers)]


def _device_side_next_token(logits: mx.array) -> mx.array:
    """Return the greedy next-token id on the device.

    logits shape: (B, T, V).  We take the last position and keep the result on
    the device so the following forward pass does not wait for a CPU round-trip.
    """
    # logits[:, -1, :] -> (B, V); argmax -> (B,)
    return mx.argmax(logits[:, -1, :], axis=-1)


def _measure_decode_loop(
    model,
    cache: List[Any],
    tokens: List[int],
    num_decode: int,
    warm_up: int = 2,
) -> float:
    """Prefill and decode, returning tok/s."""
    prompt_mx = mx.array(tokens)[None, :]
    model(prompt_mx, cache=cache)
    mx.eval(mx.array(0))

    # Warm-up decode steps to stabilise caches/kernels.
    next_token = mx.array([[tokens[-1]]])
    for _ in range(warm_up):
        logits = model(next_token, cache=cache)
        next_token = _device_side_next_token(logits).reshape(-1, 1)
        mx.eval(next_token)

    next_token = mx.array([[tokens[-1]]])
    start = time.perf_counter()
    for _ in range(num_decode):
        logits = model(next_token, cache=cache)
        next_token = _device_side_next_token(logits).reshape(-1, 1)
        mx.eval(next_token)
    elapsed = time.perf_counter() - start
    return num_decode / elapsed if elapsed > 0 else 0.0


def benchmark_length(
    model,
    tokens: List[int],
    num_decode: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    adapter: TurboPolarLlamaAdapter,
    turbo_first: bool,
) -> Tuple[float, float]:
    """Return (dense_tok_per_sec, turbo_tok_per_sec) for one prefill length.

    The ``turbo_first`` flag alternates which path is measured first.
    """
    num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)

    methods = [
        ("dense", lambda: _make_dense_cache(num_layers)),
        ("turbo", lambda: make_turbo_caches(num_layers, num_q_heads, num_kv_heads, head_dim)),
    ]
    if turbo_first:
        methods = list(reversed(methods))

    results: Dict[str, float] = {}
    for name, make_cache in methods:
        cache = make_cache()
        if name == "turbo":
            adapter.install(model)
            try:
                tok_s = _measure_decode_loop(model, cache, tokens, num_decode)
            finally:
                adapter.uninstall()
        else:
            tok_s = _measure_decode_loop(model, cache, tokens, num_decode)
        results[name] = tok_s

    return results["dense"], results["turbo"]


def main():
    parser = argparse.ArgumentParser(
        description="Decode speed matrix: dense KV cache vs fused TurboPolar"
    )
    parser.add_argument("--model", required=True, help="MLX model path or HF identifier")
    parser.add_argument(
        "--token-fixtures",
        type=Path,
        default=Path(__file__).parent / "exact_token_fixtures.jsonl",
        help="Exact-token fixture file used for deterministic prefill sequences",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048],
        help="Prefill lengths to benchmark",
    )
    parser.add_argument(
        "--num-decode", type=int, default=128, help="Decode steps per measurement"
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of alternating trials per length"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "speed_matrix",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer = load(str(args.model))

    num_q_heads, num_kv_heads, head_dim = _model_cache_config(model)
    adapter = TurboPolarLlamaAdapter(_make_turbo_config(num_q_heads, num_kv_heads, head_dim))

    normalized = normalize_prompts(tokenizer, args.token_fixtures)
    if not normalized:
        raise ValueError(f"No token fixtures found in {args.token_fixtures}")
    base_tokens = normalized[0]["tokens"]
    max_length = max(args.lengths)
    if len(base_tokens) < max_length:
        # Cycle through the fixture tokens to reach the required length.
        base_tokens = [
            base_tokens[i % len(base_tokens)] for i in range(max_length)
        ]

    records = []
    print(f"Benchmarking lengths: {args.lengths}")
    for length in sorted(args.lengths):
        tokens = base_tokens[:length]
        dense_rates = []
        turbo_rates = []
        for trial in range(args.trials):
            dense_tok_s, turbo_tok_s = benchmark_length(
                model,
                tokens,
                args.num_decode,
                num_q_heads,
                num_kv_heads,
                head_dim,
                adapter,
                turbo_first=(trial % 2 == 1),
            )
            dense_rates.append(dense_tok_s)
            turbo_rates.append(turbo_tok_s)
            print(
                f"  length={length} trial={trial + 1}/{args.trials} "
                f"dense={dense_tok_s:.2f} tok/s turbo={turbo_tok_s:.2f} tok/s"
            )

        record = {
            "length": length,
            "dense_mean_tok_per_sec": float(np.mean(dense_rates)),
            "dense_std_tok_per_sec": float(np.std(dense_rates)),
            "turbo_mean_tok_per_sec": float(np.mean(turbo_rates)),
            "turbo_std_tok_per_sec": float(np.std(turbo_rates)),
            "speedup": (
                float(np.mean(turbo_rates) / np.mean(dense_rates))
                if np.mean(dense_rates) > 0 else None
            ),
        }
        records.append(record)

    print("\n=== Speed Matrix ===")
    print(f"{'Length':>8} {'Dense tok/s':>14} {'Turbo tok/s':>14} {'Speedup':>10}")
    for r in records:
        print(
            f"{r['length']:>8} "
            f"{r['dense_mean_tok_per_sec']:>8.2f} ±{r['dense_std_tok_per_sec']:>4.2f} "
            f"{r['turbo_mean_tok_per_sec']:>8.2f} ±{r['turbo_std_tok_per_sec']:>4.2f} "
            f"{r['speedup'] if r['speedup'] is not None else 'n/a':>10}"
        )

    report = {
        "model": str(args.model),
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "dtype": _first_param_dtype(model.parameters()),
        "seed": args.seed,
        "num_decode": args.num_decode,
        "trials": args.trials,
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "speed_matrix.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {json_path}")


if __name__ == "__main__":
    main()
