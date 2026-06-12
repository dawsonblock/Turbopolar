# TurboPolar Benchmarks

This directory contains the real-model validation harness for TurboPolar.

## Files

- `run_dense_vs_turbopolar.py` — main benchmark script.
- `turbopolar_mlxlm_cache.py` — MLX-LM-compatible KV cache wrapper that stores K/V in TurboPolar format and decompresses on read.
- `report_schema.py` — report dataclass and promotion gate thresholds.
- `report_writer.py` — JSON and Markdown report writers.
- `prompt_suite.jsonl` — default prompt suite.
- `outputs/` — generated reports.

## Usage

Run against any mlx_lm-compatible model:

```bash
make bench MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
```

Or directly:

```bash
python benchmarks/run_dense_vs_turbopolar.py \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --prompt-suite benchmarks/prompt_suite.jsonl \
  --output-dir benchmarks/outputs \
  --max-tokens 128 \
  --num-decode 32
```

## What it measures

- **Logit cosine** between dense and TurboPolar teacher-forced outputs.
- **Top-5 / top-10 token overlap** per position.
- **KL divergence** and **perplexity delta**.
- **KV compression ratio** and **peak KV bytes**.
- **Decode tokens/sec** (optional, `--skip-decode-speed` to disable).

## Important caveat

The current wrapper decompresses K/V back to dense tensors so that standard MLX attention runs unchanged. This measures the **quality degradation from the compressed KV representation**, not the final fused-Metal-kernel speed. The optimized TurboPolar attention kernels are tested separately in `tests/benchmarks/test_turbo_polar_online_attention.py`.

## Expected result today

Promotion gates are **not** expected to pass on the first run. The benchmark is designed to produce a reproducible report that shows exactly how far the current quantization is from the dense baseline.
