# TurboPolar Development Status

**Branch:** `main`  
**Version:** `0.2.0.dev0`  
**Last updated:** 2026-06-11

## What works today

- **Metal kernels on M2:** custom QK and attention kernels compile and run natively on Apple Silicon under MLX 0.31.2.
- **MLX dispatch is correct:** we probe and adapt to the `grid == total_threads` semantics in this MLX version.
- **Unit tests pass:** 25/25 tests green; coverage includes configuration, polar key quantization, V quantization, QJL estimation, cache telemetry, CPU/Metal attention paths, and cache input validation.
- **Honest KV reduction:** ~1.66–1.78× on standard configs without faking the accounting.
- **Repository is clean:** pyproject.toml, Makefile, README, STATUS.md, LICENSE, .gitignore, scripts/, benchmarks/ exist.
- **Phase 1 hotfixes complete:** README quickstart corrected, `append()` validates inputs, `bench_attention.py` renamed to `bench_cache_compression.py`, `make smoke` added.
- **Phase 2 Metal cleanup complete:** online-attention no-QJL API accepts `None` payloads, Metal accumulators use `float` internally for long-context stability, 4k/8k stability test passes.
- **Phase 3 benchmark harness complete:** `benchmarks/run_dense_vs_turbopolar.py` runs dense-vs-TurboPolar teacher-forced comparison on any mlx_lm model, writes JSON + Markdown reports, and is wired to `make bench MODEL=...`.

## What is still experimental

- **QJL score estimation:** disabled by default. The current sign-agreement correction is a heuristic and has not passed a real-model quality gate.
- **Real model logits:** we have not yet validated against a Llama-style model. All correctness checks are synthetic or unit-level.
- **Promotion gate:** `promotion_allowed=False` until real-model gates pass.

## Promotion gates (must all be true)

1. KV memory reduction ≥ 1.7×.
2. Logit cosine similarity vs dense ≥ 0.995.
3. Top-5 token overlap vs dense ≥ 0.95.
4. Perplexity delta vs dense ≤ 0.02.
5. Long-context decode speed ≥ dense baseline.
6. No tail / partial-block crashes.

## Current compression levers

| V storage | K radius | QJL | Expected honest ratio |
|-----------|----------|-----|----------------------|
| int8      | float16  | off | ~1.66–1.78×          |
| int8      | int8     | off | ~2.0×+               |
| 4-bit     | float16  | off | ~2.0×+               |

Quantizing the radius code to int8 or V to 4-bit is the next step to exceed 2×.

## Build commands

```bash
make install-dev   # install in editable mode
make test          # run all tests
make compile       # compileall sanity check
make status        # print this kind of summary
```
