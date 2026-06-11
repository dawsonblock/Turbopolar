# TurboPolar Development Status

**Branch:** `main`  
**Version:** `0.2.0.dev0`  
**Last updated:** 2026-06-11

## What works today

- **Metal kernels on M2:** custom QK and attention kernels compile and run natively on Apple Silicon under MLX 0.31.2.
- **MLX dispatch is correct:** we probe and adapt to the `grid == total_threads` semantics in this MLX version.
- **Unit tests pass:** 24/24 tests green; coverage includes configuration, polar key quantization, V quantization, QJL estimation, cache telemetry, and both CPU/Metal attention paths.
- **Honest KV reduction:** ~1.66–1.78× on standard configs without faking the accounting.
- **Repository is clean:** pyproject.toml, Makefile, README, LICENSE, .gitignore, scripts/, benchmarks/ exist.

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
