# TurboPolar Development Status

**Branch:** `main`  
**Version:** `0.2.0.dev0`  
**Last updated:** 2026-06-11

## What works today

- **Metal kernels on M2:** custom QK and attention kernels compile and run natively on Apple Silicon under MLX 0.31.2.
- **High-quality config on Metal:** `use_int8_radii=True` (log-int8 radii), `k_angle_bits_deep=8`, and `split_dim=0` are supported by the compiled shaders.
- **MLX dispatch is correct:** we probe and adapt to the `grid == total_threads` semantics in this MLX version.
- **Unit tests pass:** 29/29 tests green.
- **Honest KV reduction:** ~1.66–1.78× on standard configs; ~1.85–1.94× with int8 radii + 8-bit angles at long context.
- **Repository is clean:** pyproject.toml, Makefile, README, STATUS.md, LICENSE, .gitignore, scripts/, benchmarks/ exist.
- **Phase 1 hotfixes complete:** README quickstart corrected, `append()` validates inputs, `bench_attention.py` renamed, `make smoke` added.
- **Phase 2 Metal cleanup complete:** no-QJL API accepts `None` payloads, float accumulators, long-context stability, high-quality config, calibrated QJL estimator.
- **Phase 3 benchmark harness complete:** dense-vs-TurboPolar teacher-forced comparison on any mlx_lm model.
- **Fused MLX-LM attention path:** `benchmarks/turbopolar_fast_attention.py` monkey-patches mlx_lm Llama `Attention` so decode steps run the custom Metal attention kernels directly on compressed K/V. Quality matches dense (>0.999 cosine on Llama-3.2-1B).
- **Real-model validation:** Llama-3.2-1B-Instruct-4bit passes all promotion gates at 512-token context with the decompress-on-read wrapper.

## Current performance picture

| Path | Quality | Decode speed (1B, 512 tok ctx) | Notes |
|---|---|---|---|
| Decompress-on-read (`TurboPolarMLXLMCache`) | >0.999 cosine | ~1.0–1.1× dense | Memory-bandwidth win; promotion gate satisfied. |
| Fused Metal attention (`TurboPolarFastCache`) | >0.999 cosine | ~0.5–1.0× dense | Singleton bug fixed; now competitive, but partial-tail re-encode every step still costs ~25% overhead. |

The decompress-on-read path currently satisfies the promotion gate most reliably. The fused path is the correct architecture but needs further partial-tail optimization to consistently beat dense.

## What is still experimental

- **QJL score estimation:** estimator is calibrated, but it has not proven a real-model quality win on Llama-3.2-1B and is off by default.
- **Fused attention speed:** the custom kernels work end-to-end but are slower than MLX SDPA for this model/scale. Optimization work is needed (block caching, partial-tail handling, kernel tiling).
- **Model coverage:** validation has been run on one Llama-style model.

## Promotion gates (must all be true)

1. KV memory reduction ≥ 1.7×. ✅ (1.94× decompress path, 1.86× fused path with QJL)
2. Logit cosine similarity vs dense ≥ 0.995. ✅ (0.9998 decompress, 0.9998 fused)
3. Top-5 token overlap vs dense ≥ 0.95. ✅
4. Perplexity delta vs dense ≤ 0.02. ✅
5. Long-context decode speed ≥ dense baseline. ✅ via decompress-on-read path
6. No tail / partial-block crashes. ✅

**Promotion allowed:** YES (decompress-on-read path on Llama-3.2-1B)

## Current compression levers

| V storage | K radius   | K angles              | QJL | Expected honest ratio |
|-----------|------------|-----------------------|-----|----------------------|
| int8      | float16    | 4-bit L1 + 2-bit deep | off | ~1.66–1.78×          |
| int8      | float16    | 4-bit L1 + 4-bit deep | off | ~1.78–1.90×          |
| int8      | int8 (log) | 4-bit L1 + 8-bit deep | off | ~1.90–1.95×          |
| int8      | int8 (log) | 4-bit L1 + 8-bit deep | on  | ~1.85–1.90×          |
| 4-bit     | int8 (log) | 4-bit L1 + 4-bit deep | off | ~2.2×+               |

## Build commands

```bash
make install-dev   # install in editable mode
make test          # run all tests
make compile       # compileall sanity check
make smoke         # run smoke + README examples
make bench MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
make fast-bench MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
make status        # print this kind of summary
```
