# TurboPolar Development Status

**Branch:** `main`  
**Version:** `0.3.0.dev0`  
**Last updated:** 2026-06-12

## Status summary

**Promotion state:** Maximum state is capped at `REVIEW_REQUIRED`. `PROMOTED_EXPERIMENTAL` is **locked** until the full evidence suite has been independently validated on native Apple Silicon.

TurboPolar remains an end-to-end research alpha. It has a strict compressed-page
and dense-tail Metal prototype with synthetic multi-page GQA coverage. Explicit
`METAL_STRICT` and `DEVELOPMENT_AUTO` execution modes are defined, the model
runtime passes the mode into every attention call, and the forced-decode
benchmark can run in strict mode. The promotion pipeline expects strict Metal
evidence fields. However, real-model long-context, speed, memory, and
comparative-value evidence remains incomplete. Native Apple Silicon benchmark
artifacts are still required to prove the quantitative thresholds in
`rfsn_v11/promotion/gate.py`.
Promotion is blocked until reproducible artifacts independently prove:

1. **Correctness:** fused compressed attention produces acceptably close logits to dense attention during actual autoregressive decode.
2. **Memory:** measured device memory is reduced, not merely logical payload bytes.
3. **Performance:** steady-state decode is faster or at least non-regressive at long context under a sound benchmark.

No promotion claim may be made by any component other than the single promotion gate in `rfsn_v11/promotion/gate.py`.

## What works today

- **Metal kernels on M2:** custom QK and attention kernels compile and run natively on Apple Silicon under MLX 0.31.2.
- **MLX dispatch is correct:** we probe and adapt to the `grid == total_threads` semantics in this MLX version.
- **Unit tests present:** core unit and packaging tests pass. Integration tests requiring mlx-lm model loading are present but not executed in this environment.
- **Decompress-on-read path:** satisfies historical promotion-style gates on Llama-3.2-1B at 512-token context, but this path is not sufficient for `PROMOTED_EXPERIMENTAL`.
- **Fused attention path:** partial-tail re-encoding is eliminated: completed blocks stay compressed in persistent storage and the dense partial tail is attended via a separate dense-tail Metal kernel. Page-state merging and finalization currently use ordinary MLX operations. Native Apple Silicon results require attached benchmark artifacts.
- **Instance-level Llama adapter:** `TurboPolarLlamaAdapter` installs per-model, rolls back on failure, and prevents double install. Parameter-tree and state-dict preservation remain under validation.
- **Truthful memory accounting:** `CacheMemoryStats` separates logical payload, allocated capacity, dense tail, metadata, and dense equivalent; `measure_append_peak_memory()` probes the MLX allocator peak.
- **Paged storage:** persistent compressed blocks now use fixed-size pages (16 blocks/page), eliminating the quadratic historical copying that occurred with single-block-at-a-time array growth.
- **Promotion governance:** consolidated into `rfsn_v11/promotion/` with nested JSON constructors, tri-state git (`CLEAN`/`DIRTY`/`UNKNOWN`), kernel-source hashing, and a full orchestration script at `scripts/run_promotion_suite.py`. The gate is capped at `REVIEW_REQUIRED` until independent validation.
- **Benchmark suite:**
  - `benchmarks/run_dense_vs_turbopolar.py` — teacher-forced quality comparison.
  - `benchmarks/run_fused_forced_decode.py` — fused decode teacher-forced comparison with kernel stats.
  - `benchmarks/run_speed_matrix.py` — alternating-trial decode speed matrix with device-side token selection.
  - `benchmarks/run_memory_bench.py` — truthful memory accounting across sequence lengths.
  - `benchmarks/run_cartesian_int8_baseline.py` — grouped Cartesian int8 baseline.
- **Deterministic prompts:** `benchmarks/exact_token_fixtures.jsonl` provides exact-length fixtures across short/boundary/medium/long/stress categories.

## Supported configuration

See `docs/SUPPORTED_CONFIGURATION.md`. The narrow supported scope is:

- MLX + mlx-lm on Apple Silicon
- Llama-style GQA
- `head_dim == 128`
- `block_size == 64`
- `storage_mode == "kv_quant"`
- single-batch autoregressive decode
- full causal attention
- QJL disabled

## What is incomplete

- **Quantization tuning:** the default config does not yet hit the 1.85× logical KV compression target required by the promotion gate.
- **Real-model validation:** the new fused-decode and speed-matrix scripts are wired but have not been validated on a production-scale model.

## Promotion gates (must all be true)

Promotion to `PROMOTED_EXPERIMENTAL` requires:

1. All unit, kernel, and integration tests pass, including `tests.kernels.test_metal_strict` and `tests.kernels.test_fallback_injection`.
2. Fused forced-decode runs in `METAL_STRICT` mode with zero fallback calls and emits all required Metal dispatch fields.
3. Fused forced-decode mean logit cosine ≥ 0.995, p05 ≥ 0.990, minimum ≥ 0.975.
4. Top-5 overlap ≥ 0.95, top-10 overlap ≥ 0.97, argmax agreement ≥ 0.97.
5. Perplexity delta vs dense ≤ 0.02.
6. No NaNs or infinities; no catastrophic outlier prompt.
7. Logical KV compression ≥ 1.85×.
8. Measured persistent storage improves materially.
9. Peak device memory improves at 8192+ context.
10. No dense full-history cache remains resident.
11. No >3% regression at 4096+ context.
12. At least one long-context tier improves ≥ 5%.
13. 8192+ median steady-state decode ratio exceeds 1.03× across alternating trials.
14. TurboPolar beats or meaningfully differentiates from grouped Cartesian int8 K/V.
15. Exact model revision, software versions, prompt hashes, and config hashes recorded.
16. Promotion decision produced solely by `rfsn_v11/promotion/gate.py`.

**Promotion allowed:** NO.

## Build commands

```bash
make install-dev   # install in editable mode with dev+bench dependencies
make test          # run all tests
make compile       # compileall sanity check
make lint          # compileall
make smoke         # run smoke + README examples
make bench MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
make fused-bench MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
make speed-matrix MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
make memory-bench
make cartesian-bench MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
make promote MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
```
