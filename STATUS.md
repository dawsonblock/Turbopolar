# TurboPolar Development Status

**Branch:** `main`  
**Version:** `0.2.0.dev0`  
**Last updated:** 2026-06-11

## Status summary

**Promotion state:** `PROMOTED_EXPERIMENTAL` is **locked**.

TurboPolar remains an end-to-end research alpha. Promotion is blocked until reproducible artifacts independently prove:

1. **Correctness:** fused compressed attention produces acceptably close logits to dense attention during actual autoregressive decode.
2. **Memory:** measured device memory is reduced, not merely logical payload bytes.
3. **Performance:** steady-state decode is faster or at least non-regressive at long context under a sound benchmark.

No promotion claim may be made by any component other than the single promotion gate in `rfsn_v11/promotion/gate.py`.

## What works today

- **Metal kernels on M2:** custom QK and attention kernels compile and run natively on Apple Silicon under MLX 0.31.2.
- **MLX dispatch is correct:** we probe and adapt to the `grid == total_threads` semantics in this MLX version.
- **Unit tests pass:** 71/71 tests green.
- **Decompress-on-read path:** satisfies historical promotion-style gates on Llama-3.2-1B at 512-token context, but this path is not sufficient for `PROMOTED_EXPERIMENTAL`.
- **Fused attention path:** quality matches dense (>0.999 cosine). Per-step partial-tail re-encoding is eliminated: completed blocks stay compressed in persistent storage and the dense partial tail is attended separately in a single fused kernel.

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

- **Fused decode quality validation:** forced-autoregressive-decode quality matrix across 512–16384 tokens is not yet complete.
- **Long-context speed validation:** sound alternating-trial speed benchmark at 4096+ tokens is not yet complete.
- **Actual peak-memory validation:** device peak-memory measurement via MLX Metal APIs is not yet complete.
- **Promotion governance:** consolidated into `rfsn_v11/promotion/` (`schema.py`, `gate.py`, `provenance.py`, `cli.py`). Only the central gate may produce promotion decisions; missing evidence now fails.
- **Benchmark provenance:** `BenchmarkProvenance` captures software versions, prompt/config hashes, dirty-tree state, and writes immutable non-overwriting artifacts.
- **Cartesian int8 baseline:** fair competitor baseline is not yet implemented.
- **Persistent compressed block storage:** implemented in `rfsn_v11/generation/storage.py` with capacity-doubling growth for `PolarKBlockStorage` and `QuantVBlockStorage`.

## Promotion gates (must all be true)

Promotion to `PROMOTED_EXPERIMENTAL` requires:

1. All unit, kernel, and integration tests pass.
2. Fused forced-decode mean logit cosine ≥ 0.995, p05 ≥ 0.990, minimum ≥ 0.975.
3. Top-5 overlap ≥ 0.95, top-10 overlap ≥ 0.97, argmax agreement ≥ 0.97.
4. Perplexity delta vs dense ≤ 0.02.
5. No NaNs or infinities; no catastrophic outlier prompt.
6. Logical KV compression ≥ 1.85×.
7. Measured persistent storage improves materially.
8. Peak device memory improves at 8192+ context.
9. No dense full-history cache remains resident.
10. No >3% regression at 4096+ context.
11. At least one long-context tier improves ≥ 5%.
12. 8192+ median steady-state decode ratio exceeds 1.03× across alternating trials.
13. TurboPolar beats or meaningfully differentiates from grouped Cartesian int8 K/V.
14. Exact model revision, software versions, prompt hashes, and config hashes recorded.
15. Promotion decision produced solely by `rfsn_v11/promotion/gate.py`.

**Promotion allowed:** NO.

## Build commands

```bash
make install-dev   # install in editable mode
make test          # run all tests
make compile       # compileall sanity check
make lint          # ruff + mypy + compileall
make smoke         # run smoke + README examples
make bench MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
make fast-bench MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
```
