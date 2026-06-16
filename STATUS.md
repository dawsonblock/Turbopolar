# TurboPolar Development Status

**Branch:** `repair/r5-7-runtime-and-evidence`  
**Version:** `0.4.0.dev0`  
**Architecture:** Adaptive Tiered KV Cache v1  
**Last updated:** 2026-06-15 (REPAIR R5-7 Rev16)

---

## Promotion verdict

**State:** `NO-GO` — Promotion is hard-locked at `REVIEW_REQUIRED`.

`PROMOTED_EXPERIMENTAL` is locked until the full evidence matrix is independently
validated on native Apple Silicon with the required model and pinned software stack.

---

## What is fixed (this branch)

| Item | Change |
|------|--------|
| P0-1 | Boundary-crossing prefill correctly increments `actual_seq_len` |
| P0-2 | Speed trials use 0-based indices; real `execution_order` preserved |
| P0-3 | `dense_to_turbo_peak_ratio` convention unified: >1 = Turbo saves memory |
| P1-1 | Native test modules guarded with `pytest.importorskip("mlx")` |
| P1-3 | Memory bench runs dense and Turbo in isolated subprocesses |
| P1-5 | Memory accounting uses arithmetic, not device slices |
| P1-2 | `compression_time_ns` and `bytes_written` now instrumented in all encode paths |
| P1-9 | Config `__post_init__` allows 4/8-bit level-1 angles; validator locks to 8-bit |
| P1-10 | `make_turbo_caches` clones the exact immutable config via `dataclasses.replace` |
| P1-11 | `validate_trace_topology` exposes stats at top level for backward compatibility |
| P1-12 | `scripts/` is now a proper Python package; `_build_provenance` is importable |
| Arch | Circular dense hot window replaces shift-based tail (Milestone 1) |
| Arch | All compressed pages concatenated into one arena; Metal called ONCE per decode step |
| Arch | 256-entry constant-memory cos/sin LUT replaces per-token transcendental calls for 8-bit angles |

---

## What is still failing

| Gate | Measurement | Threshold | Status |
|------|------------|-----------|--------|
| Memory peak | 1.12–1.14× at 8K+ | ≥ 1.20× | **FAIL** |
| Perplexity delta | 0.0326 | ≤ 0.02 | **FAIL** |
| Speed at 4K+ | 0.04–0.06× dense | ≥ 1.00× (no regression) | **FAIL** |
| Raw evidence | Absent from archive | Required | **FAIL** |

The speed failure is structural: even with the arena dispatch and LUT, the polar K
representation still requires decoding inside the kernel at every token. A full
Cartesian-int8 K redesign or the two-stage kernel architecture is needed before
the speed gate can be revisited.

---

## Supported configuration

The single supported promotion configuration is:

- MLX ≥ 0.31.2 + mlx-lm ≥ 0.19.0 on Apple Silicon (M1/M2/M3/M4)
- Llama-style GQA with `head_dim=64` or `128`, `block_size=64`
- `k_angle_bits_level1=8`, `k_angle_bits_deep=8`
- `split_dim=0` (all 64 pairs use deep bucket)
- `storage_mode="kv_quant"`, `v_bits=8`, `group_size=32`
- Single batch (`batch_size=1`), full causal attention, `mask=None`
- QJL disabled

Anything outside this scope raises at construction or validation time.

---

## Required evidence for promotion

All of the following must be present, hash-verified, and passing:

1. Raw speed matrix JSON — 5 paired alternating trials at 512/2048/4096/8192/16384 tokens
2. Raw memory matrix JSON — isolated subprocess measurements at 8 context lengths
3. Teacher-forced fused decode report — per-prompt quality metrics
4. JUnit XML — full native test suite
5. Promotion evidence JSON — gate evaluation record
6. Provenance bundle — git commit, MLX version, model revision, fixture hashes, Metal source hashes

No hand-authored numbers are accepted. All displayed ratios and pass/fail verdicts
must be recomputable from raw byte fields in the attached JSON artifacts.

---

## Promotion gates (must all pass)

1. All unit, kernel, and integration tests pass including `test_metal_strict` and `test_fallback_injection`
2. Fused decode runs `METAL_STRICT` with zero fallbacks
3. Mean logit cosine ≥ 0.995, p05 ≥ 0.990, minimum ≥ 0.975
4. Top-5 overlap ≥ 0.95, top-10 overlap ≥ 0.97, argmax agreement ≥ 0.97
5. Perplexity delta ≤ 0.02
6. No NaNs or infinities; no catastrophic outlier prompt
7. Logical KV compression ≥ 1.85×
8. Allocated storage compression material improvement
9. `dense_to_turbo_peak_ratio` ≥ 1.20 at 8192+ context
10. No dense full-history cache residency
11. No speed regression >3% at 4096+ context
12. At least one long-context tier improves ≥ 5%
13. 8192+ median decode ratio > 1.03× across alternating trials
14. TurboPolar outperforms or materially differentiates from Cartesian int8 baseline
15. Exact model revision, software versions, and prompt hashes recorded

---

## Running benchmarks

Requires a model with `head_dim=128` (Llama 3.1/3.2 8B or 70B family).

```bash
make bench MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make fused-bench MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make speed-matrix MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make memory-bench MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make cartesian-bench MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make promote MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
make evidence-markdown
```

Note: `Llama-3.2-1B` (`head_dim=64`) is supported as of v0.4.0.dev0.
