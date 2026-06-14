# Phases 12-14: Infrastructure Complete, Benchmark Data Required

## Phase 12: Repair Memory Evidence

**Status:** Validation infrastructure complete, requires actual benchmark runs

The memory evidence module (`rfsn_v11/evidence/memory.py`) already has:
- MemoryReport dataclass with all required fields
- CacheMemoryStats for detailed memory accounting
- Validation infrastructure in the promotion gate

**What requires actual benchmark data:**
- Real memory measurements from native Apple Silicon runs
- Validation of peak memory improvements at 8192+ context
- Verification of storage memory reduction
- Comparison against dense baseline memory

**Infrastructure in place:**
- `benchmarks/run_memory_bench.py` - Memory benchmark script
- Promotion gate validates memory report fields
- Truthful memory accounting with logical payload vs allocated capacity separation

## Phase 13: Make the Cartesian Baseline Fair

**Status:** Validation infrastructure complete, requires actual benchmark runs

The Cartesian baseline module already has:
- Grouped Cartesian int8 K/V baseline implementation
- MLX-LM-compatible interface
- Comprehensive tests for Cartesian cache

**What requires actual benchmark data:**
- Real benchmark runs comparing TurboPolar vs Cartesian int8
- Validation that TurboPolar beats or meaningfully differentiates from Cartesian
- Fair comparison methodology (both candidates vs dense, not Turbo vs Cartesian)

**Infrastructure in place:**
- `benchmarks/run_cartesian_int8_baseline.py` - Cartesian baseline benchmark
- Promotion gate validates baseline comparison report
- Logical tail accounting correctly counts only valid tail tokens

## Phase 14: Release Gates

**Status:** All release gate infrastructure is in place and enforced

The release gates are fully implemented in `rfsn_v11/promotion/gate.py`:

**Portable Gates (Infrastructure Complete):**
1. ✅ All unit, kernel, and integration tests must pass
2. ✅ Fused forced-decode runs in METAL_STRICT mode with zero fallbacks
3. ✅ Fused forced-decode quality thresholds (cosine, overlap, perplexity)
4. ✅ No NaNs or infinities
5. ✅ Proper provenance tracking (git, model, config hashes)

**Native Gates (Infrastructure Complete, Requires Apple Silicon Data):**
6. ⏳ Native Metal tests pass on Apple Silicon
7. ⏳ Logical KV compression ≥ 1.85× (requires benchmark data)
8. ⏳ Measured persistent storage improves (requires benchmark data)
9. ⏳ Peak device memory improves at 8192+ (requires benchmark data)
10. ⏳ No dense full-history cache remains resident (requires benchmark data)

**Evidence Gates (Infrastructure Complete, Requires Benchmark Data):**
11. ⏳ No >3% regression at 4096+ context (requires speed benchmark data)
12. ⏳ At least one long-context tier improves ≥ 5% (requires speed data)
13. ⏳ 8192+ median steady-state decode ratio exceeds 1.03× (requires speed data)

**Completeness Gates (Infrastructure Complete):**
14. ✅ Exact model revision, software versions, prompt hashes, config hashes recorded
15. ✅ Promotion decision produced solely by `rfsn_v11/promotion/gate.py`

**Locked Review Gate (Infrastructure Complete):**
16. ✅ PROMOTION_LOCKED=True prevents auto-promotion, requires independent review
17. ✅ Evidence kind field distinguishes experimental vs synthetic dry-run

## Summary

**Infrastructure Status:** ✅ COMPLETE
All validation infrastructure, schemas, tests, and gate logic are in place.

**Benchmark Data Status:** ⏳ PENDING
Actual benchmark runs on native Apple Silicon are required to populate:
- Memory evidence (Phase 12)
- Cartesian baseline comparisons (Phase 13)
- Speed evidence at long contexts (Phase 14)

**Next Steps:**
1. Run benchmarks on Apple Silicon hardware
2. Validate all evidence passes the promotion gate
3. Independent review of promotion decision
4. Set PROMOTION_LOCKED=False after independent validation

## Test Coverage

All infrastructure has comprehensive test coverage:
- Phase 5: 10 speed schema tests
- Phase 6: 13 promotion tests (all passing)
- Phase 7: 12 fixture schema tests
- Phase 8: 11 trace invariant tests
- Phase 10: 9 provenance validation tests
- Phase 11: 11 platform validation tests

**Total: 66 new tests added, all passing**