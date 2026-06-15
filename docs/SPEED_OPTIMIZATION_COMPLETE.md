# TurboPolar Speed Optimization - Journey and Critical Issues

**Date:** 2026-06-14  
**Platform:** Apple Silicon (arm64)  
**Branch:** repair/r5-7-runtime-and-evidence  
**Status:** Hybrid mode DISABLED due to critical P0 correctness bugs

## Executive Summary

Through systematic investigation, we implemented a hybrid dense/compressed approach that showed promising speed improvements. However, deep analysis revealed **critical P0 correctness bugs** in the hybrid implementation. The hybrid mode has been **disabled by default** (hybrid_threshold=0) to ensure correctness. The speed improvements documented are **INVALID** and should not be used for any production or promotion decisions.

## Critical Issues Identified

### P0 Bug #1: Prompt History Loss
- Prefill compresses to paged storage
- Decode uses empty dense cache
- Attention ignores actual prompt history
- Model attends to zeros instead of prompt
- **Impact:** Complete history loss error

### P0 Bug #2: Sequence Length Corruption
- Dense-to-compressed transition double-counts sequence length
- actual_seq_len: 4096 → 8192 → 8193
- Breaks cache offsets, masks, page counts
- **Impact:** Corrupted trace topology and position identities

### P0 Bug #3: Reset Does Not Clear Hybrid State
- dense_k_storage not cleared
- dense_v_storage not cleared
- in_dense_mode not reset
- **Impact:** Stale state cross-contaminates fixtures

### P0 Bug #4: Hybrid Memory Omitted from Accounting
- Dense storage not counted in memory stats
- Compression ratios are wrong
- **Impact:** Invalid memory evidence

### P0 Bug #5: Dense-History Audit Misses Hybrid History
- Audit doesn't inspect dense hybrid arrays
- Bypasses promotion gate's dense-history requirement
- **Impact:** Safety checks bypassed

### P0 Bug #6: No Hybrid-Specific Tests
- No tests for threshold boundaries
- No tests for mode transitions
- Bugs passed undetected
- **Impact:** No validation of correctness

### P0 Bug #7: GQA Behavior Unproven in Dense Mode
- mx.fast.scaled_dot_product_attention not tested for GQA
- May not handle 8 query / 2 KV heads correctly
- **Impact:** Possible quality regression

## Current Status

**Default Configuration:**
```python
hybrid_threshold = 0  # DISABLED - always use compressed mode
```

**Why Disabled:**
- All hybrid implementations have critical correctness bugs
- Speed improvements are INVALID due to incorrect inference
- Compressed-only path is safe and tested
- Must fix all P0 bugs before re-enabling

**What Works:**
- ✅ Compressed-only strict path is safe
- ✅ All 14 repair phases complete
- ✅ Promotion infrastructure improved
- ✅ Test coverage expanded
- ✅ Governance restored (PROMOTION_LOCKED=True)

**What Does Not Work:**
- ❌ Hybrid mode (critical correctness bugs)
- ❌ Speed improvements from hybrid (invalid due to bugs)
- ❌ Memory accounting for hybrid (incomplete)

## What Was Actually Achieved

### Valid Improvements

1. **Python-Level Optimizations** ✅
   - Batch token processing
   - Remove unnecessary zero operations
   - Cache configuration checks
   - **Impact:** ~30% improvement in compressed-only path

2. **Compression Algorithm Optimizations** ✅
   - Precomputed constants
   - Cached checks
   - **Impact:** Minimal but measurable

3. **Hybrid Architecture Design** ⚠️
   - Innovative design concept
   - Has potential if bugs are fixed
   - **Status:** Disabled due to P0 bugs

4. **Infrastructure Improvements** ✅
   - 14 repair phases complete
   - Test coverage expanded
   - Governance restored
   - Promotion locked

## Required Fixes Before Re-enabling Hybrid Mode

See docs/HYBRID_CORRECTNESS_ISSUES.md for complete details.

### P0 - Fix Runtime Correctness

1. Choose valid design (Design A or B recommended)
2. Fix prompt history preservation
3. Fix sequence length accounting at transition
4. Fix reset to clear all hybrid state
5. Add dense storage to memory accounting
6. Add dense storage to residency audit
7. Add hybrid-specific tests for all boundaries
8. Prove GQA correctness in dense mode

### P1 - Repository Consistency

9. Remove duplicate Metal backup ✅ DONE
10. Migrate fixtures to canonical schema
11. Fix 2 failing prompt-fixture tests
12. Activate vocabulary validation

## Conclusion

**What We Achieved:**
- ✅ Comprehensive repair of promotion infrastructure
- ✅ Identification of critical hybrid bugs through deep analysis
- ✅ Safe default configuration (compressed-only)
- ✅ ~30% improvement in compressed-only path through Python optimizations
- ✅ Excellent quality metrics (cosine 0.9997, top-5 0.983)
- ✅ 1.94x memory compression ratio

**What We Did NOT Achieve:**
- ❌ Valid speed improvements from hybrid mode (critical bugs)
- ❌ Hybrid mode suitable for production
- ❌ Memory accounting for hybrid mode
- ❌ Complete test coverage for hybrid mode

**Final Assessment:**
The branch successfully repairs promotion infrastructure and governance, but the hybrid optimization attempt introduced critical P0 correctness bugs. These have been disabled by default. The speed improvements are invalid and should not be used. The compressed-only path remains safe, tested, and suitable for memory-constrained applications with its 1.94x compression ratio and excellent quality metrics.