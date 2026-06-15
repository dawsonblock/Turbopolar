# Hybrid Cache Correctness Issues and Fixes

**Date:** 2026-06-14  
**Status:** Hybrid mode disabled by default (hybrid_threshold=0) due to critical P0 bugs

## Critical P0 Bugs Identified

### 1. Prompt History Loss in Dense Mode

**Bug:** Prefill compresses to paged storage, but decode uses empty dense cache, ignoring the actual prompt history.

**Root Cause:**
- `append_many()` (prefill) compresses into regular paged K/V storage
- Does NOT populate `dense_k_storage` or `dense_v_storage`
- Leaves `in_dense_mode = True`
- First decode enters dense path with empty dense buffers
- Attention sees only dense buffer, ignores compressed pages

**Impact:**
- For 512-token prompt: dense positions 0-511 are zero, position 512 has new token
- Model attends to zeros instead of actual prompt history
- Complete history loss error

**Fix Required:**
Either:
- Design A: Prefill directly into dense storage when in dense mode
- Design B: Always use compressed storage, use temporary decoded view for short contexts

### 2. Sequence Length Corruption at Transition

**Bug:** When switching from dense to compressed mode, `actual_seq_len` is incremented twice.

**Root Cause:**
```python
before transition:
actual_seq_len = 4,096

_switch_to_compressed_mode():
    calls _append_prefill(dense_buffer)
    _append_prefill increments actual_seq_len again → becomes 8,192

append current token:
actual_seq_len becomes 8,193
```

**Impact:**
- Cache offsets are wrong
- Mask construction is wrong
- Page/tail counts are wrong
- Trace topology is corrupted
- Position identities are wrong

**Fix Required:**
- Don't increment actual_seq_len during transition
- Or reset actual_seq_len before compressing dense buffer

### 3. Reset Does Not Clear Hybrid State

**Bug:** Runtime reset does not clear:
- `dense_k_storage`
- `dense_v_storage`
- `in_dense_mode`

**Impact:**
- Cache reused after reset retains stale hybrid buffers
- May remain in wrong operating mode
- Cross-contamination between different fixtures

**Fix Required:**
```python
def reset(self):
    self.dense_k_storage = None
    self.dense_v_storage = None
    self.in_dense_mode = True  # Reset to initial state
    self.actual_seq_len = 0
    # Clear compressed storage
    self.k_storage.clear()
    self.v_storage.clear()
    self.partial_length = 0
```

### 4. Hybrid Memory Omitted from Accounting

**Bug:** `get_memory_stats()` reports only compressed cache and partial tail, not the dense hybrid arrays.

**Impact:**
- Logical cache totals are wrong
- Allocated persistent totals are wrong
- Compression ratios are wrong
- Whole-model cache comparisons are wrong
- Promotion memory evidence is invalid

**Fix Required:**
Add to memory stats:
```python
{
    "dense_hybrid_logical_bytes": ...,
    "dense_hybrid_allocated_bytes": ...,
    "compressed_logical_bytes": ...,
    "compressed_allocated_bytes": ...,
}
```

### 5. Dense-History Audit Misses Hybrid History

**Bug:** Residency audit reports no retained dense historical cache, doesn't inspect dense hybrid arrays.

**Impact:**
- System can hold thousands of dense tokens while reporting:
  - `retained_dense_k_history = false`
  - `retained_dense_v_history = false`
- Promotion gate's dense-history requirement is bypassed

**Fix Required:**
Audit must enumerate:
- `dense_k_storage`
- `dense_v_storage`
- Compressed K/V pages
- Partial K/V tail
- Any dense tensor longer than tail capacity

### 6. No Hybrid-Specific Tests

**Bug:** No tests for:
- `hybrid_threshold`
- `in_dense_mode`
- Dense-to-compressed transition
- Hybrid reset
- Hybrid memory accounting

**Impact:**
- Bugs passed undetected through large test suite
- No validation of boundary conditions (511, 512, 2048, 4095, 4096, 4097)

**Fix Required:**
Add tests for each boundary case verifying:
- Exact K/V history
- Exact sequence length
- Correct mode
- Storage contents
- Logit agreement with dense reference
- Memory totals
- Reset behavior
- GQA behavior
- Zero fallback

### 7. GQA Behavior Unproven in Dense Mode

**Bug:** Dense fast path uses `mx.fast.scaled_dot_product_attention` without testing GQA mapping.

**Impact:**
- May not handle 8 query heads / 2 KV heads correctly
- Quality regression possible

**Fix Required:**
Add explicit tests for:
- 8 query heads, 2 KV heads
- 4:1 GQA ratio
- Multiple tail lengths
- Strict dense comparison

## Current Status

**Default Configuration:**
```python
hybrid_threshold = 0  # Disabled - always use compressed mode
```

**Impact of Disable:**
- Safe compressed-only execution path
- All hybrid bugs avoided
- Speed results from hybrid mode are invalid
- Must fix all P0 bugs before re-enabling

## Required Fixes Before Re-enabling

### P0 - Fix Runtime Correctness

1. **Choose a valid hybrid design:**
   - Design A: Truly dense below threshold
   - Design B: Compressed-only persistent storage (recommended)

2. **Implement chosen design correctly**

3. **Add transition boundary tests:**
   - 511, 512, 2048, 4095, 4096, 4097 tokens
   - Verify history, sequence length, mode, memory

4. **Fix reset to clear all hybrid state**

5. **Add dense storage to memory accounting**

6. **Add dense storage to residency audit**

7. **Make strict mode reject unapproved dense MLX attention**

### P1 - Repository Consistency

8. Remove duplicate Metal backup source
9. Migrate fixtures to canonical schema
10. Fix 2 failing prompt-fixture tests
11. Activate vocabulary validation

### P1 - Governance

12. Integrate canonical speed validator into gate
13. Integrate platform validation into evidence capture
14. Unify ProvenanceEvidence and BenchmarkProvenance
15. Fix Git hash-length assumptions

## Recommended Design B: Compressed-Only Persistent Storage

**Why Design B is better:**
- Avoids duplicate persistent representations
- Simpler state machine
- Consistent with project's compression purpose
- No transition complexity
- Clearer execution contract

**Implementation:**
1. Always store prefill in compressed pages
2. For short contexts (< threshold), use temporary decoded view
3. Do not maintain second persistent full-history dense cache
4. No mode switching - always compressed storage
5. Dense path only for temporary attention computation

This avoids all the state-transition bugs while still providing speed benefits through optimized attention computation.