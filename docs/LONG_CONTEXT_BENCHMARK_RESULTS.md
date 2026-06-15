# Long Context Benchmark Results

**Date:** 2026-06-14  
**Platform:** Apple Silicon (arm64)  
**macOS Version:** 26.2  
**Model:** mlx-community/Llama-3.2-3B-Instruct-4bit  
**Branch:** repair/r5-7-runtime-and-evidence

## Long Context Memory Benchmark Results

**Command:** `python benchmarks/run_memory_bench.py --lengths 8192 16384`

| Context | Logical KV Ratio | Peak Memory Ratio | Storage Ratio | Status |
|---------|------------------|-------------------|---------------|---------|
| 8192    | 1.94x            | 1.12x             | 1.91x         | ❌ Peak memory regression |
| 16384   | 1.94x            | 1.14x             | 1.92x         | ❌ Peak memory regression |

## Analysis

### ✅ Positive Results

**Logical KV Compression:**
- Target: ≥1.85x
- Achieved: 1.94x across all contexts (512, 2048, 4096, 8192, 16384)
- **Status:** ✅ **CONSISTENTLY EXCEEDS TARGET**

**Storage Memory:**
- Storage ratio improves with context length (0.86x → 1.92x)
- **Status:** ✅ **IMPROVES WITH CONTEXT**

### ❌ Hard Gate Failures

**Peak Device Memory (Gate 9: must improve at 8192+):**
- 512 tokens: 0.63x (37% reduction) ✅
- 2048 tokens: 0.73x (27% reduction) ✅
- 4096 tokens: 0.98x (2% reduction) ⚠️ Near parity
- 8192 tokens: 1.12x (12% increase) ❌ **FAILS GATE 9**
- 16384 tokens: 1.14x (14% increase) ❌ **FAILS GATE 9**

**Peak Memory Regression at Long Contexts:**
- At 8192+, TurboPolar uses MORE peak memory than dense baseline.
- This is a hard quantitative failure of promotion gate 9, not a minor regression.
- Possible causes:
  - Memory overhead in compressed page system at very long contexts
  - Potential memory leak or inefficiency in current implementation
  - Page capacity or block management overhead

## Speed Benchmark Status

**Attempted:** Speed matrix benchmark at 2048, 4096, 8192, 16384 with 5 trials each
**Result:** Timeout - benchmarks taking too long (>5 minutes for 3B model)

**Alternative Approaches:**
- Use smaller model (1B instead of 3B) for faster iteration
- Reduce trial count from 5 to 2 for initial validation
- Test individual contexts separately
- Use non-quantized model if available

## Teacher-Forced Quality at Long Contexts

**Attempted:** Teacher-forced benchmark at 512, 2048, 4096 contexts
**Result:** Monitoring issue, benchmark running in background

**Expected Results:** Based on short context results:
- Quality metrics should remain excellent (cosine >0.995)
- Quality is typically consistent across context lengths

## Promotion Gate Implications

### Gates That Pass ✅

1. ✅ Logical KV compression ≥1.85x (achieved 1.94x)
2. ✅ Mean cosine ≥0.995 (achieved 0.9997)
3. ✅ Top-5 overlap ≥0.95 (achieved 0.983)
4. ✅ Storage memory improvement
5. ✅ All infrastructure validation (~380 tests passing)

### Hard Gate Failures ❌

1. ❌ Gate 5 — Perplexity delta ≤0.02 (achieved 0.0326, 63 % overshoot)
2. ❌ Gate 9 — Peak memory improvement at 8192+ (1.12x–1.14x regression)
3. ❌ Gates 11–13 — Speed non-regression / improvement (0.18x–0.04x slowdown, timeouts at 8192+)

### Gates Not Yet Tested

1. ⏳ Fused decode metrics at long contexts
2. ⏳ Cartesian baseline comparison
3. ⏳ Complete speed matrix at required contexts (blocked by timeouts)

## Recommendations

### Immediate Actions

1. **Investigate peak memory regression at 8192+ contexts**
   - Profile memory allocation patterns
   - Check for memory leaks in compressed page system
   - Review page capacity and block management
   - Consider if overhead is expected or a bug

2. **Complete speed benchmarks with optimized approach**
   - Use 1B model instead of 3B for faster iteration
   - Test contexts separately rather than in one long run
   - Start with 4096 context to validate speedup before going to 8192+

3. **Complete teacher-forced quality at long contexts**
   - Validate quality metrics remain excellent at 2048, 4096
   - Ensure quality doesn't degrade with context length

### For Production

1. **Use non-quantized model** (if available) for cleaner results
2. **Investigate and fix peak memory regression** before promotion
3. **Complete full evidence package** with all required contexts
4. **Independent review** required before PROMOTION_LOCKED=False

## Conclusion

**✅ Infrastructure Complete:** All validation infrastructure tested and working  
**✅ Quality Metrics Excellent:** Cosine and overlap metrics exceed targets  
**✅ Logical Compression Target Achieved:** 1.94x vs 1.85x target  
**❌ Multiple Hard Gate Failures:** Peak memory regression (gate 9), perplexity overshoot (gate 5), and severe speed slowdowns (gates 11–13) quantitatively block promotion  
**⏳ Speed Validation Incomplete:** Timeouts at 8192+ prevent even gathering required data  

TurboPolar is currently failing at least three hard promotion gates with no known fix path other than fundamental architectural redesign or threshold relaxation. Promotion is quantitatively blocked, not merely "not yet ready."