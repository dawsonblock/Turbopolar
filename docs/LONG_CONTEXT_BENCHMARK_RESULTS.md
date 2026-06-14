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

### ❌ Concerning Results

**Peak Device Memory:**
- 512 tokens: 0.63x (37% reduction) ✅
- 2048 tokens: 0.73x (27% reduction) ✅
- 4096 tokens: 0.98x (2% reduction) ⚠️ Near parity
- 8192 tokens: 1.12x (12% increase) ❌ **REGRESSION**
- 16384 tokens: 1.14x (14% increase) ❌ **REGRESSION**

**Peak Memory Regression at Long Contexts:**
- At 8192+, TurboPolar uses MORE peak memory than dense baseline
- This is unexpected and indicates:
  - Possible memory overhead in compressed page system at very long contexts
  - Potential memory leak or inefficiency in current implementation
  - May need investigation into memory allocation patterns
  - Could be related to page capacity or block management

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
5. ✅ All infrastructure validation (66 tests passing)

### Gates That Require Attention ⚠️

1. ⚠️ Peak memory improvement at 8192+ (REGRESSION detected)
2. ⚠️ No >3% regression at 4096+ (needs speed data)
3. ⚠️ Long-context improvement ≥5% (needs speed data)
4. ⚠️ 8192+ median ratio ≥1.03x (needs speed data)

### Gates Not Yet Tested

1. ⏳ Fused decode metrics at long contexts
2. ⏳ Cartesian baseline comparison
3. ⏳ Complete speed matrix at required contexts

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
**❌ Peak Memory Regression Detected:** Needs investigation before promotion  
**⏳ Speed Validation Incomplete:** Requires optimized benchmarking approach  

The peak memory regression at 8192+ contexts is a significant finding that must be addressed before promotion. This indicates the current implementation may have memory efficiency issues at very long contexts that need investigation and resolution.