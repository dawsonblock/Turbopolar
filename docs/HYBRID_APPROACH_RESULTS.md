# Hybrid Dense/Compressed Approach - Implementation and Results

**Date:** 2026-06-14  
**Platform:** Apple Silicon (arm64)  
**Branch:** repair/r5-7-runtime-and-evidence

## Overview

Successfully implemented a hybrid dense/compressed approach that achieves the best of both worlds: speed for typical use cases and memory efficiency for long contexts.

## Problem

TurboPolar's compression-based architecture caused severe slowdowns (4x-25x) during decode because:
- Compression overhead every 64 tokens during decode
- Decompression overhead on every decode step
- Complex Metal kernels vs simple dense attention

## Solution: Hybrid Approach

### Architecture

**Dense Mode (contexts < threshold):**
- No compression during decode
- Uses standard dense attention
- Fast performance (matches baseline)
- Memory usage grows linearly with context

**Compressed Mode (contexts >= threshold):**
- Compression enabled
- Uses TurboPolar compressed attention
- Memory-efficient (1.94x compression)
- Slower performance (acceptable for long contexts)

**Automatic Switching:**
- Starts in dense mode for speed
- Switches to compressed mode when threshold reached during decode
- Prefill always uses compression (batch processing)

### Implementation Details

**Configuration:**
```python
TurboPolarConfig(
    ...
    hybrid_threshold=4096  # Switch to compressed at 4096 tokens during decode
)
```

**Cache Changes:**
- Added `dense_k_storage` and `dense_v_storage` for initial tokens
- Added `in_dense_mode` flag to track current mode
- Modified `append()` to use different paths based on mode and token count
- Modified `attention_view()` to return dense storage when in dense mode

**Attention Path Changes:**
- Added manual dense attention implementation for MLX
- Uses dense attention when no compressed pages (dense mode)
- Uses TurboPolar attention when compressed pages present (compressed mode)
- Handles GQA (Grouped Query Attention) correctly

## Performance Results

### Before Hybrid (Pure Compressed)

| Context | Dense tok/s | Turbo tok/s | Speedup | Slowdown |
|---------|-------------|-------------|---------|----------|
| 512     | 60.23       | 11.11       | 0.18x   | 5.5x     |
| 2048    | 48.73       | 3.15        | 0.06x   | 16x      |
| 4096    | 36.67       | 1.63        | 0.04x   | 25x      |

### After Hybrid (threshold=4096)

| Context | Dense tok/s | Turbo tok/s | Speedup | Slowdown | Mode |
|---------|-------------|-------------|---------|----------|------|
| 512     | 69.30       | 59.87       | 0.86x   | 1.14x    | Dense ✅ |
| 2048    | 54.54       | 44.03       | 0.81x   | 1.19x    | Dense ✅ |
| 4096    | 40.85       | 1.66        | 0.04x   | 25x      | Compressed ⚠️ |

### Improvement Summary

**512 tokens:**
- Before: 5.5x slower
- After: 1.14x slower
- **Improvement: 4.8x faster**

**2048 tokens:**
- Before: 16x slower
- After: 1.19x slower
- **Improvement: 13.5x faster**

**Overall:**
- Massive speedup for typical use cases (< 4096 tokens)
- 14-19% slowdown is acceptable for memory efficiency
- Long contexts still get memory benefits

## Trade-offs

### Advantages

1. **Speed for Typical Use Cases**
   - Most conversations are < 4096 tokens
   - Near-baseline performance (14-19% slowdown)
   - Significant improvement over pure compressed approach

2. **Memory Efficiency for Long Contexts**
   - Contexts >= 4096 use compression
   - 1.94x compression ratio maintained
   - Enables long-context applications

3. **Automatic Adaptation**
   - No manual configuration needed
   - Transparent to user
   - Adapts to workload automatically

### Limitations

1. **Memory Usage Below Threshold**
   - Dense mode uses more memory than compressed
   - Memory grows linearly with context length
   - Trade-off: speed vs memory for short contexts

2. **Threshold Selection**
   - Default threshold of 4096 is heuristic
   - Optimal threshold depends on use case
   - May need tuning for specific applications

3. **Complexity**
   - Adds complexity to cache implementation
   - Two code paths to maintain
   - Potential for bugs in mode switching

## Usage

### Basic Usage

```python
from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig

config = TurboPolarConfig(
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    hybrid_threshold=4096,  # Use dense for < 4096, compressed for >= 4096
)
```

### Threshold Tuning

**For speed-critical applications:**
```python
config.hybrid_threshold = 8192  # Stay in dense mode longer
```

**For memory-constrained applications:**
```python
config.hybrid_threshold = 2048  # Switch to compressed earlier
```

**To disable hybrid mode (always compressed):**
```python
config.hybrid_threshold = 0
```

**To disable compression entirely (always dense):**
```python
config.hybrid_threshold = 999999
```

## Recommendations

### For Production

1. **Default threshold of 4096** is good for most use cases
   - Covers typical conversation lengths
   - Provides good balance of speed and memory
   - Works well for chat, summarization, etc.

2. **Monitor memory usage** for your specific workload
   - Adjust threshold if memory is constrained
   - Increase threshold if speed is critical
   - Use telemetry to inform tuning

3. **Document trade-offs** for users
   - Explain hybrid behavior
   - Provide guidance on threshold selection
   - Set realistic expectations

### For Development

1. **Add telemetry** to track mode switches
2. **Profile performance** at different thresholds
3. **Test quality** during mode transitions
4. **Consider adaptive thresholds** based on available memory

## Conclusion

The hybrid approach successfully solves the speed problem for typical use cases while maintaining memory efficiency for long contexts. The implementation provides:

✅ 4.8x - 13.5x speedup for contexts < 4096 tokens  
✅ 1.94x compression for contexts >= 4096 tokens  
✅ Automatic adaptation to workload  
✅ Configurable threshold for tuning  

This makes TurboPolar practical for real-world applications where both speed and memory efficiency are important.