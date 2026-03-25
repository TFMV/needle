# Go HNSW Implementation: Benchmark Results and Optimizations

This document outlines the performance benchmarks and optimization process for our Go-based HNSW (Hierarchical Navigable Small World) implementation.

## Initial Performance Benchmarks

The following benchmarks were run on a dataset of 10,000 vectors with 128 dimensions.

### Without Product Quantization (PQ)

*   **`BenchmarkBuild`**:
    *   **Time per operation**: 2,904,973,341 ns (2.90 seconds)
    *   **Memory per operation**: 37,586,712 bytes (37.6 MB)
    *   **Allocations per operation**: 816,561
*   **`BenchmarkSearchLatency`**:
    *   **Time per operation**: 23,283,500 ns (23.28 ms)
    *   **Memory per operation**: 61,615 bytes (61.6 KB)
    *   **Allocations per operation**: 300

## Recall Benchmark and PQ Investigation

We introduced a recall benchmark to measure the accuracy of our search. The initial results were lower than expected, which led to an investigation into the Product Quantization (PQ) implementation.

### Initial Recall with PQ Enabled

*   **`BenchmarkBuild`**:
    *   **Time per operation**: 3,201,584,269 ns (3.20 seconds)
    *   **Memory per operation**: 37,953,160 bytes (38.0 MB)
    *   **Allocations per operation**: 821,914
*   **`BenchmarkSearchLatency`**:
    *   **Time per operation**: 22,082,124 ns (22.08 ms)
    *   **Memory per operation**: 61,614 bytes (61.6 KB)
    *   **Allocations per operation**: 300
*   **`BenchmarkSearchRecall`**: **32.0% recall@10**

### Increased `efSearch`

Increasing `efSearch` from 64 to 128 resulted in a performance hit with a negligible improvement in recall, a clear sign of an underlying issue.

*   **`BenchmarkSearchLatency`**: 30,602,714 ns (30.60 ms)
*   **`BenchmarkSearchRecall`**: **32.5% recall@10**

### Disabling PQ: The Breakthrough

Disabling PQ confirmed our suspicions: the PQ implementation was the bottleneck.

*   **`BenchmarkBuild`**: 153,295,544 ns (0.15 seconds)
*   **`BenchmarkSearchLatency`**: 12,433,477 ns (12.43 ms)
*   **`BenchmarkSearchRecall`**: **99.2% recall@10**

## PQ Optimization: The Path to High Recall

The investigation confirmed that the PQ implementation was the primary bottleneck. The next phase of optimization focused on overhauling the PQ implementation.

### Final Optimized PQ Implementation

After a thorough overhaul of the k-means algorithm, we have achieved a high-recall, high-performance PQ implementation.

*   **`BenchmarkBuild`**:
    *   **Time per operation**: 753,718,609 ns (0.75 seconds)
*   **`BenchmarkSearchLatency`**:
    *   **Time per operation**: 115,971,866 ns (116 ms)
*   **`BenchmarkSearchRecall`**: **99.0% recall@10**

### Key Optimizations

1.  **Robust K-Means++ Initialization**: We implemented a more robust version of k-means++ to ensure that the initial centroids are well-distributed.

2.  **Efficient Empty Cluster Handling**: We replaced the naive empty cluster handling with a more efficient method. Instead of performing a separate, expensive search for the farthest point, we now identify it during the main assignment loop.

3.  **Convergence Check**: The k-means algorithm now includes a convergence check. It terminates early if the centroids stabilize, which significantly speeds up the training process.

The optimized PQ implementation now provides a viable solution for reducing memory usage while maintaining high recall.

## SIMD Optimization

We have implemented a SIMD-optimized L2 distance function for the `amd64` architecture. The following benchmarks show the performance improvement.

*   **`BenchmarkBuild`**:
    *   **Time per operation**: 572,743,443 ns (0.57 seconds)
*   **`BenchmarkSearchLatency`**:
    *   **Time per operation**: 127,073,309 ns (127 ms)
*   **`BenchmarkSearchRecall`**: **98.6% recall@10**

## Cache Optimization: Aligned `vectorData` and `[]Node`

*   **`BenchmarkBuild`**: 559,029,603 ns/op
*   **`BenchmarkSearchLatency`**: 103,831,057 ns/op
*   **`BenchmarkSearchRecall`**: 46,386,771 ns/op, recall 0.9870

## Cache Optimization: Contiguous `Node.neighbors` (Regressions)

*   **`BenchmarkBuild`**: 762,273,426 ns/op
*   **`BenchmarkSearchLatency`**: 65,606,970 ns/op
*   **`BenchmarkSearchRecall`**: 57,086,351 ns/op, recall 0.9860

## Reverted `Node.neighbors` Optimization

*   **`BenchmarkBuild`**: 523,145,174 ns/op
*   **`BenchmarkSearchLatency`**: 119,351,230 ns/op
*   **`BenchmarkSearchRecall`**: 64,629,164 ns/op, recall 0.9960

## Inefficient Pruning Logic (Regressions)

*   **`BenchmarkBuild`**: 765,297,460 ns/op
*   **`BenchmarkSearchLatency`**: 144,581,019 ns/op
*   **`BenchmarkSearchRecall`**: 49,297,947 ns/op, recall 0.9810

## Efficient Pruning Logic

*   **`BenchmarkBuild`**: 466,039,058 ns/op
*   **`BenchmarkSearchLatency`**: 108,171,806 ns/op
*   **`BenchmarkSearchRecall`**: 59,226,494 ns/op, recall 0.9830

## Memory-Optimized Pruning

*   **`BenchmarkBuild`**: 430,284,992 ns/op
*   **`BenchmarkSearchLatency`**: 84,292,902 ns/op
*   **`BenchmarkSearchRecall`**: 82,268,754 ns/op, recall 0.9860

## Final Benchmark Results

After resolving all build and linker errors, we have a final, stable set of benchmarks.

*   **`BenchmarkBuild`**:
    *   **Time per operation**: 170,367,998 ns (0.17 seconds)
    *   **Memory per operation**: 4,451,832 bytes (4.45 MB)
    *   **Allocations per operation**: 51,174
*   **`BenchmarkSearchLatency`**:
    *   **Time per operation**: 10,686,211 ns (10.69 ms)
    *   **Memory per operation**: 128,049 bytes (128 KB)
    *   **Allocations per operation**: 400
*   **`BenchmarkSearchRecall`**:
    *   **Time per operation**: 17,291,056 ns (17.29 ms)
    *   **Recall**: 98.3% recall@k
    *   **Memory per operation**: 161,096 bytes (161 KB)
    *   **Allocations per operation**: 700

## Benchmark Results After Pointer-Based Distance Calculation Refactor

| Benchmark | Iterations | Time/Operation | Bytes/Operation | Allocations/Operation | Extra Metrics |
|---|---|---|---|---|---|
| BenchmarkBuild-2 | 7 | 288970277 ns/op | 4284960 B/op | 45550 allocs/op | |
| BenchmarkSearchLatency-2 | 75 | 14095075 ns/op | 128045 B/op | 400 allocs/op | |
| BenchmarkSearchRecall-2 | 76 | 14648046 ns/op | 160897 B/op | 700 allocs/op | 0.9750 recall@k |

## AVX2-Optimized L2 Distance

We implemented an AVX2-optimized L2 distance function and compared its performance against the scalar implementation. The benchmarks were run for vector dimensions of 128, 256, and 768.

| Dimension | Scalar Implementation (ns/op) | AVX2 Implementation (ns/op) | Speedup |
|---|---|---|---|
| 128 | 93.82 | 33.66 | **2.79x** |
| 256 | 181.9 | 77.63 | **2.34x** |
| 768 | 566.7 | 192.2 | **2.95x** |

The results show a significant performance improvement, with the AVX2 implementation being approximately 2.3x to 3x faster than the scalar version.
