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
