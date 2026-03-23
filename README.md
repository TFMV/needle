# Needle

[![Go Report Card](https://goreportcard.com/badge/github.com/TFMV/needle)](https://goreportcard.com/report/github.com/TFMV/needle)
[![Go Reference](https://pkg.go.dev/badge/github.com/TFMV/needle.svg)](https://pkg.go.dev/github.com/TFMV/needle)

---

# Needle: High-Performance Approximate Nearest Neighbor Search in Go

**Needle** is a **fast, memory-efficient, HNSW-style vector search library** written in Go. It’s designed for developers and researchers needing **large-scale vector search**, flexible storage, and integration with analytics pipelines.

It supports **generics**, **float32 and float64 vectors**, **memory-mapped persistence**, and **Arrow IPC export**, while providing a modular execution engine for custom search strategies.

---

## 🚀 Features

* **Generics & Multi-Type Support** – `float32` and `float64` vectors supported.
* **HNSW-style graph search** – Approximate nearest neighbor search with configurable `M` and `ef`.
* **Batch insertion** – Efficiently add large datasets at once using `AddBatch`.
* **Memory-mapped storage** – Use `MMapStore` to store vectors on disk for instant reloads.
* **Arrow IPC integration** – Export vectors to Apache Arrow arrays for analytics pipelines.
* **Pluggable execution layer** – Custom query executors for experimentations or optimized search.

---

## 📦 Installation

```bash
go get github.com/TFMV/needle
```

Requires **Go 1.20+** for generics support.

---

## 🏗 Quick Example

```go
package main

import (
    "fmt"
    "math/rand"
    "github.com/TFMV/needle"
    "github.com/apache/arrow-go/v18/arrow/memory"
)

func main() {
    dim := 128
    n := 1000

    // Generate random float32 vectors
    data := make([][]float32, n)
    for i := range data {
        data[i] = make([]float32, dim)
        for j := range data[i] {
            data[i][j] = rand.Float32()
        }
    }

    // Create graph and executor
    g := needle.NewGraphFloat32(dim)
    g.SetExecutor(&needle.SimpleExecutor[float32]{})

    // Batch insert
    items := make([]struct {
        ID  int
        Vec []float32
    }, n)
    for i, v := range data {
        items[i].ID = i
        items[i].Vec = v
    }
    g.AddBatch(items)

    // Query top 10 neighbors
    query := data[0]
    results, _ := g.Search(query, 10)
    fmt.Println("Top 10 nearest neighbors:", results)

    // Export to Arrow
    alloc := memory.DefaultAllocator
    arrowArray := needle.ArrowExport(data, alloc)
    defer arrowArray.Release()
}
```

---

## 🧪 Benchmarking

Needle includes a benchmark program demonstrating **graph construction, search performance, and recall evaluation**:

```bash
go run bench/needle_benchmark.go -dim 128 -n 10000 -queries 100 -k 10
```

Metrics reported:

* **build_time_ms** – Time to construct the graph.
* **avg_latency_ms** – Average query latency.
* **p95_latency_ms** – 95th percentile query latency.
* **qps** – Queries per second.
* **memory_mb** – Memory footprint of graph and vectors.
* **recall_at_K** – Accuracy compared to brute-force top-K search.

Sample output:

```
build_time_ms 350.23
avg_latency_ms 0.42
p95_latency_ms 0.87
qps 238095.00
memory_mb 45
recall_at_10 0.9400
```

---

## ⚡ Design Highlights

* **Graph Structure** – Each node stores neighbors at multiple levels for efficient HNSW search.
* **VisitedList** – Lightweight memory-efficient tracking during searches.
* **Pluggable PQ Codec** – Optionally compress vectors with custom Product Quantization.
* **MMapStore** – Persistent storage for large datasets.
* **Executor Interface** – Swap in optimized search strategies without modifying the core graph.

---

## 🛠 Advanced Usage

* **Custom Distance Functions** – Supply your own `DistanceFunc[T]`.
* **Custom Executors** – Implement `Executor[T]` for alternative query strategies.
* **Memory-Mapped Graphs** – `MMapStore` allows persistence of vector datasets to disk.
* **Arrow IPC** – Export vectors to Arrow for fast analytics or interoperability with DuckDB or PyArrow.

---

## 🤝 Contributing

Contributions welcome! Needle is modular and designed for extension:

* Add distance metrics (cosine, Manhattan, etc.).
* Implement advanced PQ codecs or compression strategies.
* Extend storage backends (disk, cloud, streaming).
* Integrate with analytics pipelines for real-time vector search.

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

---
