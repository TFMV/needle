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

* **🟢** — excellent (low latency / high recall)
* **🟡** — moderate (okay latency / slightly lower recall)
* **🔴** — poor (high latency / low recall)

---

## Needle Benchmark Heatmap (n=5k–10k, dim=64–128)

| n   | dim | k  | ef  | m  | Build Time | Avg Latency | Recall   |
| --- | --- | -- | --- | -- | ---------- | ----------- | -------- |
| 5k  | 64  | 5  | 64  | 16 | 🟢 327 ms  | 🟡 0.68 ms  | 🟢 0.996 |
| 5k  | 64  | 5  | 64  | 32 | 🟡 504 ms  | 🟢 0.33 ms  | 🟢 1.000 |
| 5k  | 64  | 5  | 128 | 16 | 🟡 643 ms  | 🟡 0.70 ms  | 🟢 0.996 |
| 5k  | 64  | 5  | 128 | 32 | 🟡 710 ms  | 🟢 0.48 ms  | 🟢 1.000 |
| 5k  | 64  | 10 | 64  | 16 | 🟢 423 ms  | 🟢 0.15 ms  | 🟡 0.988 |
| 5k  | 64  | 10 | 64  | 32 | 🟡 397 ms  | 🟡 0.74 ms  | 🟢 1.000 |
| 5k  | 128 | 5  | 64  | 16 | 🟡 459 ms  | 🟡 0.88 ms  | 🟡 0.976 |
| 5k  | 128 | 5  | 64  | 32 | 🟡 568 ms  | 🟢 0.44 ms  | 🟢 1.000 |
| 5k  | 128 | 5  | 128 | 16 | 🟡 855 ms  | 🟡 0.55 ms  | 🟢 0.996 |
| 5k  | 128 | 5  | 128 | 32 | 🟡 1022 ms | 🟡 1.33 ms  | 🟢 1.000 |
| 10k | 64  | 5  | 64  | 16 | 🟡 1111 ms | 🟢 0.32 ms  | 🟡 0.980 |
| 10k | 64  | 5  | 64  | 32 | 🟡 1105 ms | 🟢 0.24 ms  | 🟢 1.000 |
| 10k | 64  | 5  | 128 | 16 | 🟡 1733 ms | 🟡 0.51 ms  | 🟢 1.000 |
| 10k | 64  | 5  | 128 | 32 | 🟡 1687 ms | 🟢 0.49 ms  | 🟢 1.000 |
| 10k | 128 | 5  | 64  | 16 | 🟡 1245 ms | 🟡 1.10 ms  | 🟡 0.964 |
| 10k | 128 | 5  | 64  | 32 | 🟡 1671 ms | 🟢 0.40 ms  | 🟢 0.992 |
| 10k | 128 | 5  | 128 | 16 | 🟡 1954 ms | 🟡 0.45 ms  | 🟡 0.984 |
| 10k | 128 | 5  | 128 | 32 | 🟡 3014 ms | 🟢 0.63 ms  | 🟢 1.000 |

---

### 🔹 Quick Takeaways

* **🟢 Green cells** highlight sweet spots where latency is low **and** recall is near-perfect.
* **🟡 Yellow cells** indicate moderate trade-offs: slightly higher latency or recall < 0.995.
* **🔴 Red cells** (none here yet) would indicate configurations to avoid.
* For **small-medium vectors (`dim ≤ 64`, `n ≤ 5k`)**, `m=32, ef=64` is ideal.
* For **higher dimensions or larger datasets**, boost `ef` and `m` to maintain perfect recall.

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
