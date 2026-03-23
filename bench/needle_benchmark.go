package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/TFMV/needle"
)

func generateDataFloat32(n, dim int) [][]float32 {
	rng := rand.New(rand.NewSource(42))
	data := make([][]float32, n)
	for i := range data {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		data[i] = vec
	}
	return data
}

func euclideanSquaredFloat32(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func computeGroundTruthFloat32(base, queries [][]float32, k int) [][]int {
	res := make([][]int, len(queries))
	for qi, q := range queries {
		dists := make([]struct {
			idx  int
			dist float32
		}, len(base))
		for i, v := range base {
			dists[i].idx = i
			dists[i].dist = euclideanSquaredFloat32(q, v)
		}
		sort.Slice(dists, func(i, j int) bool { return dists[i].dist < dists[j].dist })
		top := make([]int, k)
		for j := 0; j < k; j++ {
			top[j] = dists[j].idx
		}
		res[qi] = top
	}
	return res
}

func benchmark(n, dim, queries, k, ef, m int) {
	base := generateDataFloat32(n, dim)
	qvecs := generateDataFloat32(queries, dim)
	gt := computeGroundTruthFloat32(base, qvecs, k)

	// Build graph
	start := time.Now()
	g := needle.NewGraphFloat32(dim)
	g.SetExecutor(&needle.SimpleExecutor[float32]{})
	g.SetParams(m, ef) // Optional setter if implemented
	items := make([]struct {
		ID  int
		Vec []float32
	}, len(base))
	for i, vec := range base {
		items[i].ID = i
		items[i].Vec = vec
	}
	if err := g.AddBatch(items); err != nil {
		panic(err)
	}
	buildTime := time.Since(start)

	// Memory usage
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	usedMB := mem.Alloc / (1024 * 1024)

	// Parallel query execution
	latencies := make([]time.Duration, queries)
	var wg sync.WaitGroup
	wg.Add(queries)
	start = time.Now()
	for i, vec := range qvecs {
		go func(i int, vec []float32) {
			defer wg.Done()
			t0 := time.Now()
			res, err := g.Search(vec, k)
			if err != nil {
				panic(err)
			}
			latencies[i] = time.Since(t0)

			hit := 0
			gtSet := make(map[int]struct{}, k)
			for _, idx := range gt[i] {
				gtSet[idx] = struct{}{}
			}
			for _, id := range res {
				if _, ok := gtSet[id]; ok {
					hit++
				}
			}
			gt[i] = []int{hit}
		}(i, vec)
	}
	wg.Wait()
	totalQueryTime := time.Since(start)

	// Compute recall
	hits := 0
	for _, h := range gt {
		hits += h[0]
	}
	recall := float64(hits) / float64(queries*k)

	// Latency metrics
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	sum := time.Duration(0)
	for _, l := range latencies {
		sum += l
	}
	avgLat := sum / time.Duration(len(latencies))
	p95 := latencies[int(float64(len(latencies))*0.95)]
	qps := float64(queries) / totalQueryTime.Seconds()

	fmt.Printf("=== Benchmark: n=%d, dim=%d, queries=%d, k=%d, ef=%d, m=%d ===\n", n, dim, queries, k, ef, m)
	fmt.Printf("Build Time (ms)   : %.2f\n", float64(buildTime.Microseconds())/1000)
	fmt.Printf("Avg Latency (ms)  : %.3f\n", float64(avgLat.Microseconds())/1000)
	fmt.Printf("P95 Latency (ms)  : %.3f\n", float64(p95.Microseconds())/1000)
	fmt.Printf("QPS               : %.2f\n", qps)
	fmt.Printf("Memory Usage (MB) : %d\n", usedMB)
	fmt.Printf("Recall@%d          : %.4f\n\n", k, recall)
}

func main() {
	nList := []int{5000, 10000}
	dimList := []int{64, 128}
	queryList := []int{50, 100}
	kList := []int{5, 10}
	efList := []int{64, 128} // Search depth sweep
	mList := []int{16, 32}   // Optional HNSW connectivity sweep

	for _, n := range nList {
		for _, dim := range dimList {
			for _, queries := range queryList {
				for _, k := range kList {
					for _, ef := range efList {
						for _, m := range mList {
							benchmark(n, dim, queries, k, ef, m)
						}
					}
				}
			}
		}
	}
}
