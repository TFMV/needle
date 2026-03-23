package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"time"

	"github.com/TFMV/needle"
	"github.com/apache/arrow-go/v18/arrow/memory"
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

func main() {
	var (
		dim = flag.Int("dim", 128, "vector dimension")
		n   = flag.Int("n", 10000, "dataset size")
		q   = flag.Int("queries", 100, "number of queries")
		k   = flag.Int("k", 10, "neighbors per query")
	)
	flag.Parse()

	base := generateDataFloat32(*n, *dim)
	queries := generateDataFloat32(*q, *dim)

	gt := computeGroundTruthFloat32(base, queries, *k)

	alloc := memory.DefaultAllocator
	start := time.Now()

	// Build graph
	g := needle.NewGraphFloat32(*dim)
	g.SetExecutor(&needle.SimpleExecutor[float32]{})
	git config --global user.name "TFMV"
    git config --global user.email mcgeehan@gmail.com
	// Add batch using AddBatch
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

	// Arrow export for demonstration
	arrowArray := needle.ArrowExport(base, alloc)
	defer arrowArray.Release()

	// Query benchmarking
	latencies := make([]time.Duration, len(queries))
	start = time.Now()
	for i, vec := range queries {
		t0 := time.Now()
		res, err := g.Search(vec, *k)
		if err != nil {
			panic(err)
		}
		latencies[i] = time.Since(t0)

		hit := 0
		gtSet := make(map[int]struct{}, *k)
		for _, idx := range gt[i] {
			gtSet[idx] = struct{}{}
		}
		for _, id := range res {
			if _, ok := gtSet[id]; ok {
				hit++
			}
		}
		gt[i] = []int{hit}
	}
	totalQueryTime := time.Since(start)

	// Compute recall
	hits := 0
	for _, h := range gt {
		hits += h[0]
	}
	recall := float64(hits) / float64(len(queries)*(*k))

	// Latency metrics
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	sum := time.Duration(0)
	for _, l := range latencies {
		sum += l
	}
	avgLat := sum / time.Duration(len(latencies))
	p95 := latencies[int(float64(len(latencies))*0.95)]
	qps := float64(len(latencies)) / totalQueryTime.Seconds()

	fmt.Printf("build_time_ms %.2f\n", float64(buildTime.Microseconds())/1000)
	fmt.Printf("avg_latency_ms %.2f\n", float64(avgLat.Microseconds())/1000)
	fmt.Printf("p95_latency_ms %.2f\n", float64(p95.Microseconds())/1000)
	fmt.Printf("qps %.2f\n", qps)
	fmt.Printf("memory_mb %d\n", usedMB)
	fmt.Printf("recall_at_%d %.4f\n", *k, recall)
}
