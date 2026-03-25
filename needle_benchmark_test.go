package needle

import (
	"math/rand"
	"sort"
	"testing"
)

const (
	numVectors = 1000
	dim        = 128
	k          = 10
)

// --- Ground Truth Calculation ---

// bruteForceKNN finds the true k-nearest neighbors via exhaustive search.
func bruteForceKNN(dataset [][]float32, query []float32, k int) []int {
	type distIdx struct {
		dist float32
		idx  int
	}

	distances := make([]distIdx, len(dataset))
	for i, vec := range dataset {
		distances[i] = distIdx{dist: l2Float32Scalar(query, vec), idx: i}
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].dist < distances[j].dist
	})

	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = distances[i].idx
	}
	return result
}

// calculateRecall computes the recall@k.
func calculateRecall(groundTruth, approximate []int) float64 {
	set := make(map[int]struct{}, len(groundTruth))
	for _, id := range groundTruth {
		set[id] = struct{}{}
	}

	intersect := 0
	for _, id := range approximate {
		if _, found := set[id]; found {
			intersect++
		}
	}
	return float64(intersect) / float64(len(groundTruth))
}

// --- Benchmark Setup ---

func generateData(n, dim int) [][]float32 {
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rand.Float32()
		}
	}
	return vectors
}

func setupGraph(b *testing.B, n, dim int, vectors [][]float32) *Graph[float32] {
	b.Helper()
	config := DefaultConfig(dim)
	g := NewGraph[float32](config)

	items := make([]Item[float32], n)
	for i := 0; i < n; i++ {
		items[i] = Item[float32]{ID: i, Vec: vectors[i]}
	}

	if err := g.AddBatch(items); err != nil {
		b.Fatalf("Failed to add vectors in batch: %v", err)
	}

	b.ReportAllocs()
	return g
}

// --- Benchmarks ---

func BenchmarkBuild(b *testing.B) {
	vectors := generateData(numVectors, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		setupGraph(b, numVectors, dim, vectors)
	}
}

func BenchmarkSearchLatency(b *testing.B) {
	vectors := generateData(numVectors, dim)
	g := setupGraph(b, numVectors, dim, vectors)
	queries := generateData(100, dim)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		for _, q := range queries {
			_, _ = g.Search(q, k)
		}
	}
}

func BenchmarkSearchRecall(b *testing.B) {
	vectors := generateData(numVectors, dim)
	g := setupGraph(b, numVectors, dim, vectors)
	queries := generateData(100, dim)

	groundTruths := make([][]int, len(queries))
	for i, q := range queries {
		groundTruths[i] = bruteForceKNN(vectors, q, k)
	}

	b.ResetTimer()

	var totalRecall float64
	var numSearches int

	for i := 0; i < b.N; i++ {
		for j, q := range queries {
			approximate, _ := g.Search(q, k)
			recall := calculateRecall(groundTruths[j], approximate)
			totalRecall += recall
			numSearches++
		}
	}

	b.ReportMetric(totalRecall/float64(numSearches), "recall@k")
}
