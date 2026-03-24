package needle

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func benchmarkAdd(b *testing.B, g *Graph[float32], vecs [][]float32) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		g.Add(i, vecs[i])
	}
}

func benchmarkSearch(b *testing.B, g *Graph[float32], queries [][]float32, k int) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		g.Search(q, k)
	}
}

func BenchmarkHNSW(b *testing.B) {
	n, dim, k := 10000, 128, 10
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = make([]float32, dim)
		for j := range vecs[i] {
			vecs[i][j] = rand.Float32()
		}
	}

	queries := make([][]float32, 100)
	for i := range queries {
		queries[i] = make([]float32, dim)
		for j := range queries[i] {
			queries[i][j] = rand.Float32()
		}
	}

	for _, m := range []int{16, 32} {
		for _, efc := range []int{64, 128} {
			for _, efs := range []int{32, 64} {

				g := NewGraphFloat32(dim)
				g.SetParams(m, efs, efc)

				// Populate graph before search benchmark
				for i := 0; i < n; i++ {
					g.Add(i, vecs[i])
				}

				b.Run(fmt.Sprintf("Search-M%d-efC%d-efS%d", m, efc, efs), func(b *testing.B) {
					b.ResetTimer()
					start := time.Now()
					benchmarkSearch(b, g, queries, k)
					duration := time.Since(start)
					qps := float64(b.N) / duration.Seconds()
					b.ReportMetric(qps, "qps")
				})
			}
		}
	}
}
