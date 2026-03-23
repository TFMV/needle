package needle

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
)

// TestNewGraphFloat32 validates the constructor
func TestNewGraphFloat32(t *testing.T) {
	g := NewGraphFloat32(2)
	if g == nil {
		t.Fatal("NewGraphFloat32 returned nil")
	}
	if g.dim != 2 {
		t.Errorf("expected dim=2, got %d", g.dim)
	}
	if g.m != 16 {
		t.Errorf("expected default m=16, got %d", g.m)
	}
	if g.ef != 64 {
		t.Errorf("expected default ef=64, got %d", g.ef)
	}
	if g.dist == nil {
		t.Error("distance function is nil")
	}
}

// TestAddAndSearch verifies adding nodes and nearest neighbor search
func TestAddAndSearch(t *testing.T) {
	points := [][]float32{
		{0, 0},
		{1, 1},
		{2, 2},
	}

	g := NewGraphFloat32(2)
	for i, p := range points {
		if err := g.Add(i, p); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	query := []float32{0.1, 0.1}
	results, err := g.Search(query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
	if results[0] != 0 {
		t.Errorf("expected first result to be 0, got %d", results[0])
	}
}

// TestConcurrentAccess validates thread safety
func TestConcurrentAccess(t *testing.T) {
	numPoints := 10
	g := NewGraphFloat32(2)

	for i := 0; i < numPoints; i++ {
		if err := g.Add(i, []float32{float32(i), float32(i)}); err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			query := []float32{float32(i), float32(i)}
			results, err := g.Search(query, 3)
			if err != nil {
				t.Errorf("Search failed: %v", err)
			}
			if len(results) != 3 {
				t.Errorf("expected 3 results, got %d", len(results))
			}
		}(i)
	}
	wg.Wait()
}

// TestEdgeCases covers empty graph, single node, dimension mismatch
func TestEdgeCases(t *testing.T) {
	g := NewGraphFloat32(2)

	// Empty graph
	results, err := g.Search([]float32{0, 0}, 1)
	if err != nil {
		t.Fatalf("Search on empty graph failed: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}

	// Single node
	if err := g.Add(1, []float32{1, 1}); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	results, err = g.Search([]float32{0, 0}, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}

	// Dimension mismatch
	err = g.Add(2, []float32{1, 1, 1})
	if err == nil {
		t.Error("expected error for dimension mismatch, got nil")
	}
	_, err = g.Search([]float32{1}, 1)
	if err == nil {
		t.Error("expected error for search dimension mismatch, got nil")
	}
}

// BenchmarkAddFloat32 measures insertion performance
func BenchmarkAddFloat32(b *testing.B) {
	dim := 128
	g := NewGraphFloat32(dim)
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vec[0] = rand.Float32()
		if err := g.Add(i, vec); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkConcurrentAddFloat32 measures parallel insertion
func BenchmarkConcurrentAddFloat32(b *testing.B) {
	dim := 128
	g := NewGraphFloat32(dim)

	vectors := make([][]float32, b.N)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
	}

	var counter int64
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			id := atomic.AddInt64(&counter, 1) - 1
			if int(id) >= len(vectors) {
				continue
			}
			if err := g.Add(int(id), vectors[id]); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkSearchFloat32 measures search performance
func BenchmarkSearchFloat32(b *testing.B) {
	dim := 128
	g := NewGraphFloat32(dim)

	vec := make([]float32, dim)
	for i := 0; i < 1000; i++ {
		for j := range vec {
			vec[j] = rand.Float32()
		}
		if err := g.Add(i, vec); err != nil {
			b.Fatal(err)
		}
	}

	query := make([]float32, dim)
	for i := range query {
		query[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query[0] = rand.Float32()
		if _, err := g.Search(query, 10); err != nil {
			b.Fatal(err)
		}
	}
}

func TestAddBatchFloat32(t *testing.T) {
	g := NewGraphFloat32(2)

	items := make([]struct {
		ID  int
		Vec []float32
	}, 10)
	for i := 0; i < 10; i++ {
		items[i].ID = i
		items[i].Vec = []float32{float32(i), float32(i)}
	}

	if err := g.AddBatch(items); err != nil {
		t.Fatalf("AddBatch failed: %v", err)
	}

	if len(g.nodes) != 10 {
		t.Errorf("expected 10 nodes, got %d", len(g.nodes))
	}

	query := []float32{1.1, 1.1}
	results, err := g.Search(query, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}
}
