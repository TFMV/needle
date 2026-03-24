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
	if g.efSearch != 64 {
		t.Errorf("expected default efSearch=64, got %d", g.efSearch)
	}
	if g.efConstruction != 128 {
		t.Errorf("expected default efConstruction=128, got %d", g.efConstruction)
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

	// Search for the point closest to {0.1, 0.1}
	// The result should be point {0, 0} which has ID 0
	q := []float32{0.1, 0.1}
	res, err := g.Search(q, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(res) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res))
	}
	if res[0] != 0 {
		t.Errorf("expected result ID 0, got %d", res[0])
	}
}

// TestSetParams validates parameter setting
func TestSetParams(t *testing.T) {
	g := NewGraphFloat32(4)
	g.SetParams(32, 100, 200)

	if g.m != 32 {
		t.Errorf("expected m=32, got %d", g.m)
	}
	if g.efSearch != 100 {
		t.Errorf("expected efSearch=100, got %d", g.efSearch)
	}
	if g.efConstruction != 200 {
		t.Errorf("expected efConstruction=200, got %d", g.efConstruction)
	}
}

// TestConcurrentAddAndSearch tests the graph under concurrent access
func TestConcurrentAddAndSearch(t *testing.T) {
	g := NewGraphFloat32(8)
	var wg sync.WaitGroup
	var itemsAdded int32

	// Concurrently add items
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			vec := make([]float32, 8)
			for j := range vec {
				vec[j] = rand.Float32()
			}
			if err := g.Add(id, vec); err == nil {
				atomic.AddInt32(&itemsAdded, 1)
			}
		}(i)
	}

	// Concurrently search
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := make([]float32, 8)
			for j := range q {
				q[j] = rand.Float32()
			}
			_, err := g.Search(q, 5)
			if atomic.LoadInt32(&itemsAdded) > 1 && err != nil {
				t.Errorf("Search failed during concurrent adds: %v", err)
			}
		}()
	}

	wg.Wait()
}
