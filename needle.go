package needle

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"container/heap"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

const cacheLineSize = 64

func alignedAllocator[T Float](size int) []T {
	buf := make([]T, size+cacheLineSize)
	ptr := unsafe.Pointer(&buf[0])
	offset := int(uintptr(ptr)) & (cacheLineSize - 1)
	var aligned []T
	if offset == 0 {
		aligned = buf[:size]
	} else {
		aligned = buf[cacheLineSize-offset : size+cacheLineSize-offset]
	}
	return aligned
}

func syscallMmap(f *os.File, size int) ([]byte, error) {
	return syscall.Mmap(
		int(f.Fd()),
		0,
		size,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_SHARED,
	)
}

//////////////////////////////
// GENERICS + DISTANCE CORE //
//////////////////////////////

type Float interface {
	~float32 | ~float64
}

type DistanceFunc[T Float] func(a, b []T) float32

func l2Float64(a, b []float64) float32 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(sum)
}

//////////////////////////
// CORE DATA STRUCTURES //
//////////////////////////

type candidate struct {
	idx  int
	dist float32
}

//////////////////////
// VISITED TRACKING //
//////////////////////

type VisitedList struct {
	cur uint32
	arr []uint32
}

func NewVisited(n int) *VisitedList {
	return &VisitedList{
		cur: 1,
		arr: make([]uint32, n),
	}
}

func (v *VisitedList) Reset() { v.cur++ }
func (v *VisitedList) Seen(i int) bool {
	return v.arr[i] == v.cur
}
func (v *VisitedList) Mark(i int) {
	v.arr[i] = v.cur
}

//////////////////
// NODE / GRAPH //
//////////////////

type Node struct {
	id        int
	idx       int
	level     int
	neighbors [][]uint32
}

func newNode(id, idx, lvl int) Node {
	nbrs := make([][]uint32, lvl+1)
	for i := range nbrs {
		nbrs[i] = make([]uint32, 0, 32)
	}
	return Node{id: id, idx: idx, level: lvl, neighbors: nbrs}
}

// Config holds the configuration for the Graph.
// Use DefaultConfig() to get a new Config with default values.
type Config[T Float] struct {
	dim            int
	m              int
	efSearch       int
	efConstruction int
	pqThreshold    int
	useOPQ         bool // Add this line
	dist           DistanceFunc[T]
}

// DefaultConfig returns a new Config with default values for float32.
func DefaultConfig() *Config[float32] {
	return &Config[float32]{
		dim:            128,
		m:              16,
		efSearch:       64,
		efConstruction: 128,
		pqThreshold:    1000,
		useOPQ:         false, // Add this line
		dist:           l2Float32,
	}
}

// Graph represents the HNSW graph.
// It stores the graph structure, vector data, and configurations for search and insertion.
type Graph[T Float] struct {
	dim int // Dimension of the vectors

	// HNSW parameters
	m              int // Max number of neighbors for each node
	efSearch       int // Size of the dynamic candidate list for search
	efConstruction int // Size of the dynamic candidate list for construction

	dist DistanceFunc[T] // Distance function to use

	// Data storage
	vectorData []T // Stores the raw vectors

	// Product Quantization (PQ)
	pq          *PQCodec[T]   // PQ codec for vector compression
	pqCodes     [][]byte      // PQ codes for each vector
	pqThreshold int           // Number of vectors to store before training PQ
	pqTrained   atomic.Bool   // Flag indicating whether PQ is trained

	// Optimized Product Quantization (OPQ)
	opq         *OPQCodec[T]  // OPQ codec for vector compression
	opqTrained  atomic.Bool   // Flag indicating whether OPQ is trained
	useOPQ      bool          // Flag to enable OPQ

	// Graph structure
	nodes   []Node      // Slice of nodes in the graph
	idToIdx map[int]int // Map from external ID to internal index

	enter unsafe.Pointer // Entry point to the graph
	level atomic.Int32   // Highest level in the graph

	mu sync.RWMutex // Mutex for concurrent access

	visited *VisitedList // Visited list for search

	// Memory pools for performance optimization
	minHeapPool        sync.Pool // Pool for min-heaps used in search
	maxHeapPool        sync.Pool // Pool for max-heaps used in search
	arenaPool          sync.Pool // Pool for candidate arenas used in search
	candidateSlicePool sync.Pool // Pool for candidate slices used in searchLayer

	// For testing purposes
	trainingWg *sync.WaitGroup
}

// NewGraphFromConfig creates a new Graph from the given configuration.
func NewGraphFromConfig[T Float](config *Config[T]) *Graph[T] {
	return &Graph[T]{
		dim:            config.dim,
		m:              config.m,
		efSearch:       config.efSearch,
		efConstruction: config.efConstruction,
		dist:           config.dist,
		vectorData:     alignedAllocator[T](0),
		nodes:          make([]Node, 0),
		idToIdx:        make(map[int]int),
		visited:        NewVisited(0),
		pqThreshold:    config.pqThreshold,
		useOPQ:         config.useOPQ,
		minHeapPool: sync.Pool{
			New: func() interface{} {
				return &minHeap{}
			},
		},
		maxHeapPool: sync.Pool{
			New: func() interface{} {
				return &maxHeap{}
			},
		},
		arenaPool: sync.Pool{
			New: func() interface{} {
				return make([]candidate, 0, 1024)
			},
		},
		candidateSlicePool: sync.Pool{
			New: func() interface{} {
				slice := make([]*candidate, 0, 128)
				return &slice
			},
		},
	}
}

func (g *Graph[T]) getVector(idx int) []T {
	start := idx * g.dim
	return g.vectorData[start : start+g.dim]
}

//////////////////////
// STORAGE ABSTRACTION
//////////////////////

type Storage interface {
	Append(raw []byte) error
	Read(idx int) ([]byte, error)
	Close() error
}

//////////////////////
// INSERTION        //
//////////////////////

// Add adds a new vector to the graph with a given ID.
//
// The process involves:
// 1. Assigning an internal index to the new vector.
// 2. Storing the vector data.
// 3. If PQ/OPQ is enabled and not yet trained, triggering the training process
//    once the number of vectors reaches the specified threshold.
// 4. Creating a new node for the vector with a randomly assigned level.
// 5. Connecting the new node to the graph by finding its neighbors.
//
// This function acquires a lock to ensure thread safety during graph modification.
func (g *Graph[T]) Add(id int, vec []T) error {
	if len(vec) != g.dim {
		return fmt.Errorf("dimension mismatch")
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	idx := len(g.nodes)

	if idx >= len(g.visited.arr) {
		newArr := make([]uint32, (idx+1)*2)
		copy(newArr, g.visited.arr)
		g.visited.arr = newArr
	}

	// Grow vectorData if needed
	if len(g.vectorData) < (idx+1)*g.dim {
		newSize := (idx + 1) * g.dim * 2 // Double the size
		newVecData := alignedAllocator[T](newSize)
		copy(newVecData, g.vectorData)
		g.vectorData = newVecData
	}

	copy(g.getVector(idx), vec)

	// Handle PQ code
	if g.pqTrained.Load() {
		g.pqCodes = append(g.pqCodes, g.pq.Encode(vec))
	} else if g.opqTrained.Load() {
		g.pqCodes = append(g.pqCodes, g.opq.Encode(vec))
	} else {
		g.pqCodes = append(g.pqCodes, nil) // Placeholder
	}

	node := newNode(id, idx, g.randomLevel())

	g.nodes = append(g.nodes, node)
	g.idToIdx[id] = idx

	if idx == 0 {
		atomic.StorePointer(&g.enter, unsafe.Pointer(&g.nodes[0]))
		g.level.Store(int32(node.level))
		return nil
	}

	if !g.pqTrained.Load() && !g.opqTrained.Load() && len(g.nodes) >= g.pqThreshold {
		go g.trainQuantizer()
	}

	g.connect(&g.nodes[idx])

	return nil
}

// trainQuantizer trains the PQ or OPQ quantizer in a separate goroutine.
func (g *Graph[T]) trainQuantizer() {
	// For testing purposes, we can wait for the training to finish.
	if g.trainingWg != nil {
		defer g.trainingWg.Done()
	}

	// This function should be called without holding the graph's lock.
	// It will acquire the lock when needed.
	g.mu.Lock()
	vectors := make([][]T, len(g.nodes))
	for i := 0; i < len(g.nodes); i++ {
		vectors[i] = g.getVector(i)
	}
	g.mu.Unlock()

	if g.useOPQ {
		opq, opqErr := NewOPQCodec[T](g.dim, 8, 256, g.dist)
		if opqErr != nil {
			// Handle error appropriately, e.g., log it
			return
		}
		if trainErr := opq.Train(vectors); trainErr != nil {
			// Handle error
			return
		}

		g.mu.Lock()
		g.opq = opq
		for i := 0; i < len(g.nodes); i++ {
			g.pqCodes[i] = g.opq.Encode(g.getVector(i))
		}
		g.opqTrained.Store(true)
		g.mu.Unlock()
	} else {
		pq, pqErr := NewPQCodec[T](g.dim, 8, 256, g.dist)
		if pqErr != nil {
			// Handle error
			return
		}
		if trainErr := pq.Train(vectors); trainErr != nil {
			// Handle error
			return
		}

		g.mu.Lock()
		g.pq = pq
		for i := 0; i < len(g.nodes); i++ {
			g.pqCodes[i] = g.pq.Encode(g.getVector(i))
		}
		g.pqTrained.Store(true)
		g.mu.Unlock()
	}
}


//////////////////////
// SEARCH           //
//////////////////////

// Search performs a k-nearest neighbor search for the given query vector.
//
// The search process starts from the entry point at the highest level of the graph.
// It greedily traverses the graph at each level to find the best entry point for the next lower level.
// Once it reaches the base layer (level 0), it performs a more exhaustive search to find the k-nearest neighbors.
//
// This function acquires a read lock to allow for concurrent searches.
func (g *Graph[T]) Search(q []T, k int) ([]int, error) {
	if len(q) != g.dim {
		return nil, fmt.Errorf("bad query dim")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	if len(g.nodes) == 0 {
		return []int{}, nil
	}

	res := g.search(q, k)

	out := make([]int, len(res))
	for i, c := range res {
		out[i] = g.nodes[c.idx].id
	}
	return out, nil
}

func (g *Graph[T]) search(q []T, k int) []*candidate {
	ep := (*Node)(atomic.LoadPointer(&g.enter))

	maxLevel := int(g.level.Load())

	// Greedily traverse from the top level down to level 1
	for l := maxLevel; l > 0; l-- {
		ep = g.greedy(q, ep, l)
	}

	// Perform the main search on the base layer (level 0)
	cands := g.searchLayer(q, ep, 0, max(g.efSearch, k), g.visited)

	// If the number of candidates is greater than k, return the top k
	if len(cands) > k {
		cands = cands[:k]
	}

	return cands
}


//////////////////////
// CORE SEARCH      //
//////////////////////

func (g *Graph[T]) greedy(q []T, cur *Node, lvl int) *Node {
	best := cur
	bestDist := g.dist(q, g.getVector(cur.idx))

	changed := true
	for changed {
		changed = false
		for _, ni := range best.neighbors[lvl] {
			n := &g.nodes[int(ni)]
			d := g.dist(q, g.getVector(n.idx))
			if d < bestDist {
				bestDist = d
				best = n
				changed = true
			}
		}
	}
	return best
}

func (g *Graph[T]) searchLayer(q []T, entry *Node, lvl, ef int, visited *VisitedList) []*candidate {
	visited.Reset()

	arena := g.arenaPool.Get().([]candidate)
	arena = arena[:0]
	defer g.arenaPool.Put(arena)

	arenaIdx := 0
	newCand := func(idx int, dist float32) *candidate {
		if arenaIdx < cap(arena) {
			arena = arena[:arenaIdx+1]
			c := &arena[arenaIdx]
			c.idx = idx
			c.dist = dist
			arenaIdx++
			return c
		}
		// Fallback for large searches
		return &candidate{idx, dist}
	}

	pq := g.minHeapPool.Get().(*minHeap)
	pq.Reset()
	defer g.minHeapPool.Put(pq)

	res := g.maxHeapPool.Get().(*maxHeap)
	res.Reset()
	defer g.maxHeapPool.Put(res)

	d0 := g.dist(q, g.getVector(entry.idx))
	c0 := newCand(entry.idx, d0)

	heap.Push(pq, c0)
	heap.Push(res, c0)
	visited.Mark(entry.idx)

	for pq.Len() > 0 {
		c := heap.Pop(pq).(*candidate)

		if res.Len() >= ef && c.dist > (*res)[0].dist {
			break
		}

		node := &g.nodes[c.idx]

		for _, ni := range node.neighbors[lvl] {
			i := int(ni)
			if visited.Seen(i) {
				continue
			}
			visited.Mark(i)

			var d float32
			if g.opqTrained.Load() {
				d = g.opq.Distance(q, g.pqCodes[i])
			} else if g.pqTrained.Load() {
				d = g.pq.Distance(q, g.pqCodes[i])
			} else {
				d = g.dist(q, g.getVector(i))
			}

			if res.Len() < ef || d < (*res)[0].dist {
				c := newCand(i, d)
				heap.Push(pq, c)
				heap.Push(res, c)

				if res.Len() > ef {
					heap.Pop(res)
				}
			}
		}
	}

	outPtr := g.candidateSlicePool.Get().(*[]*candidate)
	out := *outPtr
	out = out[:0]

	for res.Len() > 0 {
		out = append(out, heap.Pop(res).(*candidate))
	}

	// Reverse the slice to get the correct order (nearest to farthest).
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}

	*outPtr = out
	return out
}

//////////////////////
// GRAPH BUILD      //
//////////////////////

func (g *Graph[T]) connect(n *Node) {
	ep := (*Node)(atomic.LoadPointer(&g.enter))

	maxLevel := int(g.level.Load())

	cur := ep

	vector := g.getVector(n.idx)
	for l := maxLevel; l > n.level; l-- {
		cur = g.greedy(vector, cur, l)
	}

	for l := min(n.level, maxLevel); l >= 0; l-- {
		cands := g.searchLayer(vector, cur, l, g.efConstruction, g.visited)
		selected := g.selectNeighbors(cands, g.m)
		g.candidateSlicePool.Put(&cands)

		for _, c := range selected {
			n.neighbors[l] = append(n.neighbors[l], uint32(c.idx))
			g.nodes[c.idx].neighbors[l] = append(g.nodes[c.idx].neighbors[l], uint32(n.idx))
		}
	}
}

// selectNeighbors selects M diverse neighbors from a set of candidates.
func (g *Graph[T]) selectNeighbors(cands []*candidate, m int) []*candidate {
	// This is the heuristic from the HNSW paper.
	// It attempts to select a diverse set of neighbors.
	// This is NOT the same as the simpler heuristic from the reference implementation.

	selected := make([]*candidate, 0, m)
	if len(cands) == 0 {
		return selected
	}

	for _, cand := range cands {
		if len(selected) >= m {
			break
		}

		isGood := true
		candVec := g.getVector(cand.idx)

		for _, selectedNeighbor := range selected {
			var distToSelected float32
			if g.opqTrained.Load() {
				distToSelected = g.opq.Distance(candVec, g.pqCodes[selectedNeighbor.idx])
			} else if g.pqTrained.Load() {
				distToSelected = g.pq.Distance(candVec, g.pqCodes[selectedNeighbor.idx])
			} else {
				selectedNeighborVec := g.getVector(selectedNeighbor.idx)
				distToSelected = g.dist(candVec, selectedNeighborVec)
			}

			if distToSelected < cand.dist {
				isGood = false
				break
			}
		}

		if isGood {
			selected = append(selected, cand)
		}
	}
	return selected
}

//////////////////////
// UTIL             //
//////////////////////

func (g *Graph[T]) randomLevel() int {
	l := 0
	for rand.Float64() < 1/math.E && l < 16 {
		l++
	}
	return l
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

//////////////////////
// MMAP STORAGE     //
//////////////////////

type MMapStore struct {
	file *os.File
	data []byte
}

func NewMMapStore(path string, size int) (*MMapStore, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}

	if err := f.Truncate(int64(size)); err != nil {
		return nil, err
	}

	data, err := syscallMmap(f, size)
	if err != nil {
		return nil, err
	}

	return &MMapStore{file: f, data: data}, nil
}

func (m *MMapStore) Append(b []byte) error {
	copy(m.data, b)
	return nil
}

func (m *MMapStore) Read(idx int) ([]byte, error) {
	return m.data, nil
}

func (m *MMapStore) Close() error {
	return m.file.Close()
}

//////////////////////
// ARROW IPC HOOK   //
//////////////////////

func ArrowExport(g *Graph[float32], alloc memory.Allocator) *array.Float32 {
	builder := array.NewFloat32Builder(alloc)
	builder.AppendValues(g.vectorData, nil)
	return builder.NewFloat32Array()
}


// AddBatch adds multiple items to the graph at once.
func (g *Graph[T]) AddBatch(items []struct {
	ID  int
	Vec []T
}) error {
	for _, item := range items {
		if err := g.Add(item.ID, item.Vec); err != nil {
			return err
		}
	}
	return nil
}

// SetParams allows tuning of HNSW parameters.
func (g *Graph[T]) SetParams(m, efSearch, efConstruction int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if m > 0 {
		g.m = m
	}
	if efSearch > 0 {
		g.efSearch = efSearch
	}
	if efConstruction > 0 {
		g.efConstruction = efConstruction
	}
}
