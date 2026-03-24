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

func newNode(id, idx, lvl int) *Node {
	nbrs := make([][]uint32, lvl+1)
	for i := range nbrs {
		nbrs[i] = make([]uint32, 0, 32)
	}
	return &Node{id: id, idx: idx, level: lvl, neighbors: nbrs}
}

type Graph[T Float] struct {
	dim int

	m              int
	efSearch       int
	efConstruction int

	dist DistanceFunc[T]

	// storage
	vectorData []T

	nodes   []*Node
	idToIdx map[int]int

	enter unsafe.Pointer
	level int32

	mu sync.RWMutex

	visited *VisitedList

	/////////////////////
	// FUTURE EXTENSION //
	/////////////////////

	pq      PQCodec[T]
	storage Storage
	exec    Executor[T]
}

func (g *Graph[T]) getVector(idx int) []T {
	start := idx * g.dim
	return g.vectorData[start : start+g.dim]
}

/////////////////////////
// PQ (PLUGGABLE CORE) //
/////////////////////////

type PQCodec[T Float] interface {
	Encode(vec []T) []byte
	Decode(code []byte) []T
	Distance(query []T, code []byte) float32
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
// EXECUTION LAYER  //
//////////////////////

type Executor[T Float] interface {
	Scan(g *Graph[T], query []T, k int) []int
}

//////////////////////////
// CONSTRUCTOR         //
//////////////////////////

func NewGraphFloat32(dim int) *Graph[float32] {
	return &Graph[float32]{
		dim:            dim,
		m:              16,
		efSearch:       64,
		efConstruction: 128,
		dist:           l2Float32,
		vectorData:     make([]float32, 0),
		nodes:          make([]*Node, 0),
		idToIdx:        make(map[int]int),
		visited:        NewVisited(0),
	}
}

//////////////////////
// INSERTION        //
//////////////////////

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

	g.vectorData = append(g.vectorData, vec...)
	node := newNode(id, idx, g.randomLevel())

	g.nodes = append(g.nodes, node)
	g.idToIdx[id] = idx

	if idx == 0 {
		atomic.StorePointer(&g.enter, unsafe.Pointer(node))
		atomic.StoreInt32(&g.level, int32(node.level))
		return nil
	}

	g.connect(node)

	return nil
}

//////////////////////
// SEARCH           //
//////////////////////

func (g *Graph[T]) Search(q []T, k int) ([]int, error) {
	if len(q) != g.dim {
		return nil, fmt.Errorf("bad query dim")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	if len(g.nodes) == 0 {
		return []int{}, nil
	}

	if g.exec != nil {
		return g.exec.Scan(g, q, k), nil
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

	maxLevel := int(atomic.LoadInt32(&g.level))

	for l := maxLevel; l > 0; l-- {
		ep = g.greedy(q, ep, l)
	}

	cands := g.searchLayer(q, ep, 0, max(g.efSearch, k), g.visited)

	if len(cands) > k {
		return cands[:k]
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
			n := g.nodes[int(ni)]
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

	// Arena-based allocation for this search
	arena := make([]candidate, ef*20)
	arenaIdx := 0
	newCand := func(idx int, dist float32) *candidate {
		if arenaIdx < len(arena) {
			c := &arena[arenaIdx]
			c.idx = idx
			c.dist = dist
			arenaIdx++
			return c
		}
		// Fallback for large searches
		return &candidate{idx, dist}
	}

	pq := &minHeap{}
	res := &maxHeap{}

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

		node := g.nodes[c.idx]

		for _, ni := range node.neighbors[lvl] {
			i := int(ni)
			if visited.Seen(i) {
				continue
			}
			visited.Mark(i)

			d := g.dist(q, g.getVector(i))

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

	out := make([]*candidate, res.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(res).(*candidate)
	}
	return out
}

//////////////////////
// GRAPH BUILD      //
//////////////////////

func (g *Graph[T]) connect(n *Node) {
	ep := (*Node)(atomic.LoadPointer(&g.enter))

	maxLevel := int(atomic.LoadInt32(&g.level))

	cur := ep

	vector := g.getVector(n.idx)
	for l := maxLevel; l > n.level; l-- {
		cur = g.greedy(vector, cur, l)
	}

	for l := min(n.level, maxLevel); l >= 0; l-- {
		cands := g.searchLayer(vector, cur, l, g.efConstruction, g.visited)
		selected := g.selectNeighborsHeuristic(cands, g.m)

		for _, c := range selected {
			n.neighbors[l] = append(n.neighbors[l], uint32(c.idx))
			g.nodes[c.idx].neighbors[l] = append(g.nodes[c.idx].neighbors[l], uint32(n.idx))
		}
	}
}

// selectNeighborsHeuristic selects M diverse neighbors from a set of candidates using the HNSW heuristic.
func (g *Graph[T]) selectNeighborsHeuristic(cands []*candidate, m int) []*candidate {
	selected := make([]*candidate, 0, m)
	if len(cands) == 0 {
		return selected
	}

	for _, cand := range cands {
		if len(selected) >= m {
			break
		}

		// Heuristic: Add the candidate if it's closer to the query node than to any neighbor already selected.
		isGood := true
		candVec := g.getVector(cand.idx)

		for _, selectedNeighbor := range selected {
			selectedNeighborVec := g.getVector(selectedNeighbor.idx)
			distToSelected := g.dist(candVec, selectedNeighborVec)

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

//////////////////////
// EXECUTION ENGINE //
//////////////////////

type SimpleExecutor[T Float] struct{}

func (e *SimpleExecutor[T]) Scan(g *Graph[T], q []T, k int) []int {
	res := g.search(q, k)
	out := make([]int, len(res))
	for i := range res {
		out[i] = g.nodes[res[i].idx].id
	}
	return out
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

func (g *Graph[T]) SetExecutor(exec Executor[T]) {
	g.exec = exec
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
