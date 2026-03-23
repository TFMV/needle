package needle

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

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

func l2Float32(a, b []float32) float32 {
	var sum float32
	i := 0
	for i <= len(a)-8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7
		i += 8
	}
	for i < len(a) {
		d := a[i] - b[i]
		sum += d * d
		i++
	}
	return sum
}

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

type minHeap []*candidate
type maxHeap []*candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(*candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)        { *h = append(*h, x.(*candidate)) }
func (h *maxHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
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

	m  int
	ef int

	dist DistanceFunc[T]

	// storage
	vectors [][]T

	nodes   []*Node
	idToIdx map[int]int

	enter unsafe.Pointer
	level int32

	mu sync.RWMutex

	/////////////////////
	// FUTURE EXTENSION //
	/////////////////////

	pq      PQCodec[T]
	storage Storage
	exec    Executor[T]
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
		dim:     dim,
		m:       16,
		ef:      64,
		dist:    l2Float32,
		vectors: make([][]float32, 0),
		nodes:   make([]*Node, 0),
		idToIdx: make(map[int]int),
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

	g.vectors = append(g.vectors, vec)
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

	return g.searchLayer(q, ep, 0, max(g.ef, k))
}

//////////////////////
// CORE SEARCH      //
//////////////////////

func (g *Graph[T]) greedy(q []T, cur *Node, lvl int) *Node {
	best := cur
	bestDist := g.dist(q, g.vectors[cur.idx])

	changed := true
	for changed {
		changed = false
		for _, ni := range best.neighbors[lvl] {
			n := g.nodes[int(ni)]
			d := g.dist(q, g.vectors[n.idx])
			if d < bestDist {
				bestDist = d
				best = n
				changed = true
			}
		}
	}
	return best
}

func (g *Graph[T]) searchLayer(q []T, entry *Node, lvl, ef int) []*candidate {
	visited := NewVisited(len(g.nodes))

	pq := &minHeap{}
	res := &maxHeap{}

	d0 := g.dist(q, g.vectors[entry.idx])

	heap.Push(pq, &candidate{entry.idx, d0})
	heap.Push(res, &candidate{entry.idx, d0})
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

			d := g.dist(q, g.vectors[i])

			if res.Len() < ef || d < (*res)[0].dist {
				heap.Push(pq, &candidate{i, d})
				heap.Push(res, &candidate{i, d})

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

	for l := maxLevel; l > n.level; l-- {
		cur = g.greedy(g.vectors[n.idx], cur, l)
	}

	for l := min(n.level, maxLevel); l >= 0; l-- {
		cands := g.searchLayer(g.vectors[n.idx], cur, l, g.ef)
		selected := selectTopK(cands, g.m)

		for _, c := range selected {
			n.neighbors[l] = append(n.neighbors[l], uint32(c.idx))
			g.nodes[c.idx].neighbors[l] = append(g.nodes[c.idx].neighbors[l], uint32(n.idx))
		}
	}
}

//////////////////////
// UTIL             //
//////////////////////

func selectTopK(c []*candidate, k int) []*candidate {
	if len(c) <= k {
		return c
	}
	sort.Slice(c, func(i, j int) bool { return c[i].dist < c[j].dist })
	return c[:k]
}

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

func ArrowExport(vecs [][]float32, alloc memory.Allocator) *array.Float32 {
	builder := array.NewFloat32Builder(alloc)
	for _, v := range vecs {
		builder.AppendValues(v, nil)
	}
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

// SetParams allows tuning of HNSW parameters: connectivity (m) and ef (search depth)
func (g *Graph[T]) SetParams(m int, ef int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if m > 0 {
		g.m = m
	}
	if ef > 0 {
		g.ef = ef
	}
}
