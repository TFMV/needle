package needle

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"
)

//////////////////////////////
// DISTANCE CORE (ZERO COST)//
//////////////////////////////

type Float interface {
	~float32 | ~float64
}

type DistanceFunc[T Float] func(a, b *T, dim int) float32

func l2Float32Ptr(aPtr, bPtr *float32, dim int) float32 {
	if useAVX2 {
		return l2Float32AVX2(aPtr, bPtr, dim)
	}
	return l2Float32Scalar(aPtr, bPtr, dim)
}

func l2Generic[T Float](aPtr, bPtr *T, dim int) float32 {
	var sum float32
	size := unsafe.Sizeof(*aPtr)

	for i := 0; i < dim; i++ {
		a := *(*T)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i)*size))
		b := *(*T)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i)*size))
		d := float32(a - b)
		sum += d * d
	}
	return sum
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

func (v *VisitedList) Reset() {
	v.cur++
	if v.cur == 0 {
		for i := range v.arr {
			v.arr[i] = 0
		}
		v.cur = 1
	}
}

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

func newNode(id, idx, lvl int, m int) Node {
	nbrs := make([][]uint32, lvl+1)
	for i := range nbrs {
		if i == 0 {
			nbrs[i] = make([]uint32, 0, 2*m)
		} else {
			nbrs[i] = make([]uint32, 0, m)
		}
	}
	return Node{id: id, idx: idx, level: lvl, neighbors: nbrs}
}

/////////////////////////
// CONFIG              //
/////////////////////////

type Config[T Float] struct {
	dim            int
	m              int
	efSearch       int
	efConstruction int
	dist           DistanceFunc[T]
}

func DefaultConfig(dim int) *Config[float32] {
	return &Config[float32]{
		dim:            dim,
		m:              16,
		efSearch:       64,
		efConstruction: 128,
		dist:           l2Float32Ptr,
	}
}

/////////////////////////
// GRAPH               //
/////////////////////////

type Item[T Float] struct {
	ID  int
	Vec []T
}

type Graph[T Float] struct {
	dim int

	m              int
	efSearch       int
	efConstruction int

	dist DistanceFunc[T]

	vectorData []T

	nodes   []Node
	idToIdx map[int]int

	enter unsafe.Pointer
	level atomic.Int32

	mu sync.RWMutex

	minHeapPool        sync.Pool
	maxHeapPool        sync.Pool
	arenaPool          sync.Pool
	pruneArenaPool     sync.Pool
	candidateSlicePool sync.Pool
	visitedPool        sync.Pool
}

/////////////////////////
// CONSTRUCTOR         //
/////////////////////////

func NewGraph[T Float](config *Config[T]) *Graph[T] {
	g := &Graph[T]{
		dim:            config.dim,
		m:              config.m,
		efSearch:       config.efSearch,
		efConstruction: config.efConstruction,
		dist:           config.dist,
		vectorData:     make([]T, 0, config.dim*1024),
		nodes:          make([]Node, 0, 1024),
		idToIdx:        make(map[int]int, 1024),
		minHeapPool: sync.Pool{
			New: func() any { h := minHeap(make([]*candidate, 0, 128)); return &h },
		},
		maxHeapPool: sync.Pool{
			New: func() any { h := maxHeap(make([]*candidate, 0, 128)); return &h },
		},
		arenaPool:          sync.Pool{New: func() any { return make([]candidate, 0, 1024) }},
		pruneArenaPool:     sync.Pool{New: func() any { return make([]candidate, 0, 64) }},
		candidateSlicePool: sync.Pool{New: func() any { s := make([]*candidate, 0, 128); return &s }},
		visitedPool:        sync.Pool{New: func() any { return NewVisited(1024) }},
	}
	return g
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

	g.vectorData = append(g.vectorData, vec...)

	node := newNode(id, idx, g.randomLevel(), g.m)
	g.nodes = append(g.nodes, node)
	g.idToIdx[id] = idx

	if idx == 0 {
		atomic.StorePointer(&g.enter, unsafe.Pointer(&g.nodes[0]))
		g.level.Store(int32(node.level))
		return nil
	}

	g.connect(&g.nodes[idx])
	return nil
}

func (g *Graph[T]) AddBatch(items []Item[T]) error {
	for _, item := range items {
		if len(item.Vec) != g.dim {
			return fmt.Errorf("dimension mismatch for item with ID %d", item.ID)
		}
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	if cap(g.vectorData) < len(g.vectorData)+len(items)*g.dim {
		newCap := cap(g.vectorData) + len(items)*g.dim
		newData := make([]T, len(g.vectorData), newCap)
		copy(newData, g.vectorData)
		g.vectorData = newData
	}
	if cap(g.nodes) < len(g.nodes)+len(items) {
		newCap := cap(g.nodes) + len(items)
		newNodes := make([]Node, len(g.nodes), newCap)
		copy(newNodes, g.nodes)
		g.nodes = newNodes
	}

	for _, item := range items {
		idx := len(g.nodes)
		g.vectorData = append(g.vectorData, item.Vec...)
		node := newNode(item.ID, idx, g.randomLevel(), g.m)
		g.nodes = append(g.nodes, node)
		g.idToIdx[item.ID] = idx

		if idx == 0 {
			atomic.StorePointer(&g.enter, unsafe.Pointer(&g.nodes[0]))
			g.level.Store(int32(node.level))
			continue
		}
		g.connect(&g.nodes[idx])
	}
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
		return nil, nil
	}

	visited := g.visitedPool.Get().(*VisitedList)
	if cap(visited.arr) < len(g.nodes) {
		visited.arr = make([]uint32, len(g.nodes))
	}
	defer g.visitedPool.Put(visited)

	res := g.searchInternal(&q[0], k, visited)

	out := make([]int, len(res))
	for i, c := range res {
		out[i] = g.nodes[c.idx].id
	}
	return out, nil
}

//////////////////////
// CORE SEARCH      //
//////////////////////

func (g *Graph[T]) searchInternal(qPtr *T, k int, visited *VisitedList) []*candidate {
	ep := (*Node)(atomic.LoadPointer(&g.enter))
	maxLevel := int(g.level.Load())

	for l := maxLevel; l > 0; l-- {
		ep = g.greedy(qPtr, ep, l)
	}

	cands := g.searchLayer(qPtr, ep, 0, max(g.efSearch, k), visited)

	if len(cands) > k {
		cands = cands[:k]
	}
	return cands
}

func (g *Graph[T]) greedy(qPtr *T, cur *Node, lvl int) *Node {
	best := cur
	bestDist := g.dist(qPtr, g.vecPtr(cur.idx), g.dim)

	for {
		changed := false
		for _, ni := range best.neighbors[lvl] {
			n := &g.nodes[int(ni)]
			d := g.dist(qPtr, g.vecPtr(n.idx), g.dim)
			if d < bestDist {
				bestDist = d
				best = n
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	return best
}

func (g *Graph[T]) searchLayer(qPtr *T, entry *Node, lvl, ef int, visited *VisitedList) []*candidate {
	visited.Reset()

	arena := g.arenaPool.Get().([]candidate)
	arena = arena[:0]
	defer g.arenaPool.Put(arena)

	pq := g.minHeapPool.Get().(*minHeap)
	pq.Reset()
	defer g.minHeapPool.Put(pq)

	res := g.maxHeapPool.Get().(*maxHeap)
	res.Reset()
	defer g.maxHeapPool.Put(res)

	newCand := func(idx int, dist float32) *candidate {
		arena = append(arena, candidate{idx: idx, dist: dist})
		return &arena[len(arena)-1]
	}

	d0 := g.dist(qPtr, g.vecPtr(entry.idx), g.dim)
	c0 := newCand(entry.idx, d0)

	heap.Push(pq, c0)
	heap.Push(res, c0)
	visited.Mark(entry.idx)

	for pq.Len() > 0 {
		c := heap.Pop(pq).(*candidate)

		if res.Len() >= ef && c.dist > (*res)[0].dist {
			break
		}

		for _, ni := range g.nodes[c.idx].neighbors[lvl] {
			i := int(ni)
			if visited.Seen(i) {
				continue
			}
			visited.Mark(i)

			d := g.dist(qPtr, g.vecPtr(i), g.dim)

			if res.Len() < ef || d < (*res)[0].dist {
				cand := newCand(i, d)
				heap.Push(pq, cand)
				heap.Push(res, cand)

				if res.Len() > ef {
					heap.Pop(res)
				}
			}
		}
	}

	outPtr := g.candidateSlicePool.Get().(*[]*candidate)
	out := (*outPtr)[:0]

	for res.Len() > 0 {
		out = append(out, heap.Pop(res).(*candidate))
	}

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
	vectorPtr := g.vecPtr(n.idx)

	for l := maxLevel; l > n.level; l-- {
		cur = g.greedy(vectorPtr, cur, l)
	}

	visited := g.visitedPool.Get().(*VisitedList)
	if cap(visited.arr) < len(g.nodes) {
		visited.arr = make([]uint32, len(g.nodes))
	}
	defer g.visitedPool.Put(visited)

	for l := min(n.level, maxLevel); l >= 0; l-- {
		cands := g.searchLayer(vectorPtr, cur, l, g.efConstruction, visited)

		selected := g.selectNeighbors(cands, g.m)

		for _, c := range selected {
			n.neighbors[l] = append(n.neighbors[l], uint32(c.idx))
		}
		g.prune(n, l)

		for _, c := range selected {
			neighbor := &g.nodes[c.idx]
			neighbor.neighbors[l] = append(neighbor.neighbors[l], uint32(n.idx))
			g.prune(neighbor, l)
		}
	}
}

func (g *Graph[T]) prune(n *Node, l int) {
	maxN := g.m
	if l == 0 {
		maxN *= 2
	}

	if len(n.neighbors[l]) <= maxN {
		return
	}

	cands := g.pruneArenaPool.Get().([]candidate)
	cands = cands[:0]

	vecPtr := g.vecPtr(n.idx)

	for _, ni := range n.neighbors[l] {
		i := int(ni)
		cands = append(cands, candidate{
			idx:  i,
			dist: g.dist(vecPtr, g.vecPtr(i), g.dim),
		})
	}

	sort.Slice(cands, func(i, j int) bool {
		return cands[i].dist < cands[j].dist
	})

	n.neighbors[l] = n.neighbors[l][:0]
	for i := 0; i < maxN && i < len(cands); i++ {
		n.neighbors[l] = append(n.neighbors[l], uint32(cands[i].idx))
	}

	cands = cands[:0]
	g.pruneArenaPool.Put(cands)
}

func (g *Graph[T]) selectNeighbors(cands []*candidate, m int) []*candidate {
	if len(cands) <= m {
		return cands
	}
	return cands[:m]
}

//////////////////////
// UTIL             //
//////////////////////

func (g *Graph[T]) vecPtr(idx int) *T {
	return &g.vectorData[idx*g.dim]
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

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
