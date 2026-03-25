package needle

import "math"

// topK stores the top-k candidates (idx and distance) in sorted order.
// It replaces a max-heap for collecting results and is optimized for small k.
type topK struct {
	idx  []int
	dist []float32
	k    int
	size int
	maxD float32
}

// newTopK creates a new topK structure with capacity k.
func newTopK(k int) *topK {
	return &topK{
		idx:  make([]int, k),
		dist: make([]float32, k),
		k:    k,
		size: 0,
		maxD: math.MaxFloat32,
	}
}

// isFull returns true if the structure is at capacity.
func (tk *topK) isFull() bool {
	return tk.size == tk.k
}

// Insert adds a new candidate. If the structure is full and the candidate's
// distance is greater than the worst distance, it is ignored. Otherwise, it is
// inserted in sorted order, potentially evicting the current worst candidate.
func (tk *topK) Insert(idx int, dist float32) {
	if tk.isFull() && dist >= tk.maxD {
		return
	}

	i := tk.size
	if tk.isFull() {
		i = tk.k - 1
	}

	for i > 0 && tk.dist[i-1] > dist {
		i--
	}

	if i >= tk.k {
		return
	}

	copy(tk.idx[i+1:], tk.idx[i:tk.k-1])
	copy(tk.dist[i+1:], tk.dist[i:tk.k-1])

	tk.idx[i] = idx
	tk.dist[i] = dist

	if !tk.isFull() {
		tk.size++
	}

	if tk.isFull() {
		tk.maxD = tk.dist[tk.k-1]
	}
}

// Reset clears the structure for reuse.
func (tk *topK) Reset() {
	tk.size = 0
	tk.maxD = math.MaxFloat32
}
