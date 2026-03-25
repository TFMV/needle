package needle

import "container/heap"

// fusedScanAVX2 is a placeholder for the AVX2 implementation of fused scan.
func fusedScanAVX2(
	q []float32,
	base [][]float32,
	cands []int,
	topk *topK,
	pq *minHeap,
) {
	for _, idx := range cands {
		dist := l2Float32AVX2(q, base[idx])

		if dist > topk.maxD {
			continue
		}

		heap.Push(pq, &candidate{idx: idx, dist: dist})
		topk.Insert(idx, dist)
	}
}
