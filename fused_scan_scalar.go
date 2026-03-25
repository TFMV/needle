package needle

import "container/heap"

func fusedScanScalar(
	q []float32,
	base [][]float32,
	cands []int,
	topk *topK,
	pq *minHeap,
) {
	for _, idx := range cands {
		dist := l2Float32Scalar(q, base[idx])

		if dist > topk.maxD {
			continue
		}

		heap.Push(pq, &candidate{idx: idx, dist: dist})
		topk.Insert(idx, dist)
	}
}
