package needle

func fusedScan(q []float32, base [][]float32, cands []int, topk *topK, pq *minHeap) {
	if useAVX2 {
		fusedScanAVX2(q, base, cands, topk, pq)
	} else {
		fusedScanScalar(q, base, cands, topk, pq)
	}
}
