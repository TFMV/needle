package needle

// l2Float32Scalar computes the L2 distance between two float32 vectors.
func l2Float32Scalar(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
