//go:build amd64 || arm64

package needle

// l2Float32AVX2 computes the L2 distance between two float32 vectors using AVX2 intrinsics.
func l2Float32AVX2(a, b []float32) float32 {
	// This is a placeholder implementation. A real implementation would use AVX2 intrinsics.
	var sum float32
	for i := 0; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// l2AVX2Supported returns true if the CPU supports AVX2.
func l2AVX2Supported() bool {
	// This is a placeholder implementation. A real implementation would check CPU features.
	return true
}
