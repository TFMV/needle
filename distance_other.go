//go:build !amd64 && !arm64

package needle

func L2Float32SIMD(a, b []float32) float32 {
	return L2Float32(a, b)
}
