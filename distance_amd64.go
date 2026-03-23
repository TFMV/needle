//go:build amd64
// +build amd64

package needle

// euclideanSquaredAVX2 computes euclidean squared distance using AVX2 instructions
//go:noescape
func euclideanSquaredAVX2(a, b []float64) float64
