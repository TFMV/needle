//go:build !amd64 && !arm64
// +build !amd64,!arm64

package needle

var hasAVX2 = false
func euclideanSquaredAVX2(a, b []float64) float64 { return 0 }

var hasNEON = false
func euclideanSquaredNEON(a, b []float64) float64 { return 0 }
