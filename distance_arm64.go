//go:build arm64
// +build arm64

package needle

// euclideanSquaredNEON utilizes Go's native ARM64 ASIMD auto-vectorization.
// By completely unrolling the loop into 8-wide scalar bounds, the Go 1.20+
// compiler natively compiles this into VFMLA and FADD.
func euclideanSquaredNEON(a, b []float64) float64 {
	var sum float64
	i := 0
	for i <= len(a)-8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]

		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 +
			d4*d4 + d5*d5 + d6*d6 + d7*d7
		i += 8
	}

	for i < len(a) {
		d := a[i] - b[i]
		sum += d * d
		i++
	}
	return sum
}
