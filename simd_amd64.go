//go:build amd64
// +build amd64

package needle

import "golang.org/x/sys/cpu"

var hasAVX2 = cpu.X86.HasAVX2

var hasNEON = false
func euclideanSquaredNEON(a, b []float64) float64 { return 0 }
