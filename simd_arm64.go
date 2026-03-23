//go:build arm64
// +build arm64

package needle

import "golang.org/x/sys/cpu"

var hasNEON = cpu.ARM64.HasASIMD

var hasAVX2 = false
func euclideanSquaredAVX2(a, b []float64) float64 { return 0 }
