package needle

import "golang.org/x/sys/cpu"

// useAVX2 is a package-level variable that indicates if AVX2 is available.
var useAVX2 = detectAVX2()

// detectAVX2 checks for AVX2 support.
func detectAVX2() bool {
	return cpu.X86.HasAVX2
}
