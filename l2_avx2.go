package needle

//go:noescape

// l2Float32AVX2 computes the L2 distance between two float32 vectors using AVX2 instructions.
func l2Float32AVX2(a, b *float32, dim int) float32
