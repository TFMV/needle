package needle

import "unsafe"

// l2Float32Scalar computes the L2 distance between two float32 vectors using a scalar implementation.
// This is the fallback when AVX2 is not available.
func l2Float32Scalar(aPtr, bPtr *float32, dim int) float32 {
	var sum float32
	i := 0

	// Unrolled loop to process 8 elements at a time.
	for ; i+7 < dim; i += 8 {
		a0 := *(*float32)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i)*4))
		b0 := *(*float32)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i)*4))

		a1 := *(*float32)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i+1)*4))
		b1 := *(*float32)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i+1)*4))

		a2 := *(*float32)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i+2)*4))
		b2 := *(*float32)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i+2)*4))

		a3 := *(*float32)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i+3)*4))
		b3 := *(*float32)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i+3)*4))

		a4 := *(*float32)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i+4)*4))
		b4 := *(*float32)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i+4)*4))

		a5 := *(*float32)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i+5)*4))
		b5 := *(*float32)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i+5)*4))

		a6 := *(*float32)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i+6)*4))
		b6 := *(*float32)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i+6)*4))

		a7 := *(*float32)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i+7)*4))
		b7 := *(*float32)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i+7)*4))

		d0 := a0 - b0
		d1 := a1 - b1
		d2 := a2 - b2
		d3 := a3 - b3
		d4 := a4 - b4
		d5 := a5 - b5
		d6 := a6 - b6
		d7 := a7 - b7

		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 +
			d4*d4 + d5*d5 + d6*d6 + d7*d7
	}

	// Handle the remainder
	for ; i < dim; i++ {
		a := *(*float32)(unsafe.Add(unsafe.Pointer(aPtr), uintptr(i)*4))
		b := *(*float32)(unsafe.Add(unsafe.Pointer(bPtr), uintptr(i)*4))
		d := a - b
		sum += d * d
	}

	return sum
}
