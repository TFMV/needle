package needle

import (
	"math/rand"
	"testing"
)

const delta = 1e-4

func TestL2Distance(t *testing.T) {
	for _, dim := range []int{1, 8, 16, 32, 128, 256, 768} {
		t.Run("l2Float32Scalar", func(t *testing.T) {
			a := make([]float32, dim)
			b := make([]float32, dim)
			for i := 0; i < dim; i++ {
				a[i] = rand.Float32()
				b[i] = rand.Float32()
			}

			expected := l2Float32Scalar(&a[0], &b[0], dim)
			got := l2Float32Ptr(&a[0], &b[0], dim)

			if (expected-got)*(expected-got) > delta {
				t.Errorf("Expected %f, got %f", expected, got)
			}
		})

		if useAVX2 {
			t.Run("l2Float32AVX2", func(t *testing.T) {
				a := make([]float32, dim)
				b := make([]float32, dim)
				for i := 0; i < dim; i++ {
					a[i] = rand.Float32()
					b[i] = rand.Float32()
				}

				expected := l2Float32Scalar(&a[0], &b[0], dim)
				got := l2Float32AVX2(&a[0], &b[0], dim)

				if (expected-got)*(expected-got) > delta {
					t.Errorf("Expected %f, got %f", expected, got)
				}
			})
		}
	}
}

func BenchmarkL2Distance(b *testing.B) {
	for _, dim := range []int{128, 256, 768} {
		a := make([]float32, dim)
		b_ := make([]float32, dim)
		for i := 0; i < dim; i++ {
			a[i] = rand.Float32()
			b_[i] = rand.Float32()
		}

		b.Run("l2Float32Scalar", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				l2Float32Scalar(&a[0], &b_[0], dim)
			}
		})

		if useAVX2 {
			b.Run("l2Float32AVX2", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					l2Float32AVX2(&a[0], &b_[0], dim)
				}
			})
		}
	}
}
