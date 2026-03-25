package needle

import (
	"bytes"
	"math/rand"
	"testing"
)

func TestPQCodec(t *testing.T) {
	const ( // Corrected const block
		dim        = 128
		numVectors = 1000
		numSubspaces = 8
		centroidsPerSubspace = 256
	)

	// Generate some random data
	vectors := make([][]float32, numVectors)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
	}

	// Create a new PQ codec
	pq, err := NewPQCodec[float32](dim, numSubspaces, centroidsPerSubspace, l2Float32)
	if err != nil {
		t.Fatalf("failed to create PQ codec: %v", err)
	}

	// Train the codec
	if err := pq.Train(vectors); err != nil {
		t.Fatalf("failed to train PQ codec: %v", err)
	}

	// Test encoding and decoding
	for _, vec := range vectors {
		code := pq.Encode(vec)
		if len(code) != numSubspaces {
			t.Errorf("expected code length %d, got %d", numSubspaces, len(code))
		}

		decodedVec := pq.Decode(code)
		if len(decodedVec) != dim {
			t.Errorf("expected decoded vector length %d, got %d", dim, len(decodedVec))
		}
	}

	// Test asymmetric distance calculation
	query := vectors[0]
	code := pq.Encode(vectors[1])
	dist := pq.Distance(query, code)
	if dist < 0 {
		t.Errorf("expected non-negative distance, got %f", dist)
	}
}

func TestSerialization(t *testing.T) {
	const (
		dim        = 16
		numSubspaces = 4
		centroidsPerSubspace = 16
	)

	// Create a new codec
	pq, err := NewPQCodec[float32](dim, numSubspaces, centroidsPerSubspace, l2Float32)
	if err != nil {
		t.Fatalf("Failed to create codec: %v", err)
	}

	// Create dummy codebooks for serialization testing
	pq.Codebooks = make([][][]float32, numSubspaces)
	for m := 0; m < numSubspaces; m++ {
		pq.Codebooks[m] = make([][]float32, centroidsPerSubspace)
		for k := 0; k < centroidsPerSubspace; k++ {
			pq.Codebooks[m][k] = make([]float32, pq.SubvectorDim)
			for d := 0; d < pq.SubvectorDim; d++ {
				pq.Codebooks[m][k][d] = rand.Float32()
			}
		}
	}

	var buf bytes.Buffer
	if err := pq.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}

	// Create a new codec for deserialization
	pq2, err := NewPQCodec[float32](dim, numSubspaces, centroidsPerSubspace, l2Float32)
	if err != nil {
		t.Fatalf("Failed to create new codec for ReadFrom: %v", err)
	}

	if err := pq2.ReadFrom(&buf); err != nil {
		t.Fatalf("ReadFrom failed: %v", err)
	}

	// Compare the original and deserialized codecs
	if pq.NumSubspaces != pq2.NumSubspaces || pq.CentroidsPerSubspace != pq2.CentroidsPerSubspace || pq.SubvectorDim != pq2.SubvectorDim {
		t.Errorf("Codec parameters do not match after serialization")
	}

	for m := 0; m < pq.NumSubspaces; m++ {
		for k := 0; k < pq.CentroidsPerSubspace; k++ {
			for d := 0; d < pq.SubvectorDim; d++ {
				if pq.Codebooks[m][k][d] != pq2.Codebooks[m][k][d] {
					t.Errorf("Codebook data does not match at [%d][%d][%d]", m, k, d)
				}
			}
		}
	}
}
