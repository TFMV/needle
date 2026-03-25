package needle

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPQCodec(t *testing.T) {
	const (
		dim                  = 128
		numSubspaces         = 8
		centroidsPerSubspace = 256
		numVectors           = 1000
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
	pq, err := NewPQCodec[float32](dim, numSubspaces, centroidsPerSubspace, l2Float32Ptr)
	assert.NoError(t, err)

	// Train the codec
	err = pq.Train(vectors)
	assert.NoError(t, err)

	// Encode and decode a vector
	vec := vectors[0]
	code := pq.Encode(vec)
	decodedVec := pq.Decode(code)

	// Check that the decoded vector is not nil
	assert.NotNil(t, decodedVec)

	// Check that the decoded vector has the correct dimension
	assert.Equal(t, dim, len(decodedVec))

	// Check that the distance between the.original and decoded vector is reasonable
	assert.InDelta(t, 0, l2Float32Ptr(&vec[0], &decodedVec[0], dim), 5.0)
}
