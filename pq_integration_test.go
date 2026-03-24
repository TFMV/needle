package needle

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPQIntegration(t *testing.T) {
	const (
		dim         = 128
		pqThreshold = 500
		numVectors  = 1000
	)

	config := DefaultConfig()
	config.dim = dim
	config.pqThreshold = pqThreshold
	g := NewGraphFromConfig[float32](config)

	// Add initial vectors to trigger PQ training
	for i := 0; i < pqThreshold; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		g.Add(i, vec)
	}

	assert.True(t, g.pqTrained, "PQ codec should be trained")

	// Add more vectors after PQ is trained
	for i := pqThreshold; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		g.Add(i, vec)
	}

	// Search for a random vector
	query := make([]float32, dim)
	for j := range query {
		query[j] = rand.Float32()
	}

	res, err := g.Search(query, 5)

	assert.NoError(t, err)
	assert.NotNil(t, res)
	assert.NotEmpty(t, res, "Search should return some results")
}
