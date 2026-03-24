package needle

import (
	"fmt"
	"math"
	"math/rand"
)

// PQCodec implements Product Quantization for vector compression.
type PQCodec[T Float] struct {
	NumSubspaces         int
	CentroidsPerSubspace int
	SubvectorDim         int
	Codebooks            [][][]T
	dist                 DistanceFunc[T]
}

// NewPQCodec creates a new PQCodec.
func NewPQCodec[T Float](dim, numSubspaces, centroidsPerSubspace int, dist DistanceFunc[T]) (*PQCodec[T], error) {
	if dim%numSubspaces != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by numSubspaces %d", dim, numSubspaces)
	}
	return &PQCodec[T]{
		NumSubspaces:         numSubspaces,
		CentroidsPerSubspace: centroidsPerSubspace,
		SubvectorDim:         dim / numSubspaces,
		dist:                 dist,
	}, nil
}

// Train trains the PQ codec on a set of vectors using a naive k-means implementation.
func (pq *PQCodec[T]) Train(vectors [][]T) error {
	pq.Codebooks = make([][][]T, pq.NumSubspaces)
	for i := 0; i < pq.NumSubspaces; i++ {
		// 1. Prepare data for this subspace
		subspaceData := make([][]T, len(vectors))
		for j, vec := range vectors {
			start := i * pq.SubvectorDim
			end := start + pq.SubvectorDim
			subspaceData[j] = vec[start:end]
		}

		// 2. Run naive k-means
		centroids := pq.naiveKMeans(subspaceData)

		// 3. Store centroids
		pq.Codebooks[i] = centroids
	}
	return nil
}

// naiveKMeans implements a simple k-means clustering algorithm.
func (pq *PQCodec[T]) naiveKMeans(data [][]T) [][]T {
	// 1. Initialize centroids randomly from the data points
	centroids := make([][]T, pq.CentroidsPerSubspace)
	perm := rand.Perm(len(data))
	for i := 0; i < pq.CentroidsPerSubspace; i++ {
		centroids[i] = make([]T, pq.SubvectorDim)
		copy(centroids[i], data[perm[i]])
	}

	for iter := 0; iter < 100; iter++ { // Fixed number of iterations
		// 2. Assign each point to the nearest centroid
		assignments := make([][]int, pq.CentroidsPerSubspace)
		for pointIdx, point := range data {
			minDist := float32(math.MaxFloat32)
			bestCentroid := 0
			for centroidIdx, centroid := range centroids {
				dist := pq.dist(point, centroid)
				if dist < minDist {
					minDist = dist
					bestCentroid = centroidIdx
				}
			}
			assignments[bestCentroid] = append(assignments[bestCentroid], pointIdx)
		}

		// 3. Re-calculate centroids
		for i, assignedPoints := range assignments {
			if len(assignedPoints) == 0 {
				continue // Keep old centroid if no points are assigned
			}
			newCentroid := make([]T, pq.SubvectorDim)
			for _, pointIdx := range assignedPoints {
				for dim := 0; dim < pq.SubvectorDim; dim++ {
					newCentroid[dim] += data[pointIdx][dim]
				}
			}
			for dim := 0; dim < pq.SubvectorDim; dim++ {
				newCentroid[dim] /= T(len(assignedPoints))
			}
			centroids[i] = newCentroid
		}
	}

	return centroids
}

// Encode compresses a vector into a PQ code.
func (pq *PQCodec[T]) Encode(vec []T) []byte {
	code := make([]byte, pq.NumSubspaces)
	for i := 0; i < pq.NumSubspaces; i++ {
		start := i * pq.SubvectorDim
		end := start + pq.SubvectorDim
		subVec := vec[start:end]

		bestCentroid := 0
		minDist := float32(math.MaxFloat32)

		for j, centroid := range pq.Codebooks[i] {
			dist := pq.dist(subVec, centroid)
			if dist < minDist {
				minDist = dist
				bestCentroid = j
			}
		}
		code[i] = byte(bestCentroid)
	}
	return code
}

// Decode decompresses a PQ code into a vector.
func (pq *PQCodec[T]) Decode(code []byte) []T {
	vec := make([]T, pq.NumSubspaces*pq.SubvectorDim)
	for i, centroidID := range code {
		centroid := pq.Codebooks[i][int(centroidID)]
		start := i * pq.SubvectorDim
		copy(vec[start:], centroid)
	}
	return vec
}

// Distance calculates the asymmetric distance between a query vector and a PQ code.
func (pq *PQCodec[T]) Distance(query []T, code []byte) float32 {
	var totalDist float32
	for i := 0; i < pq.NumSubspaces; i++ {
		start := i * pq.SubvectorDim
		end := start + pq.SubvectorDim
		subVec := query[start:end]
		centroidID := int(code[i])
		centroid := pq.Codebooks[i][centroidID]
		totalDist += pq.dist(subVec, centroid)
	}
	return totalDist
}
