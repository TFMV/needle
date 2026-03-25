package needle

import (
	"fmt"
	"math"
	"math/rand"
)

// PQCodec implements Product Quantization for vector compression.
// It divides vectors into subspaces and quantizes each subspace individually.
type PQCodec[T Float] struct {
	NumSubspaces         int      // Number of subspaces (M)
	CentroidsPerSubspace int      // Number of centroids per subspace (K)
	SubvectorDim         int      // Dimension of each subvector (D/M)
	Codebooks            [][][]T  // Codebooks for each subspace, shape: [M][K][D/M]
	dist                 DistanceFunc[T] // Distance function to use
}

// NewPQCodec creates a new PQCodec.
//
// Parameters:
//   - dim: The dimension of the vectors to be encoded.
//   - numSubspaces: The number of subspaces to divide the vectors into.
//   - centroidsPerSubspace: The number of centroids to use for each subspace.
//   - dist: The distance function to use for clustering and search.
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

// Train trains the PQ codec on a set of vectors.
// It runs a separate k-means clustering for each subspace to generate the codebooks.
func (pq *PQCodec[T]) Train(vectors [][]T) error {
	pq.Codebooks = make([][][]T, pq.NumSubspaces)
	for i := 0; i < pq.NumSubspaces; i++ {
		// 1. Prepare data for this subspace by extracting the relevant subvectors.
		subspaceData := make([][]T, len(vectors))
		for j, vec := range vectors {
			start := i * pq.SubvectorDim
			end := start + pq.SubvectorDim
			subspaceData[j] = vec[start:end]
		}

		// 2. Run k-means clustering on the subspace data to find the centroids.
		centroids := pq.kmeans(subspaceData)

		// 3. Store the resulting centroids in the codebook for this subspace.
		pq.Codebooks[i] = centroids
	}
	return nil
}

// kmeans implements k-means clustering with k-means++ initialization and handling of empty clusters.
func (pq *PQCodec[T]) kmeans(data [][]T) [][]T {
	if len(data) == 0 {
		return nil
	}

	// 1. Initialize centroids using k-means++
	centroids := make([][]T, pq.CentroidsPerSubspace)
	centroids[0] = make([]T, pq.SubvectorDim)
	copy(centroids[0], data[rand.Intn(len(data))])

	dists := make([]float32, len(data))

	for i := 1; i < pq.CentroidsPerSubspace; i++ {
		totalDist := float32(0)
		for j, point := range data {
			minDistToCentroid := float32(math.MaxFloat32)
			for k := 0; k < i; k++ {
				d := pq.dist(&point[0], &centroids[k][0], pq.SubvectorDim)
				if d < minDistToCentroid {
					minDistToCentroid = d
				}
			}
			dists[j] = minDistToCentroid * minDistToCentroid
			totalDist += dists[j]
		}

		target := rand.Float32() * totalDist
		cumulativeDist := float32(0)
		for j, d := range dists {
			cumulativeDist += d
			if cumulativeDist >= target && centroids[i] == nil {
				centroids[i] = make([]T, pq.SubvectorDim)
				copy(centroids[i], data[j])
			}
		}
	}
    // In case any centroids were not initialized
    for i := range centroids {
        if centroids[i] == nil {
            centroids[i] = make([]T, pq.SubvectorDim)
            copy(centroids[i], data[rand.Intn(len(data))])
        }
    }


	const maxIterations = 100
	const convergenceThreshold = 1e-4

	for iter := 0; iter < maxIterations; iter++ {
		assignments := make([][]int, pq.CentroidsPerSubspace)
        // Keep track of the point with the largest min_dist to any centroid
        // This will be a candidate for re-seeding an empty cluster
		farthestPointIdx := -1
		maxDist := float32(-1.0)


		for pointIdx, point := range data {
			minDist := float32(math.MaxFloat32)
			bestCentroid := 0
			for centroidIdx, centroid := range centroids {
				dist := pq.dist(&point[0], &centroid[0], pq.SubvectorDim)
				if dist < minDist {
					minDist = dist
					bestCentroid = centroidIdx
				}
			}
			assignments[bestCentroid] = append(assignments[bestCentroid], pointIdx)

            if minDist > maxDist {
                maxDist = minDist
                farthestPointIdx = pointIdx
            }
		}

		oldCentroids := make([][]T, len(centroids))
		for i, c := range centroids {
			oldCentroids[i] = make([]T, len(c))
			copy(oldCentroids[i], c)
		}

		for i, assignedPoints := range assignments {
			if len(assignedPoints) == 0 {
				if farthestPointIdx != -1 {
                    // Re-seed empty cluster with the farthest point
					centroids[i] = make([]T, pq.SubvectorDim)
					copy(centroids[i], data[farthestPointIdx])
                    // Invalidate this point so it's not used again for re-seeding in this iteration
                    farthestPointIdx = -1
				} else {
                    // Fallback to random point if farthest point is not available
                    centroids[i] = make([]T, pq.SubvectorDim)
				    copy(centroids[i], data[rand.Intn(len(data))])
                }

				continue
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

		var totalMovement float32
		for i := range centroids {
			totalMovement += pq.dist(&oldCentroids[i][0], &centroids[i][0], pq.SubvectorDim)
		}

		if totalMovement/float32(pq.CentroidsPerSubspace) < convergenceThreshold {
			break
		}
	}

	return centroids
}


// Encode compresses a vector into a PQ code.
// It finds the closest centroid in each subspace's codebook.
func (pq *PQCodec[T]) Encode(vec []T) []byte {
	code := make([]byte, pq.NumSubspaces)
	for i := 0; i < pq.NumSubspaces; i++ {
		start := i * pq.SubvectorDim
		end := start + pq.SubvectorDim
		subVec := vec[start:end]

		bestCentroid := 0
		minDist := float32(math.MaxFloat32)

		for j, centroid := range pq.Codebooks[i] {
			dist := pq.dist(&subVec[0], &centroid[0], pq.SubvectorDim)
			if dist < minDist {
				minDist = dist
				bestCentroid = j
			}
		}
		code[i] = byte(bestCentroid)
	}
	return code
}

// Decode decompresses a PQ code back into a vector.
// It reconstructs the vector by looking up the centroids corresponding to the code.
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
// This is faster than decoding the vector and then computing the distance.
func (pq *PQCodec[T]) Distance(query []T, code []byte) float32 {
	var totalDist float32
	for i := 0; i < pq.NumSubspaces; i++ {
		start := i * pq.SubvectorDim
		end := start + pq.SubvectorDim
		subVec := query[start:end]
		centroidID := int(code[i])
		if centroidID >= len(pq.Codebooks[i]) {
			// This can happen if the codec isn't trained or data is corrupted
			// Handle this gracefully
			continue
		}
		centroid := pq.Codebooks[i][centroidID]
		totalDist += pq.dist(&subVec[0], &centroid[0], pq.SubvectorDim)
	}
	return totalDist
}
