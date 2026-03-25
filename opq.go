package needle

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// OPQCodec implements Optimized Product Quantization (OPQ).
// OPQ is an enhancement of Product Quantization (PQ) that jointly optimizes
// the rotation of the vector space and the quantization codebooks, leading to
// lower distortion and improved search accuracy.
type OPQCodec[T Float] struct {
	NumSubspaces         int         // Number of subspaces (M)
	CentroidsPerSubspace int         // Number of centroids per subspace (K)
	SubvectorDim         int         // Dimension of each subvector (D/M)
	Codebooks            [][][]T     // Codebooks for each subspace, shape: [M][K][D/M]
	Rotation             *mat.Dense  // The learned rotation matrix (R)
	dist                 DistanceFunc[T] // Distance function to use
}

// NewOPQCodec creates a new OPQCodec.
//
// Parameters:
//   - dim: The dimension of the vectors to be encoded.
//   - numSubspaces: The number of subspaces to divide the vectors into.
//   - centroidsPerSubspace: The number of centroids to use for each subspace.
//   - dist: The distance function to use for clustering and search.
func NewOPQCodec[T Float](dim, numSubspaces, centroidsPerSubspace int, dist DistanceFunc[T]) (*OPQCodec[T], error) {
	if dim%numSubspaces != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by numSubspaces %d", dim, numSubspaces)
	}
	return &OPQCodec[T]{
		NumSubspaces:         numSubspaces,
		CentroidsPerSubspace: centroidsPerSubspace,
		SubvectorDim:         dim / numSubspaces,
		dist:                 dist,
	}, nil
}

// Train trains the OPQ codec.
// The training process iteratively optimizes a rotation matrix (R) and the PQ codebooks.
// The goal is to rotate the input vectors such that the quantization error after applying PQ is minimized.
// The optimization proceeds as follows:
//  1. Rotate the input vectors using the current rotation matrix R.
//  2. Train a standard PQ model on the rotated vectors.
//  3. Reconstruct the quantized rotated vectors.
//  4. Find the optimal rotation R' that minimizes the error between the original vectors and the quantized rotated vectors.
//     This is solved using the Orthogonal Procrustes problem, which can be solved with Singular Value Decomposition (SVD).
//  5. Update R with R' and repeat until convergence.
func (opq *OPQCodec[T]) Train(vectors [][]T) error {
	numVectors := len(vectors)
	if numVectors == 0 {
		return fmt.Errorf("training set is empty")
	}
	dim := len(vectors[0])

	// Initialize with a random rotation matrix
	r, err := randomRotationMatrix(dim)
	if err != nil {
		return err
	}
	opq.Rotation = r

	// Create a PQ codec for sub-training
	pqCodec, err := NewPQCodec[T](dim, opq.NumSubspaces, opq.CentroidsPerSubspace, opq.dist)
	if err != nil {
		return err
	}

	// Pre-allocate matrices to avoid re-allocation in each iteration
	newR := mat.NewDense(dim, dim, nil)

	// Iteratively optimize the rotation and codebooks
	const maxIterations = 100
	const convergenceThreshold = 1e-5
	for i := 0; i < maxIterations; i++ {
		// 1. Rotate the training vectors
		rotatedVectors := make([][]T, numVectors)
		for j, v := range vectors {
			rotatedVectors[j] = applyRotation(v, opq.Rotation)
		}

		// 2. Train a standard PQ codebook on the rotated vectors
		if err := pqCodec.Train(rotatedVectors); err != nil {
			return err
		}

		// 3. Update the rotation matrix
		quantizedRotatedVectors := make([][]T, numVectors)
		for j, rv := range rotatedVectors {
			code := pqCodec.Encode(rv)
			quantizedRotatedVectors[j] = pqCodec.Decode(code)
		}

		X := mat.NewDense(numVectors, dim, vectorsToFloat64s(vectors))
		Y := mat.NewDense(numVectors, dim, vectorsToFloat64s(quantizedRotatedVectors))

		var C mat.Dense
		C.Mul(X.T(), Y)

		var svd mat.SVD
		if ok := svd.Factorize(&C, mat.SVDThin); !ok {
			return fmt.Errorf("SVD factorization failed")
		}

		var u, v mat.Dense
		svd.UTo(&u)
		svd.VTo(&v)

		newR.Mul(&v, u.T())

		// Check for convergence
		diff := mat.NewDense(dim, dim, nil)
		diff.Sub(newR, opq.Rotation)
		change := mat.Norm(diff, 2) // Frobenius norm
		if change < convergenceThreshold {
			break // Converged
		}

		opq.Rotation.Copy(newR)
	}

	opq.Codebooks = pqCodec.Codebooks

	return nil
}


// Encode compresses a vector into an OPQ code.
func (opq *OPQCodec[T]) Encode(vec []T) []byte {
	rotatedVec := applyRotation(vec, opq.Rotation)

	code := make([]byte, opq.NumSubspaces)
	for i := 0; i < opq.NumSubspaces; i++ {
		start := i * opq.SubvectorDim
		end := start + opq.SubvectorDim
		subVec := rotatedVec[start:end]

		bestCentroid := 0
		minDist := float32(math.MaxFloat32)

		for j, centroid := range opq.Codebooks[i] {
			dist := opq.dist(&subVec[0], &centroid[0], opq.SubvectorDim)
			if dist < minDist {
				minDist = dist
				bestCentroid = j
			}
		}
		code[i] = byte(bestCentroid)
	}
	return code
}

// Decode decompresses an OPQ code back into a vector.
func (opq *OPQCodec[T]) Decode(code []byte) []T {
	// 1. Reconstruct the quantized, rotated vector from the codebook
	quantizedRotatedVec := make([]T, opq.NumSubspaces*opq.SubvectorDim)
	for i, centroidID := range code {
		centroid := opq.Codebooks[i][int(centroidID)]
		start := i * opq.SubvectorDim
		copy(quantizedRotatedVec[start:], centroid)
	}

	// 2. Apply the inverse rotation (transpose) to get back to the original space
	vec := mat.NewVecDense(len(quantizedRotatedVec), f64SliceToFloat64(quantizedRotatedVec))
	var res mat.VecDense
	res.MulVec(opq.Rotation.T(), vec)

	return float64SliceToFloat[T](res.RawVector().Data)
}

// Distance calculates the asymmetric distance between a query vector and a PQ code.
func (opq *OPQCodec[T]) Distance(query []T, code []byte) float32 {
	// 1. Rotate the query vector
	rotatedQuery := applyRotation(query, opq.Rotation)

	// 2. Calculate distance from the rotated query to the stored code
	var totalDist float32
	for i := 0; i < opq.NumSubspaces; i++ {
		start := i * opq.SubvectorDim
		end := start + opq.SubvectorDim
		subVec := rotatedQuery[start:end]
		centroidID := int(code[i])
		centroid := opq.Codebooks[i][centroidID]
		totalDist += opq.dist(&subVec[0], &centroid[0], opq.SubvectorDim)
	}
	return totalDist
}

// randomRotationMatrix creates a random DxD rotation matrix.
func randomRotationMatrix(dim int) (*mat.Dense, error) {
	a := mat.NewDense(dim, dim, nil)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			a.Set(i, j, rand.NormFloat64())
		}
	}

	var qr mat.QR
	qr.Factorize(a)

	var q mat.Dense
	qr.QTo(&q)

	return &q, nil
}

// applyRotation rotates a vector by the given rotation matrix.
func applyRotation[T Float](v []T, r *mat.Dense) []T {
	vec := mat.NewVecDense(len(v), f64SliceToFloat64(v))
	var res mat.VecDense
	res.MulVec(r, vec)
	return float64SliceToFloat[T](res.RawVector().Data)
}

// Helper functions for type conversion

func vectorsToFloat64s[T Float](vectors [][]T) []float64 {
	if len(vectors) == 0 {
		return nil
	}
	numVectors := len(vectors)
	dim := len(vectors[0])
	data := make([]float64, numVectors*dim)
	for i := 0; i < numVectors; i++ {
		for j := 0; j < dim; j++ {
			data[i*dim+j] = float64(vectors[i][j])
		}
	}
	return data
}

func f64SliceToFloat64[T Float](s []T) []float64 {
	f64s := make([]float64, len(s))
	for i, v := range s {
		f64s[i] = float64(v)
	}
	return f64s
}

func float64SliceToFloat[T Float](s []float64) []T {
	ts := make([]T, len(s))
	for i, v := range s {
		ts[i] = T(v)
	}
	return ts
}
