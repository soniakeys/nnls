package nnls

import "log"
import "github.com/skelterjohn/go.matrix"

func NNLS(x [][]float64, y []float64, n int) ([]float64, error) {
	A := matrix.MakeDenseMatrixStacked(x)
	b := matrix.MakeDenseMatrix(y, len(y), 1)
	β := make([]float64, len(x[0])) // return value
	log.Println("β allocated:", β)
	AT := A.Transpose()
	m, err := AT.TimesDense(b)
	if err != nil {
		return nil, err
	}
	m.Scale(-1)
	μ := m.Array()
	log.Println("initial μ:", μ)
	H, err := AT.TimesDense(A)
	if err != nil {
		return nil, err
	}
	Hd := H.DiagonalCopy() // diagonal as a slice
	log.Println("Hd:", Hd)
	HT := H.Transpose().Arrays() // H columns as slices
	for i := 0; i < n; i++ {
		for k, βk := range β {
			b := βk - μ[k]/Hd[k]
			if b < 0 {
				b = 0
			}
			β[k] = b
			b -= βk
			for i, h := range HT[k] {
				μ[i] += b * h
			}
		}
	}
	return β, nil
}
