package nnls

import "github.com/skelterjohn/go.matrix"

func NNLS(x [][]float64, y []float64, n int) ([]float64, int, error) {
	A := matrix.MakeDenseMatrixStacked(x)
	b := matrix.MakeDenseMatrix(y, len(y), 1)
	β := make([]float64, len(x[0])) // return value
	AT := A.Transpose()
	m, err := AT.TimesDense(b)
	if err != nil {
		return nil, 0, err
	}
	m.Scale(-1)
	μ := m.Array()
	H, err := AT.TimesDense(A)
	if err != nil {
		return nil, 0, err
	}
	Hd := H.DiagonalCopy()
	Ha := H.Arrays() // (H is symmetric)
	if n < 0 {
		n = 10000
	}
	for i := 0; i < n; i++ {
		ch := false
		for k, βk := range β {
			b := βk - μ[k]/Hd[k]
			if b < 0 {
				b = 0
			}
			if b != βk {
				β[k] = b
				ch = true
			}
			b -= βk
			for j, h := range Ha[k] {
				μ[j] += b * h
			}
		}
		if !ch {
			n = i + 1
			break
		}
	}
	return β, n, nil
}
