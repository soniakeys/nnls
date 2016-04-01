// public domain

package nnls

import "github.com/skelterjohn/go.matrix"

func SCALimit(x [][]float64, y []float64, n int) ([]float64, int, error) {
	A := matrix.MakeDenseMatrixStacked(x)
	b := matrix.MakeDenseMatrix(y, len(y), 1)
	AT := A.Transpose()
	m, err := AT.TimesDense(b)
	if err != nil {
		return nil, 0, err
	}
	m.Scale(-1)
	μ := m.Array()
	H, _ := AT.TimesDense(A)
	Hd := H.DiagonalCopy()
	Ha := H.Arrays()                // (H is symmetric)
	β := make([]float64, len(x[0])) // return value
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

func SCAKKT(x [][]float64, y []float64, ε float64) ([]float64, int, error) {
	A := matrix.MakeDenseMatrixStacked(x)
	b := matrix.MakeDenseMatrix(y, len(y), 1)
	AT := A.Transpose()
	m, err := AT.TimesDense(b)
	if err != nil {
		return nil, 0, err
	}
	m.Scale(-1)
	μ := m.Array()
	H, _ := AT.TimesDense(A)
	Hd := H.DiagonalCopy()
	Ha := H.Arrays()                // (H is symmetric)
	β := make([]float64, len(x[0])) // return value
	nε := -ε
	i := 0
i:
	for ; ; i++ {
		for k, βk := range β {
			b := βk - μ[k]/Hd[k]
			if b < 0 {
				b = 0
			}
			β[k] = b
			b -= βk
			if b == 0 {
				continue
			}
			for j, h := range Ha[k] {
				μ[j] += b * h
			}
		}
		for k, m := range μ {
			if m < nε {
				continue i
			}
			if β[k] > 0 && m > ε {
				continue i
			}
		}
		break
	}
	return β, i + 1, nil
}
