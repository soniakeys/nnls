package nnls_test

import (
	"fmt"

	"github.com/soniakeys/nnls"
)

func ExampleNNLS() {
	// Wikipedia example data
	height := []float64{1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63,
		1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83}
	weight := []float64{52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93,
		61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46}
	A := make([][]float64, len(height))
	for i, h := range height {
		A[i] = []float64{h * h, h, 1}
	}
	fmt.Println(nnls.NNLS(A, weight, 7000))
	// Output:
}
