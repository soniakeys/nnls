// public domain

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
		A[i] = []float64{h * h, h, 1} // Vandermonde
	}
	β, n, _ := nnls.NNLS(A, weight, -1)
	fmt.Printf("coefficents:  %.2f\n", β)
	fmt.Println("iterations:  ", n)
	fmt.Println("measured  modeled    error")
	for i, h := range height {
		m := h*(h*β[0]+β[1]) + β[2]
		w := weight[i]
		fmt.Printf("%8.2f %8.2f %8.2f\n", w, m, w-m)
	}
	// Output:
	// coefficents:  [18.61 0.00 11.15]
	// iterations:   7343
	// measured  modeled    error
	//    52.21    51.36     0.85
	//    53.12    53.02     0.10
	//    54.48    54.14     0.34
	//    55.84    55.86    -0.02
	//    57.20    57.02     0.18
	//    58.57    58.79    -0.22
	//    59.93    60.59    -0.66
	//    61.29    61.81    -0.52
	//    63.11    63.67    -0.56
	//    64.47    64.93    -0.46
	//    66.28    66.84    -0.56
	//    68.10    68.14    -0.04
	//    69.92    70.11    -0.19
	//    72.19    71.44     0.75
	//    74.46    73.47     0.99
}
