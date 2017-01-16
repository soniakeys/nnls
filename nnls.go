// public domain

// NNLS solves the non-negative least squares problem.
//
// The SCA functions implement algorithms described in the paper "Sequential
// Coordinate-wise Algorithm for the Non-negative Least Squares Problem" by
// Vojtěch Franc, Václav Hlaváč, and Mirko Navara, Technical report, Department
// of Cybernetics, Czech Technical University, 2005, accessed at
// http://cmp.felk.cvut.cz/ftp/articles/franc/Franc-Hlavac-Navara-CAIP05.pdf.
//
// Three variants are presented, all use the same math for iteratively
// converging on a solution and all have in common some absolute stopping
// critia.  Iteration will stop if an absolute limit is reached in number
// of interations or if there is no change in the result from one iteration
// to the next.
package nnls

import "errors"

// Limit is an absolute limit on number of iterations.
var Limit int = 1e7

// SCA solves the non-negative least squares problem.
//
// Argument A represents a design or input matrix.  Each element of A must
// have the same length.  b represents a measurement or output vector.
// δ is a tolerance for the result.
//
// Iteration will stop when the least squares objective is within δ of
// an optimal solution.
//
// The result x returns coefficients of the fitted linear function, it
// will have the same length as elements of A.   Result i is the number of
// iterations performed.  An error is returned if A and b are not the same
// length.
func SCA(A [][]float64, b []float64, δ float64) (x []float64, i int, err error) {
	// A, b, m, n are as introduced in section 2.
	m := len(A)
	n := len(A[0])
	if len(b) != m {
		return nil, 0, errors.New("A, b must be same length")
	}

	// Under section 2, formula 4:
	// f = −Aᵀ*b ∈ ℝ ⁿ
	f := make([]float64, n)
	for j := range f {
		e := 0.
		for i, bi := range b {
			e -= bi * A[i][j]
		}
		f[j] = e
	}
	// H = Aᵀ*A ∈ ℝ ⁿⁿ
	H := make([][]float64, n)
	Hd := make([]float64, n) // and make a copy of the diagonal
	for i := range H {
		Hi := make([]float64, n)
		for j := 0; j < i; j++ {
			Hi[j] = H[j][i]
		}
		s := 0.
		for k := range b {
			e := A[k][i]
			s += e * e
		}
		Hi[i] = s
		Hd[i] = s
		for j := i + 1; j < n; j++ {
			s := 0.
			for k := range b {
				s += A[k][i] * A[k][j]
			}
			Hi[j] = s
		}
		H[i] = Hi
	}

	// Algorithm 1, step 1, initialization.
	x = make([]float64, n) // this will be the return value
	μ := append([]float64{}, f...)

	// compute ub = ⟨x⁰, e⟩, an upper bound on the result x, used for
	// stopping criterion.  See section 2, inequality (2).
	ub := 0.
	for i := 0; i < n; i++ {
		var ab, aa float64
		for j, bj := range b {
			aj := A[j][i]
			ab += aj * bj
			aa += aj * aj
		}
		if u := ab / aa; u > 0 {
			ub += u
		}
	}
	// Hx quantity used for stopping criterion
	Hx := make([]float64, n)
	for i = 1; i < Limit; i++ {
		// for absolute stopping criterion.  ch is a check to see if
		// any elements of x changed at all.
		ch := false
		for k, xk := range x {
			// first line of step 2.
			b := xk - μ[k]/Hd[k]
			if b < 0 {
				b = 0
			}
			if b == xk {
				continue
			}
			x[k] = b
			ch = true
			// second line of step 2.
			b -= xk // repurpose b as the delta, used to update μ.
			for j, h := range H[k] {
				μ[j] += b * h
			}
		}
		// compute Hx, for stopping criterion
		for i, Hi := range H {
			Hxi := 0.
			for j, Hij := range Hi {
				Hxi += Hij * x[j]
			}
			Hx[i] = Hxi
		}
		// compute ⟨x, Hx⟩
		xHx := 0.
		for i, xi := range x {
			xHx += xi * Hx[i]
		}
		// objective function F(x) = (1/2)⟨x, Hx⟩ + ⟨x, f⟩
		xf := 0.
		for i, xi := range x {
			xf += xi * f[i]
		}
		// F := xHx/2 + xf

		// min(Hx + f) = min(μ)
		mHxf := μ[0]
		for i := 1; i < n; i++ {
			if m := μ[i]; m < mHxf {
				mHxf = m
			}
		}
		// Lower bound, LB(x), Section 3, inequality 10.
		// ub*min(Hx + f) - (1/2)⟨x, Hx⟩
		// LB := ub*mHxf - xHx/2

		// δKKT stopping criterion: F - LB <= δ
		// => (xHx/2 + xf) - (ub*mHxf - xHx/2) <= δ
		// => xHx + xf - ub*mHxf <= δ
		// if F-LB <= δ {
		if xHx+xf-ub*mHxf <= δ {
			break
		}
		// absolute stopping criterion:  if no changes, stop.
		if !ch {
			break
		}
	}
	return
}

// SCAKKT solves the non-negative least squares problem.
//
// Argument A represents a design or input matrix.  Each element of A must
// have the same length.  b represents a measurement or output vector.
// ε is a tolerance for stopping iteration.
//
// Stopping criteria for this function are based on Karush–Kuhn–Tucker
// conditions (see https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions).
// Iteration stops when a result is non-negative and the gradient of the
// objective function is within ε of KKT conditions.  The advantage of this
// function over SCA is that stopping conditions are simpler to compute
// and so each iteration takes less time.  The disadvantage is that the ε of
// this function is less obviously related to the objective function and
// stopping criteria are less clear.
//
// The result x returns coefficients of the fitted linear function, it
// will have the same length as elements of A.   Result i is the number of
// iterations performed.  An error is returned if A and b are not the same
// length.
func SCAKKT(A [][]float64, b []float64, ε float64) ([]float64, int, error) {
	m := len(A)
	n := len(A[0])
	if len(b) != m {
		return nil, 0, errors.New("A, b must be same length")
	}
	H := make([][]float64, n)
	Hd := make([]float64, n)
	for i := range H {
		Hi := make([]float64, n)
		for j := 0; j < i; j++ {
			Hi[j] = H[j][i]
		}
		s := 0.
		for k := range b {
			e := A[k][i]
			s += e * e
		}
		Hi[i] = s
		Hd[i] = s
		for j := i + 1; j < n; j++ {
			s := 0.
			for k := range b {
				s += A[k][i] * A[k][j]
			}
			Hi[j] = s
		}
		H[i] = Hi
	}
	x := make([]float64, n)
	μ := make([]float64, n)
	for j := range μ {
		e := 0.
		for i, bi := range b {
			e -= bi * A[i][j]
		}
		μ[j] = e
	}

	nε := -ε
	i := 1
i:
	for ; i < Limit; i++ {
		ch := false
		for k, xk := range x {
			// first line of step 2.
			b := xk - μ[k]/Hd[k]
			if b < 0 {
				b = 0
			}
			if b == xk {
				continue
			}
			x[k] = b
			ch = true
			b -= xk
			for j, h := range H[k] {
				μ[j] += b * h
			}
		}
		if !ch {
			break
		}
		// εKKT criteria
		for k, m := range μ {
			xk := x[k]
			if xk < 0 {
				continue i
			}
			if m < nε {
				continue i
			}
			if xk > 0 && m > ε {
				continue i
			}
		}
		break
	}
	return x, i, nil
}

// SCALimit solves the non-negative least squares problem.
//
// Argument A represents a design or input matrix.  Each element of A must
// have the same length.  b represents a measurement or output vector.
// limit is a maximum number of iterations.
//
// Iteration will stops after limit iterations or when the result does not
// change from one iteration to the next.  If argument limit is < 0, the
// package value Limit is used.
//
// The result x returns coefficients of the fitted linear function, it
// will have the same length as elements of A.   Result i is the number of
// iterations performed.  An error is returned if A and b are not the same
// length.
func SCALimit(A [][]float64, b []float64, limit int) ([]float64, int, error) {
	m := len(A)
	n := len(A[0])
	if len(b) != m {
		return nil, 0, errors.New("A, b must be same length")
	}
	H := make([][]float64, n)
	Hd := make([]float64, n)
	for i := range H {
		Hi := make([]float64, n)
		for j := 0; j < i; j++ {
			Hi[j] = H[j][i]
		}
		s := 0.
		for k := range b {
			e := A[k][i]
			s += e * e
		}
		Hi[i] = s
		Hd[i] = s
		for j := i + 1; j < n; j++ {
			s := 0.
			for k := range b {
				s += A[k][i] * A[k][j]
			}
			Hi[j] = s
		}
		H[i] = Hi
	}
	x := make([]float64, n)
	μ := make([]float64, n)
	for j := range μ {
		e := 0.
		for i, bi := range b {
			e -= bi * A[i][j]
		}
		μ[j] = e
	}
	if limit < 0 {
		limit = Limit
	}
	i := 1
	for ; i < limit; i++ {
		ch := false
		for k, xk := range x {
			b := xk - μ[k]/Hd[k]
			if b < 0 {
				b = 0
			}
			if b == xk {
				continue
			}
			x[k] = b
			ch = true
			b -= xk
			for j, h := range H[k] {
				μ[j] += b * h
			}
		}
		if !ch {
			break
		}
	}
	return x, i, nil
}
