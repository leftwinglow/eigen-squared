A numerical linear algebra library implemented using only Numpy arrays.

# Features

## Matrix Decompositions
* QR
    * Gram-Schmidt
    * Modified Gram-Schmidt
    * Householder Reflections
    * Givens Rotations
* Cholesky
    * Cholesky-Crout
    * Cholesky-Banachiewicz

## Eigenpair Solvers
* Rayleigh Quotient Iteration (shifting)
* QR (shifting, pivoting)
* Power Iteration & Inverse Power Iteration (shifting)
* Lanczos Iteration

## Other Features
* Gershgorin circle solver
* Column pivoting
* Matrix checks
    * Squareness
    * Invertibility via determinant and rank
    * Symmetry
    * Hermitian
    * Positive definiteness
    * Row/column, weak/strict diagonal dominance
* Matrix Shifts
    * Simple shift
    * Spectal shift
    * Wilkinson shift
    * Rayleigh quotient shift
* Standard, diagonal & non-diagonal Frobenius norm

