# Matrix

[![Build Status](https://travis-ci.org/jblondin/matrix.svg?branch=master)](https://travis-ci.org/jblondin/matrix)

A matrix library for Rust.

Features:
* Matrix-scalar addition, subtraction, multiplication
* Matrix-matrix entrywise addition, subtraction
* Matrix-matrix multiplication
* Submatrix indexing and extraction
* Matrix decomposition:
  * LU Decomposition
  * QR Decomposition
  * Cholesky Factorization
  * Eigenvalue Decomposition
  * Singular Value Decomposition
* Systems of equations solver (including overdetermined and underdetermined systems)
* Inverse matrices
* Vector Norms: L<sub>1</sub>, L<sub>2</sub>, Inf, L<sub>p</sub>
* Matrix Norms:
* Induced: L<sub>1</sub>, L<sub>2</sub> / Spectral, Inf
  * Entry-wise (for any vector norm)
  * L<sub>p, q</sub>
  * Frobenius (same as entry-wise L<sub>2</sub>)
  * Max (same as entry-wise Inf)
  * Schatten norms (for any vector norm), including nuclear (same as L<sub>1</sub> Schatten)
