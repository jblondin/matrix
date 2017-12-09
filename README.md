# Matrix

[![Build Status](https://travis-ci.org/jblondin/matrix.svg?branch=master)](https://travis-ci.org/jblondin/matrix)
[![Documentation](https://docs.rs/wee-matrix/badge.svg)](https://docs.rs/wee-matrix)

A matrix library for Rust. Supports basic addition, subtraction, and multiplication operations, solving systems of equations (including underdetermined and overdetermined), decompositions, norms (both vector and matrix), and submatrices. Wraps calls to BLAS / LAPACK for most operations.

## Features

This library includes the following features:
* Matrix-scalar addition, subtraction, multiplication
* Matrix-matrix entrywise addition, subtraction
* Matrix-matrix multiplication
* Submatrix indexing and extraction
* Multiple matrix views into same underlying data
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

## Usage

To use, add the following to your `Cargo.toml`:
```toml
[dependencies]
wee-matrix = "0.1"
```

## Examples

This section provides a few examples of usage. For more examples, see the documentation.

### Matrix creation

Using the `mat!` macro:
```rust
let a = mat![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
assert_eq!(a.dims(), (3, 4));
```

Using `from_vec`:
```
let b = Matrix::from_vec(vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0],
    3, 4);
assert_eq!(a.iter().collect::<Vec<_>>(), b.iter().collect::<Vec<_>>());
```

Using `ones`:
```
let a = Matrix::ones(5, 4);
assert_eq!(a.dims(), (5, 4));
assert_eq!(a.get(0, 0).unwrap(), 1.0);
assert_eq!(a.get(0, 3).unwrap(), 1.0);
assert_eq!(a.get(4, 2).unwrap(), 1.0);
```

Other matrix creation methods:
- `diag`: create a matrix by providing the values of it's diagonal
- `zeros`: like `ones`, except filling the matrix with zeros
- `eye`: create a identity matrix of given size
- `rand`: create a matrix of random values betwen 0.0 and 1.0
- `randn`: create a matrix of random values drawn from normal distribution with given mean and StDev
- `randsn`: create a matrix of random values drawn from the standard normal distribution

### Matrix Concatenation

Concatenation of matrices of appropriate dimensions is possible using `hcat` and `vcat`:
```
let a = Matrix::rand(3, 2);
let b = Matrix::rand(2, 2);
let a_b = a.vcat(&b);
assert_eq!(a_b.dims(), (5, 2));

let c = Matrix::rand(3, 3);
let a_c = a.hcat(&c);
assert_eq!(a_c.dims(), (3, 5));
```

### Elementary Matrix Operations

Arithmetic matrix operations are supported:
```
let a = Matrix::ones(2, 2);
let b = Matrix::ones(2, 2);
let c = &a + &b;
assert_eq!(c.dims(), (2, 2));
assert_eq!(c.get(0, 0).unwrap(), 2.0);

let d = &c * Matrix::zeros(2, 2);
assert_eq!(d.dims(), (2, 2));
assert_eq!(d.get(0, 0).unwrap(), 0.0);
```

## Current and future state

While this library does (generally) function as intended, there is a lot of work that needs to be done. This library is currently in an ALPHA state: it has basic functionality, but is buggy, untested, poorly documented, and subject to change.

Some intended improvements are:
- Matrix pretty-printing
- Matrix comparison operators
- Ease-of-use improvements
- Additional testing
- Lots of additional examples and documentation
- Performance analysis and improvements
