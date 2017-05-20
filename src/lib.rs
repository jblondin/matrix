
extern crate blas;
extern crate lapack;
extern crate num;
#[macro_use] extern crate error_chain;
extern crate rand;
#[allow(unused_imports)] #[macro_use] extern crate unittest;

mod errors;

#[macro_use] mod macro_def;

pub mod core;
pub use core::{Matrix, MatrixRange, MatrixIter, Transpose, SymmetrizeMethod};

mod ops;
mod subm;
pub use subm::{SubMatrix, CloneSub};

mod decompose;
pub use decompose::{Compose,
    LU, LUDecompose,
    QR, QRDecompose,
    Cholesky, CholeskyDecompose,
    Eigen, EigenOptions, EigenDecompose,
    SVD, SingularValueDecompose
};

mod solve;
pub use solve::{Solve, GramSolve};

mod map;

mod norm;
pub use norm::{Norm, VectorNorm};
