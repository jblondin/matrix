
extern crate blas;
extern crate lapack;
extern crate num;
#[macro_use] extern crate error_chain;
extern crate rand;

mod errors;

#[macro_use] mod macro_def;

pub mod core;
pub use core::{Matrix, MatrixIter, Transpose};

mod ops;
mod subm;
pub use subm::SubMatrix;
