
extern crate blas;
extern crate lapack;
extern crate num;
#[macro_use] extern crate error_chain;

mod errors;

pub mod core;
pub use core::{Matrix, MatrixIter, SubMatrix, gemv};
