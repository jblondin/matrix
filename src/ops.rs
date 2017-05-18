use std::ops::{Add, Mul, Sub, Neg};

use blas;
use blas::c::Layout;

use Matrix;
use Transpose;


macro_rules! bin_inner {
    ($rhs:ty, $name:ident, $op:tt) => {
        type Output = Matrix;

        fn $name(self, rhs: $rhs) -> Matrix {
            assert_eq!(self.nrows(), rhs.nrows());
            assert_eq!(self.ncols(), rhs.ncols());

            Matrix::from_vec(
                self.iter().zip(rhs.iter()).map(|(l, r)| l $op r).collect(),
                self.nrows(), self.ncols()
            )
        }
    }
}
macro_rules! add_inner {
    ($rhs:ty) => { bin_inner!($rhs, add, +); }
}
macro_rules! implement_add {
    ($lhs:ty, $rhs:ty) => {
        impl Add<$rhs> for $lhs {
            add_inner!($rhs);
        }
    };
    ($lhs:ty, $rhs:ty, $( $lifetime:tt ),* ) => {
        impl<$($lifetime),*> Add<$rhs> for $lhs {
            add_inner!($rhs);
        }
    };
}
implement_add!(Matrix, Matrix);
implement_add!(Matrix, &'a Matrix, 'a);
implement_add!(&'a Matrix, Matrix, 'a);
implement_add!(&'a Matrix, &'b Matrix, 'a, 'b);

macro_rules! sub_inner {
    ($rhs:ty) => { bin_inner!($rhs, sub, -); }
}
macro_rules! implement_sub {
    ($lhs:ty, $rhs:ty) => {
        impl Sub<$rhs> for $lhs {
            sub_inner!($rhs);
        }
    };
    ($lhs:ty, $rhs:ty, $( $lifetime:tt),* ) => {
        impl<$($lifetime),*> Sub<$rhs> for $lhs {
            sub_inner!($rhs);
        }
    }
}

implement_sub!(Matrix, Matrix);
implement_sub!(Matrix, &'a Matrix, 'a);
implement_sub!(&'a Matrix, Matrix, 'a);
implement_sub!(&'a Matrix, &'b Matrix, 'a, 'b);

// negation
impl Neg for Matrix {
    type Output = Matrix;

    fn neg(self) -> Matrix {
        Matrix::from_vec(self.iter().map(|e| -e).collect(), self.nrows(), self.ncols())
    }
}
impl<'b> Neg for &'b Matrix {
    type Output = Matrix;

    fn neg(self) -> Matrix {
        Matrix::from_vec(self.iter().map(|e| -e).collect(), self.nrows(), self.ncols())
    }
}



// multiplication
pub fn gemv(a: &Matrix, b: &Matrix, alpha: f64, c_beta: Option<(&Matrix, f64)>)
        -> Matrix {
    let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
    assert!(k == b.nrows());
    let (out, beta) = match c_beta {
        Some((c, beta)) => {
            assert!(m == c.nrows());
            assert!(n == c.ncols());
            (c.clone(), beta)
        },
        None => {
            (Matrix::from_vec(vec![0.0; m * n], m, n), 0.0)
        }
    };

    let (m, k, n) = (m as i32, k as i32, n as i32);
    let lda = match a.transposed {
        Transpose::Yes  => { k }
        Transpose::No   => { m }
    };
    let ldb = match b.transposed {
        Transpose::Yes  => { n }
        Transpose::No   => { k }
    };

    let (a_data, b_data, out_data) = (a.data(), b.data(), out.data());
    blas::c::dgemm(Layout::ColumnMajor, a.transposed.convert_to_blas(),
        b.transposed.convert_to_blas(), m, n, k, alpha,
        &a_data.values()[..], lda,
        &b_data.values()[..], ldb, beta,
        &mut out_data.values_mut()[..], m);

    out
}

macro_rules! mul_inner {
    ($rhs:ty) => {
        type Output = Matrix;

        fn mul(self, rhs: $rhs) -> Matrix {
            gemv(&self, &rhs, 1.0, None)
        }
    }
}
macro_rules! implement_mul {
    ($lhs:ty, $rhs:ty) => {
        impl Mul<$rhs> for $lhs {
            mul_inner!($rhs);
        }
    };
    ($lhs:ty, $rhs:ty, $( $lifetime:tt),* ) => {
        impl<$($lifetime),*> Mul<$rhs> for $lhs {
            mul_inner!($rhs);
        }
    }
}
implement_mul!(Matrix, Matrix);
implement_mul!(Matrix, &'a Matrix, 'a);
implement_mul!(&'a Matrix, Matrix, 'a);
implement_mul!(&'a Matrix, &'b Matrix, 'a, 'b);

fn scalar_mul(mat: &Matrix, rhs: f64) -> Matrix {
    let mut out = mat.clone();
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            let prev_value = out.get(i, j).unwrap();
            out.set(i, j, rhs * prev_value).unwrap();
        }
    }
    out
}

impl Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Matrix {
        scalar_mul(&self, rhs)
    }
}
impl<'a> Mul<f64> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Matrix {
        scalar_mul(self, rhs)
    }
}
impl Mul<Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        scalar_mul(&rhs, self)
    }
}
impl<'a> Mul<&'a Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: &'a Matrix) -> Matrix {
        scalar_mul(rhs, self)
    }
}

pub trait Dot<T> {
    type Output;

    fn dot(&self, rhs: &T) -> Self::Output;
}
impl Dot<Matrix> for Matrix {
    type Output = f64;

    fn dot(&self, rhs: &Matrix) -> f64 {
        assert!(self.is_vector() && rhs.is_vector());
        self.iter().zip(rhs.iter()).map(|(l, r)| l * r).fold(0.0, |acc, f| acc + f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv() {
        let (m, n, k) = (2, 4, 3);
        let a = Matrix::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], m, k);
        let b = Matrix::from_vec(
            vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0], k, n);
        let c = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);

        let out = gemv(&a, &b, 1.0, Some((&c, 1.0)));

        assert_eq!(out.nrows(), m);
        assert_eq!(out.ncols(), n);
        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);

    }

    #[test]
    fn test_matrix_mul_move_ref() {
        let (m, n, k) = (2, 4, 3);
        let a = Matrix::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], m, k);
        let b = Matrix::from_vec(
            vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0], k, n);

        let out = a * &b;

        assert_eq!(out.nrows(), m);
        assert_eq!(out.ncols(), n);
        let out_data = out.data();
        assert_eq!(*out_data.values(),
                vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0]);
    }

    #[test]
    fn test_matrix_mul_move_move() {
        let (m, n, k) = (2, 4, 3);
        let a = Matrix::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], m, k);
        let b = Matrix::from_vec(
            vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0], k, n);

        let out = a * b;

        assert_eq!(out.nrows(), m);
        assert_eq!(out.ncols(), n);
        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0]);
    }

    #[test]
    fn test_matrix_mul_ref_move() {
        let (m, n, k) = (2, 4, 3);
        let a = Matrix::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], m, k);
        let b = Matrix::from_vec(
            vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0], k, n);

        let out = &a * b;

        assert_eq!(out.nrows(), m);
        assert_eq!(out.ncols(), n);
        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0]);
    }

    #[test]
    fn test_matrix_mul_ref_ref() {
        let (m, n, k) = (2, 4, 3);
        let a = Matrix::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], m, k);
        let b = Matrix::from_vec(
            vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0], k, n);

        let out = &a * &b;

        assert_eq!(out.nrows(), m);
        assert_eq!(out.ncols(), n);
        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0]);
    }

    #[test]
    fn test_matrix_scalar_mul_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], m, n);

        let out = a * 2.0;

        assert_eq!(out.nrows(), m);
        assert_eq!(out.ncols(), n);
        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![2.0, 10.0, 4.0, 12.0, 6.0, 14.0, 8.0, 16.0]);
    }

    #[test]
    fn test_matrix_scalar_mul_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], m, n);

        let out = &a * 2.0;

        assert_eq!(out.nrows(), m);
        assert_eq!(out.ncols(), n);
        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![2.0, 10.0, 4.0, 12.0, 6.0, 14.0, 8.0, 16.0]);
    }

    #[test]
    fn test_scalar_matrix_mul_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], m, n);

        let out = 2.0 * a;

        assert_eq!(out.nrows(), m);
        assert_eq!(out.ncols(), n);
        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![2.0, 10.0, 4.0, 12.0, 6.0, 14.0, 8.0, 16.0]);
    }

    #[test]
    fn test_scalar_matrix_mul_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], m, n);

        let out = 2.0 * &a;

        assert_eq!(out.nrows(), m);
        assert_eq!(out.ncols(), n);
        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![2.0, 10.0, 4.0, 12.0, 6.0, 14.0, 8.0, 16.0]);
    }
    #[test]
    fn test_matrix_add_move_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = a + &b;

        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
    }

    #[test]
    fn test_matrix_add_move_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = a + b;

        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
    }

    #[test]
    fn test_matrix_add_ref_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = &a + b;

        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
    }

    #[test]
    fn test_matrix_add_ref_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = &a + &b;

        let out_data = out.data();
        assert_eq!(*out_data.values(),
            vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
    }

    #[test]
    fn test_matrix_sub_move_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = a - &b;

        let out_data = out.data();
        assert_eq!(*out_data.values(), vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0]);
    }

    #[test]
    fn test_matrix_sub_move_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = a - b;

        let out_data = out.data();
        assert_eq!(*out_data.values(), vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0]);
    }

    #[test]
    fn test_matrix_sub_ref_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = &a - b;

        let out_data = out.data();
        assert_eq!(*out_data.values(), vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0]);
    }

    #[test]
    fn test_matrix_sub_ref_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = &a - &b;

        let out_data = out.data();
        assert_eq!(*out_data.values(), vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0]);
    }

    #[test]
    fn test_matrix_neg_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);

        let out = -a;

        let out_data = out.data();
        assert_eq!(*out_data.values(), vec![-2.0, -7.0, -6.0, -2.0, 0.0, -7.0, -4.0, -2.0]);
    }

    #[test]
    fn test_matrix_neg_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);

        let out = -&a;

        let out_data = out.data();
        assert_eq!(*out_data.values(), vec![-2.0, -7.0, -6.0, -2.0, 0.0, -7.0, -4.0, -2.0]);
    }

    #[test]
    fn test_dot() {
        let expected = 10.0 + 40.0 + 90.0 + 160.0 + 250.0 + 360.0;

        let a = mat![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = mat![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        assert!(a.is_row_vector());
        assert!(b.is_row_vector());
        assert_eq!(a.dot(&b), expected);

        let a = mat![1.0; 2.0; 3.0; 4.0; 5.0; 6.0];
        let b = mat![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        assert!(a.is_col_vector());
        assert!(b.is_row_vector());
        assert_eq!(a.dot(&b), expected);

        let a = mat![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = mat![10.0; 20.0; 30.0; 40.0; 50.0; 60.0];
        assert!(a.is_row_vector());
        assert!(b.is_col_vector());
        assert_eq!(a.dot(&b), expected);

        let a = mat![1.0; 2.0; 3.0; 4.0; 5.0; 6.0];
        let b = mat![10.0; 20.0; 30.0; 40.0; 50.0; 60.0];
        assert!(a.is_col_vector());
        assert!(b.is_col_vector());
        assert_eq!(a.dot(&b), expected);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_dot_nonvec() {
        let a = mat![1.0, -1.0; 2.0, -2.0; 3.0, -3.0; 4.0, -4.0; 5.0, -5.0; 6.0, -6.0];
        let b = mat![10.0; 20.0; 30.0; 40.0; 50.0; 60.0];
        assert!(!a.is_vector());
        assert!(b.is_col_vector());
        a.dot(&b);
    }

}
