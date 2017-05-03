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

    blas::c::dgemm(Layout::ColumnMajor, a.transposed.convert_to_blas(),
        b.transposed.convert_to_blas(), m, n, k, alpha, &a.data.values.borrow()[..], lda,
        &b.data.values.borrow()[..], ldb, beta, &mut out.data.values.borrow_mut()[..], m);
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
        assert_eq!(*out.data.values.borrow(),
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
        assert_eq!(*out.data.values.borrow(),
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
        assert_eq!(*out.data.values.borrow(),
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
        assert_eq!(*out.data.values.borrow(),
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
        assert_eq!(*out.data.values.borrow(),
            vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0]);
    }

    #[test]
    fn test_matrix_add_move_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = a + &b;

        assert_eq!(*out.data.values.borrow(),
            vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
    }

    #[test]
    fn test_matrix_add_move_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = a + b;

        assert_eq!(*out.data.values.borrow(),
            vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
    }

    #[test]
    fn test_matrix_add_ref_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = &a + b;

        assert_eq!(*out.data.values.borrow(),
            vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
    }

    #[test]
    fn test_matrix_add_ref_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = &a + &b;

        assert_eq!(*out.data.values.borrow(),
            vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);
    }

    #[test]
    fn test_matrix_sub_move_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = a - &b;

        assert_eq!(*out.data.values.borrow()    , vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0]);
    }

    #[test]
    fn test_matrix_sub_move_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = a - b;

        assert_eq!(*out.data.values.borrow(), vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0]);
    }

    #[test]
    fn test_matrix_sub_ref_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = &a - b;

        assert_eq!(*out.data.values.borrow(), vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0]);
    }

    #[test]
    fn test_matrix_sub_ref_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0], m, n);
        let b = Matrix::from_vec(vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0], m, n);

        let out = &a - &b;

        assert_eq!(*out.data.values.borrow(), vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0]);
    }

    #[test]
    fn test_matrix_neg_move() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);

        let out = -a;

        assert_eq!(*out.data.values.borrow(), vec![-2.0, -7.0, -6.0, -2.0, 0.0, -7.0, -4.0, -2.0]);
    }

    #[test]
    fn test_matrix_neg_ref() {
        let (m, n) = (2, 4);
        let a = Matrix::from_vec(vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0], m, n);

        let out = -&a;

        assert_eq!(*out.data.values.borrow(), vec![-2.0, -7.0, -6.0, -2.0, 0.0, -7.0, -4.0, -2.0]);
    }

}
