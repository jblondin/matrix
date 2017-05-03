use std::f64;
use std::ops::{Add, Mul, Sub, Neg, Range, RangeTo, RangeFrom, RangeFull};
use std::cell::RefCell;
use std::rc::Rc;

use blas;
use blas::c::Layout;
use lapack;
use lapack::c::Layout as LapackLayout;

use errors::*;


#[derive(Debug, Clone)]
pub enum Transpose {
    Yes,
    No,
}
impl Transpose {
    #[inline]
    fn convert_to_blas(&self) -> ::blas::c::Transpose {
        match *self {
            Transpose::Yes  => ::blas::c::Transpose::Ordinary,
            Transpose::No   => ::blas::c::Transpose::None,
        }
    }
    #[inline]
    fn t(&self) -> Transpose {
        match *self {
            Transpose::Yes  => Transpose::No,
            Transpose::No   => Transpose::Yes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatrixData {
    values: RefCell<Vec<f64>>,
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Rc<MatrixData>,
    transposed: Transpose,
}

impl Matrix {
    pub fn from_vec(data: Vec<f64>, nrows: usize, ncols:usize) -> Matrix {
        assert_eq!(data.len(), nrows * ncols);
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(data),
                rows: nrows,
                cols: ncols,
            }),
            transposed: Transpose::No,
        }
    }
    pub fn ones(nrows: usize, ncols: usize) -> Matrix {
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(vec![1.0; nrows * ncols]),
                rows: nrows,
                cols: ncols,
            }),
            transposed: Transpose::No,
        }
    }
    pub fn zeros(nrows: usize, ncols: usize) -> Matrix {
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(vec![0.0; nrows * ncols]),
                rows: nrows,
                cols: ncols,
            }),
            transposed: Transpose::No,
        }
    }

    pub fn nrows(&self) -> usize {
        match self.transposed {
            Transpose::Yes  => { self.data.cols }
            Transpose::No   => { self.data.rows }
        }
    }
    pub fn ncols(&self) -> usize {
        match self.transposed {
            Transpose::Yes  => { self.data.rows }
            Transpose::No   => { self.data.cols }
        }
    }
    pub fn dims(&self) -> (usize, usize) {
        match self.transposed {
            Transpose::Yes  => { (self.data.cols, self.data.rows) }
            Transpose::No   => { (self.data.rows, self.data.cols) }
        }
    }
    pub fn length(&self) -> usize { self.data.cols * self.data.rows }
    pub fn transpose(&self) -> Matrix {
        Matrix {
            data: self.data.clone(),
            transposed: self.transposed.t()
        }
    }
    #[inline]
    pub fn t(&self) -> Matrix { self.transpose() }

    pub fn iter(&self) -> MatrixIter {
        MatrixIter {
            mat: &self,
            index: 0,
        }
    }

    pub fn get(&self, r: usize, c: usize) -> Result<f64> {
        let index = match self.transposed {
            Transpose::Yes  => { self.trindex(c * self.nrows() + r) }
            Transpose::No   => { c * self.nrows() + r }
        };
        self.data.values.borrow().get(index).map(|&f| f)
            .ok_or(Error::from_kind(ErrorKind::IndexError("index out of bounds")))
    }

    pub fn hcat(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.nrows(), other.nrows());

        // use the iterators instead of direct access to the data vectors so transposition is
        // handled properly
        let mut data_vec: Vec<f64> = self.iter().collect();
        data_vec.append(&mut other.iter().collect());

        assert_eq!(data_vec.len(), self.nrows() * (self.ncols() + other.ncols()));
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(data_vec),
                rows: self.nrows(),
                cols: self.ncols() + other.ncols(),
            }),
            transposed: Transpose::No,
        }
    }
    pub fn vcat(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.ncols(), other.ncols());

        let mut data_vec: Vec<f64> = Vec::new();

        for c in 0..self.ncols() {
            for r in 0..self.nrows() {
                data_vec.push(self.get(r, c).unwrap());
            }
            for r in 0..other.nrows() {
                data_vec.push(other.get(r, c).unwrap());
            }
        }

        assert_eq!(data_vec.len(), (self.nrows() + other.nrows()) * self.ncols());
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(data_vec),
                rows: self.nrows() + other.nrows(),
                cols: self.ncols(),
            }),
            transposed: Transpose::No,
        }
    }

    pub fn gram_solve(&self, b: &Matrix) -> Matrix {
        let (n, nrhs) = (self.ncols(), b.ncols());
        assert_eq!(b.nrows(), n);
        let udu = self.t() * self;
        let mut ipiv = vec![0; n];
        let soln = b.clone();
        let res = lapack::c::dsysv(LapackLayout::ColumnMajor, b'U',  n as i32, nrhs as i32,
            &mut udu.data.values.borrow_mut()[..], n as i32, &mut ipiv[..],
            &mut soln.data.values.borrow_mut()[..], n as i32);
        assert_eq!(res, 0);
        soln
    }

    #[inline]
    fn trindex(&self, index: usize) -> usize {
        (index % self.nrows()) * self.ncols()
            + (index as f32 / self.nrows() as f32).floor() as usize
    }
}

pub struct MatrixIter<'a> {
    mat: &'a Matrix,
    index: usize,
}
impl<'a> Iterator for MatrixIter<'a> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        if self.index >= self.mat.length() { return None }
        let val = match self.mat.transposed {
            Transpose::No => {
                self.mat.data.values.borrow().get(self.index).cloned()
            }
            Transpose::Yes => {
                // do conversion from one indexing style to other
                self.mat.data.values.borrow().get(self.mat.trindex(self.index)).cloned()
            }
        };
        self.index += 1;
        val
    }
}

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


pub trait SubMatrix<R, S> {
    type Output;

    fn subm(&self, rngr: R, rngc: S) -> Result<Self::Output>;
}

#[inline]
fn fill_subm(src: &Matrix, startr: usize, endr: usize, startc: usize, endc: usize)
        -> Result<Matrix> {
    let mut vec: Vec<f64> = Vec::new();
    for r in startr..endr {
        for c in startc..endc {
            vec.push(src.get(r,c)?.clone());
        }
    }
    Ok(Matrix::from_vec(vec, endr - startr, endc - startc))
}

impl SubMatrix<Range<usize>, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, rngc: Range<usize>) -> Result<Matrix> {
        fill_subm(self, rngr.start, rngr.end, rngc.start, rngc.end)
    }
}
impl SubMatrix<Range<usize>, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_subm(self, rngr.start, rngr.end, 0, rngc.end)
    }
}
impl SubMatrix<Range<usize>, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_subm(self, rngr.start, rngr.end, rngc.start, self.ncols())
    }
}
impl SubMatrix<Range<usize>, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, _: RangeFull) -> Result<Matrix> {
        fill_subm(self, rngr.start, rngr.end, 0, self.ncols())
    }
}

impl SubMatrix<RangeTo<usize>, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, rngc: Range<usize>) -> Result<Matrix> {
        fill_subm(self, 0, rngr.end, rngc.start, rngc.end)
    }
}
impl SubMatrix<RangeTo<usize>, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_subm(self, 0, rngr.end, 0, rngc.end)
    }
}
impl SubMatrix<RangeTo<usize>, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_subm(self, 0, rngr.end, rngc.start, self.ncols())
    }
}
impl SubMatrix<RangeTo<usize>, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, _: RangeFull) -> Result<Matrix> {
        fill_subm(self, 0, rngr.end, 0, self.ncols())
    }
}

impl SubMatrix<RangeFrom<usize>, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, rngc: Range<usize>) -> Result<Matrix> {
        fill_subm(self, rngr.start, self.nrows(), rngc.start, rngc.end)
    }
}
impl SubMatrix<RangeFrom<usize>, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_subm(self, rngr.start, self.nrows(), 0, rngc.end)
    }
}
impl SubMatrix<RangeFrom<usize>, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_subm(self, rngr.start, self.nrows(), rngc.start, self.ncols())
    }
}
impl SubMatrix<RangeFrom<usize>, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, _: RangeFull) -> Result<Matrix> {
        fill_subm(self, rngr.start, self.nrows(), 0, self.ncols())
    }
}

impl SubMatrix<RangeFull, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, rngc: Range<usize>) -> Result<Matrix> {
        fill_subm(self, 0, self.nrows(), rngc.start, rngc.end)
    }
}
impl SubMatrix<RangeFull, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_subm(self, 0, self.nrows(), 0, rngc.end)
    }
}
impl SubMatrix<RangeFull, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_subm(self, 0, self.nrows(), rngc.start, self.ncols())
    }
}
impl SubMatrix<RangeFull, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, _: RangeFull) -> Result<Matrix> {
        fill_subm(self, 0, self.nrows(), 0, self.ncols())
    }
}

impl SubMatrix<usize, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: usize, rngc: Range<usize>) -> Result<Matrix> {
        fill_subm(self, rngr, rngr + 1, rngc.start, rngc.end)
    }
}
impl SubMatrix<usize, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: usize, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_subm(self, rngr, rngr + 1, 0, rngc.end)
    }
}
impl SubMatrix<usize, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: usize, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_subm(self, rngr, rngr + 1, rngc.start, self.ncols())
    }
}
impl SubMatrix<usize, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: usize, _: RangeFull) -> Result<Matrix> {
        fill_subm(self, rngr, rngr + 1, 0, self.ncols())
    }
}

impl SubMatrix<Range<usize>, usize> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, rngc: usize) -> Result<Matrix> {
        fill_subm(self, rngr.start, rngr.end, rngc, rngc + 1)
    }
}
impl SubMatrix<RangeTo<usize>, usize> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, rngc: usize) -> Result<Matrix> {
        fill_subm(self, 0, rngr.end, rngc, rngc + 1)
    }
}
impl SubMatrix<RangeFrom<usize>, usize> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, rngc: usize) -> Result<Matrix> {
        fill_subm(self, rngr.start, self.nrows(), rngc, rngc + 1)
    }
}
impl SubMatrix<RangeFull, usize> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, rngc: usize) -> Result<Matrix> {
        fill_subm(self, 0, self.nrows(), rngc, rngc + 1)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ones() {
        let (m,n) = (5, 10);
        let a = Matrix::ones(m, n);

        assert_eq!(a.dims(), (m, n));
        assert_eq!(a.iter().fold(f64::NEG_INFINITY, |acc, f| acc.max(f)), 1.0);
        assert_eq!(a.iter().fold(f64::INFINITY, |acc, f| acc.min(f)), 1.0);
    }

    #[test]
    fn test_zeros() {
        let (m,n) = (5, 10);
        let a = Matrix::zeros(m, n);

        assert_eq!(a.dims(), (m, n));
        assert_eq!(a.iter().fold(f64::NEG_INFINITY, |acc, f| acc.max(f)), 0.0);
        assert_eq!(a.iter().fold(f64::INFINITY, |acc, f| acc.min(f)), 0.0);
    }

    #[test]
    fn test_hcat() {
        let (m1, n1) = (2, 3);
        let a = Matrix::ones(m1, n1);
        let (m2, n2) = (2, 2);
        let b = Matrix::zeros(m2, n2);

        let c = a.hcat(&b);
        assert_eq!(c.dims(), (2, 5));
        assert_eq!(c.get(0, 2).unwrap(), 1.0);
        assert_eq!(c.get(1, 1).unwrap(), 1.0);
        assert_eq!(c.get(0, 3).unwrap(), 0.0);
        assert_eq!(c.get(1, 4).unwrap(), 0.0);

        let (m1, n1) = (3, 2);
        let at = Matrix::ones(m1, n1).t();

        let c = at.hcat(&b);
        assert_eq!(c.dims(), (2, 5));
        assert_eq!(c.get(0, 2).unwrap(), 1.0);
        assert_eq!(c.get(1, 1).unwrap(), 1.0);
        assert_eq!(c.get(0, 3).unwrap(), 0.0);
        assert_eq!(c.get(1, 4).unwrap(), 0.0);

        let (m2, n2) = (4, 2);
        let bt = Matrix::zeros(m2, n2).t();

        let c = at.hcat(&bt);
        assert_eq!(c.dims(), (2, 7));
        assert_eq!(c.get(0, 2).unwrap(), 1.0);
        assert_eq!(c.get(1, 1).unwrap(), 1.0);
        assert_eq!(c.get(0, 5).unwrap(), 0.0);
        assert_eq!(c.get(1, 6).unwrap(), 0.0);
    }

    #[test]
    fn test_vcat() {
        let (m1, n1) = (2, 3);
        let a = Matrix::from_vec(vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0], m1, n1);
        let (m2, n2) = (1, 3);
        let b = Matrix::from_vec(vec![3.0, 6.0, 9.0], m2, n2);
        let c = a.vcat(&b);
        assert_eq!(c.dims(), (3, 3));
        assert_eq!(*c.data.values.borrow(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let (m1, n1) = (3, 2);
        let at = Matrix::from_vec(vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0], m1, n1).t();
        let c = at.vcat(&b);
        assert_eq!(c.dims(), (3, 3));
        assert_eq!(*c.data.values.borrow(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let (m2, n2) = (3, 1);
        let bt = Matrix::from_vec(vec![3.0, 6.0, 9.0], m2, n2).t();
        let c = a.vcat(&bt);
        assert_eq!(c.dims(), (3, 3));
        assert_eq!(*c.data.values.borrow(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_transpose() {
        let (m, n) = (2, 5);
        let a = Matrix::from_vec(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0], m, n);
        let b = a.t();

        let orig: Vec<f64> = a.iter().collect();
        assert_eq!(orig, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]);
        let tr: Vec<f64> = b.iter().collect();
        assert_eq!(tr, vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]);

        assert_eq!(a.dims(), (m, n));
        assert_eq!(b.dims(), (n, m));

        assert_eq!(a.get(1, 2).unwrap(), 30.0);
        assert_eq!(a.get(0, 4).unwrap(), 5.0);
        assert_eq!(b.get(2, 1).unwrap(), 30.0);
        assert_eq!(b.get(4, 0).unwrap(), 5.0);
    }

    fn assert_subm_first_row(b: &Matrix) {
        assert_eq!(b.dims(), (1,5));
        assert_eq!(b.iter().fold(f64::NEG_INFINITY, |acc, f| acc.max(f)), 5.0);
        assert_eq!(b.get(0,0).unwrap(), 1.0);
        assert_eq!(b.get(0,1).unwrap(), 2.0);
        assert_eq!(b.get(0,2).unwrap(), 3.0);
        assert_eq!(b.get(0,3).unwrap(), 4.0);
        assert_eq!(b.get(0,4).unwrap(), 5.0);
    }

    fn assert_subm_full(b: &Matrix) {
        assert_eq!(b.dims(), (2, 5));
    }

    fn assert_subm_second_col(b: &Matrix) {
        assert_eq!(b.dims(), (2,1));
        assert_eq!(b.get(0,0).unwrap(), 2.0);
        assert_eq!(b.get(1,0).unwrap(), 20.0);
    }

    #[test]
    fn test_subm() {
        let (m, n) = (2,5);
        let a = Matrix::from_vec(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0], m, n);

        assert_subm_first_row(&a.subm(0, ..).unwrap());
        assert_subm_first_row(&a.subm(0..1, ..).unwrap());
        assert_subm_first_row(&a.subm(0, 0..5).unwrap());
        assert_subm_first_row(&a.subm(0, ..5).unwrap());
        assert_subm_first_row(&a.subm(0, 0..).unwrap());
        assert_subm_first_row(&a.subm(0..1, 0..5).unwrap());

        assert_subm_full(&a.subm(.., ..).unwrap());
        assert_subm_full(&a.subm(0.., ..).unwrap());
        assert_subm_full(&a.subm(..2, ..).unwrap());
        assert_subm_full(&a.subm(0..2, ..).unwrap());
        assert_subm_full(&a.subm(.., 0..).unwrap());
        assert_subm_full(&a.subm(.., ..5).unwrap());
        assert_subm_full(&a.subm(.., 0..5).unwrap());
        assert_subm_full(&a.subm(0..2, 0..5).unwrap());

        assert_subm_second_col(&a.subm(.., 1).unwrap());
        assert_subm_second_col(&a.subm(.., 1..2).unwrap());
        assert_subm_second_col(&a.subm(0..2, 1).unwrap());
        assert_subm_second_col(&a.subm(..2, 1).unwrap());
        assert_subm_second_col(&a.subm(0.., 1).unwrap());
        assert_subm_second_col(&a.subm(0..2, 1..2).unwrap());
    }

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

    #[test]
    fn test_gram_solve() {
        let a = Matrix::ones(5, 1).hcat(
            &Matrix::from_vec(vec![0.45642, 0.86603, 0.38062, 0.62465, 0.15748], 5, 1));
        let b = Matrix::from_vec(vec![0.886446, 0.096545], 2, 1);

        let x = a.gram_solve(&b);
        assert_eq!(x.dims(), (2, 1));
        assert!((x - Matrix::from_vec(vec![0.78168, -1.21599], 2, 1)).iter()
            .fold(f64::NEG_INFINITY, |acc, f| acc.max(f.abs())) < 0.00001);

    }
}
