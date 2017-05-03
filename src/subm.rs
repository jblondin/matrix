use std::f64;
use std::ops::{Range, RangeTo, RangeFrom, RangeFull};

use errors::*;

use Matrix;


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

}
