use std::f64;
use std::ops::{Range, RangeTo, RangeFrom, RangeFull};

use errors::*;

use Matrix;
use MatrixRange;

pub trait CloneSub<R, S> {
    type Output;

    fn clone_subm(&self, rngr: R, rngc: S) -> Result<Self::Output>;
}
pub trait SubMatrix<R, S> {
    type Output;

    fn subm(&self, rngr: R, rngc: S) -> Result<Self::Output>;
}

#[inline]
fn fill_clone_subm(src: &Matrix, startr: usize, endr: usize, startc: usize, endc: usize)
        -> Result<Matrix> {
    let mut vec: Vec<f64> = Vec::new();
    for c in startc..endc {
        for r in startr..endr {
            vec.push(src.get(r,c)?.clone());
        }
    }
    Ok(Matrix::from_vec(vec, endr - startr, endc - startc))
}

impl CloneSub<Range<usize>, Range<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: Range<usize>, rngc: Range<usize>) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, rngr.end, rngc.start, rngc.end)
    }
}
impl CloneSub<Range<usize>, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: Range<usize>, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, rngr.end, 0, rngc.end)
    }
}
impl CloneSub<Range<usize>, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: Range<usize>, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, rngr.end, rngc.start, self.ncols())
    }
}
impl CloneSub<Range<usize>, RangeFull> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: Range<usize>, _: RangeFull) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, rngr.end, 0, self.ncols())
    }
}

impl CloneSub<RangeTo<usize>, Range<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeTo<usize>, rngc: Range<usize>) -> Result<Matrix> {
        fill_clone_subm(self, 0, rngr.end, rngc.start, rngc.end)
    }
}
impl CloneSub<RangeTo<usize>, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeTo<usize>, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_clone_subm(self, 0, rngr.end, 0, rngc.end)
    }
}
impl CloneSub<RangeTo<usize>, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeTo<usize>, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_clone_subm(self, 0, rngr.end, rngc.start, self.ncols())
    }
}
impl CloneSub<RangeTo<usize>, RangeFull> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeTo<usize>, _: RangeFull) -> Result<Matrix> {
        fill_clone_subm(self, 0, rngr.end, 0, self.ncols())
    }
}

impl CloneSub<RangeFrom<usize>, Range<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeFrom<usize>, rngc: Range<usize>) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, self.nrows(), rngc.start, rngc.end)
    }
}
impl CloneSub<RangeFrom<usize>, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeFrom<usize>, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, self.nrows(), 0, rngc.end)
    }
}
impl CloneSub<RangeFrom<usize>, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeFrom<usize>, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, self.nrows(), rngc.start, self.ncols())
    }
}
impl CloneSub<RangeFrom<usize>, RangeFull> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeFrom<usize>, _: RangeFull) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, self.nrows(), 0, self.ncols())
    }
}

impl CloneSub<RangeFull, Range<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, _: RangeFull, rngc: Range<usize>) -> Result<Matrix> {
        fill_clone_subm(self, 0, self.nrows(), rngc.start, rngc.end)
    }
}
impl CloneSub<RangeFull, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, _: RangeFull, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_clone_subm(self, 0, self.nrows(), 0, rngc.end)
    }
}
impl CloneSub<RangeFull, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, _: RangeFull, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_clone_subm(self, 0, self.nrows(), rngc.start, self.ncols())
    }
}
impl CloneSub<RangeFull, RangeFull> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, _: RangeFull, _: RangeFull) -> Result<Matrix> {
        fill_clone_subm(self, 0, self.nrows(), 0, self.ncols())
    }
}

impl CloneSub<usize, Range<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: usize, rngc: Range<usize>) -> Result<Matrix> {
        fill_clone_subm(self, rngr, rngr + 1, rngc.start, rngc.end)
    }
}
impl CloneSub<usize, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: usize, rngc: RangeTo<usize>) -> Result<Matrix> {
        fill_clone_subm(self, rngr, rngr + 1, 0, rngc.end)
    }
}
impl CloneSub<usize, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: usize, rngc: RangeFrom<usize>) -> Result<Matrix> {
        fill_clone_subm(self, rngr, rngr + 1, rngc.start, self.ncols())
    }
}
impl CloneSub<usize, RangeFull> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: usize, _: RangeFull) -> Result<Matrix> {
        fill_clone_subm(self, rngr, rngr + 1, 0, self.ncols())
    }
}

impl CloneSub<Range<usize>, usize> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: Range<usize>, rngc: usize) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, rngr.end, rngc, rngc + 1)
    }
}
impl CloneSub<RangeTo<usize>, usize> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeTo<usize>, rngc: usize) -> Result<Matrix> {
        fill_clone_subm(self, 0, rngr.end, rngc, rngc + 1)
    }
}
impl CloneSub<RangeFrom<usize>, usize> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, rngr: RangeFrom<usize>, rngc: usize) -> Result<Matrix> {
        fill_clone_subm(self, rngr.start, self.nrows(), rngc, rngc + 1)
    }
}
impl CloneSub<RangeFull, usize> for Matrix {
    type Output = Matrix;

    fn clone_subm(&self, _: RangeFull, rngc: usize) -> Result<Matrix> {
        fill_clone_subm(self, 0, self.nrows(), rngc, rngc + 1)
    }
}

/*********** Sub implementations ***********/
impl SubMatrix<Range<usize>, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, rngc: Range<usize>) -> Result<Matrix> {
        if rngr.start >= self.nrows() || rngr.end > self.nrows() || rngc.start >= self.ncols()
                || rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, rngc.start)..(rngr.end, rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<Range<usize>, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, rngc: RangeTo<usize>) -> Result<Matrix> {
        if rngr.start >= self.nrows() || rngr.end > self.nrows() || rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, 0)..(rngr.end, rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<Range<usize>, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, rngc: RangeFrom<usize>) -> Result<Matrix> {
        if rngr.start >= self.nrows() || rngr.end > self.nrows() || rngc.start >= self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, rngc.start)..(rngr.end, self.ncols())),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<Range<usize>, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, _: RangeFull) -> Result<Matrix> {
        if rngr.start >= self.nrows() || rngr.end > self.nrows() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, 0)..(rngr.end, self.ncols())),
            transposed: self.transposed,
        })
    }
}

impl SubMatrix<RangeTo<usize>, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, rngc: Range<usize>) -> Result<Matrix> {
        if rngr.end > self.nrows() || rngc.start >= self.ncols() || rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, rngc.start)..(rngr.end, rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeTo<usize>, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, rngc: RangeTo<usize>) -> Result<Matrix> {
        if rngr.end > self.nrows() || rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, 0)..(rngr.end, rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeTo<usize>, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, rngc: RangeFrom<usize>) -> Result<Matrix> {
        if rngr.end > self.nrows() || rngc.start >= self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, rngc.start)..(rngr.end, self.ncols())),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeTo<usize>, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, _: RangeFull) -> Result<Matrix> {
        if rngr.end > self.nrows() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, 0)..(rngr.end, self.ncols())),
            transposed: self.transposed,
        })
    }
}

impl SubMatrix<RangeFrom<usize>, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, rngc: Range<usize>) -> Result<Matrix> {
        if rngr.start >= self.nrows() || rngc.start >= self.ncols() || rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, rngc.start)..(self.nrows(), rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeFrom<usize>, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, rngc: RangeTo<usize>) -> Result<Matrix> {
        if rngr.start >= self.nrows() || rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, 0)..(self.nrows(), rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeFrom<usize>, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, rngc: RangeFrom<usize>) -> Result<Matrix> {
        if rngr.start >= self.nrows() || rngc.start >= self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, rngc.start)..(self.nrows(), self.ncols())),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeFrom<usize>, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, _: RangeFull) -> Result<Matrix> {
        if rngr.start >= self.nrows() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, 0)..(self.nrows(), self.ncols())),
            transposed: self.transposed,
        })
    }
}

impl SubMatrix<RangeFull, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, rngc: Range<usize>) -> Result<Matrix> {
        if rngc.start >= self.ncols() || rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, rngc.start)..(self.nrows(), rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeFull, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, rngc: RangeTo<usize>) -> Result<Matrix> {
        if rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, 0)..(self.nrows(), rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeFull, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, rngc: RangeFrom<usize>) -> Result<Matrix> {
        if rngc.start >= self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, rngc.start)..(self.nrows(), self.ncols())),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeFull, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, _: RangeFull) -> Result<Matrix> {
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, 0)..(self.nrows(), self.ncols())),
            transposed: self.transposed,
        })
    }
}

impl SubMatrix<Range<usize>, usize> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: Range<usize>, c: usize) -> Result<Matrix> {
        if rngr.start >= self.nrows() || rngr.end > self.nrows() || c >= self.ncols()  {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, c)..(rngr.end, c + 1)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeTo<usize>, usize> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeTo<usize>, c: usize) -> Result<Matrix> {
        if rngr.end > self.nrows() || c >= self.ncols()  {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, c)..(rngr.end, c + 1)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeFrom<usize>, usize> for Matrix {
    type Output = Matrix;

    fn subm(&self, rngr: RangeFrom<usize>, c: usize) -> Result<Matrix> {
        if rngr.start >= self.nrows() || c >= self.ncols()  {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((rngr.start, c)..(self.nrows(), c + 1)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<RangeFull, usize> for Matrix {
    type Output = Matrix;

    fn subm(&self, _: RangeFull, c: usize) -> Result<Matrix> {
        if c >= self.ncols()  {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((0, c)..(self.nrows(), c + 1)),
            transposed: self.transposed,
        })
    }
}

impl SubMatrix<usize, Range<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, r: usize, rngc: Range<usize>) -> Result<Matrix> {
        if r >= self.nrows() || rngc.start >= self.ncols() || rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((r, rngc.start)..(r + 1, rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<usize, RangeTo<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, r: usize, rngc: RangeTo<usize>) -> Result<Matrix> {
        if r >= self.nrows() || rngc.end > self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((r, 0)..(r + 1, rngc.end)),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<usize, RangeFrom<usize>> for Matrix {
    type Output = Matrix;

    fn subm(&self, r: usize, rngc: RangeFrom<usize>) -> Result<Matrix> {
        if r >= self.nrows() || rngc.start >= self.ncols() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((r, rngc.start)..(r + 1, self.ncols())),
            transposed: self.transposed,
        })
    }
}
impl SubMatrix<usize, RangeFull> for Matrix {
    type Output = Matrix;

    fn subm(&self, r: usize, _: RangeFull) -> Result<Matrix> {
        if r >= self.nrows() {
            return Err(Error::from_kind(ErrorKind::IndexError("range out of bounds")));
        }
        Ok(Matrix {
            data: self.data.clone(),
            view: MatrixRange((r, 0)..(r + 1, self.ncols())),
            transposed: self.transposed,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn assert_sub_first_row(b: &Matrix) {
        assert_eq!(b.dims(), (1,5));
        assert_eq!(b.iter().fold(f64::NEG_INFINITY, |acc, f| acc.max(f)), 5.0);
        assert_eq!(b.get(0,0).unwrap(), 1.0);
        assert_eq!(b.get(0,1).unwrap(), 2.0);
        assert_eq!(b.get(0,2).unwrap(), 3.0);
        assert_eq!(b.get(0,3).unwrap(), 4.0);
        assert_eq!(b.get(0,4).unwrap(), 5.0);
    }
    fn assert_subview_first_row(b: &Matrix) {
        assert!(b.is_subview());
        assert_sub_first_row(b);
    }

    fn assert_sub_full(b: &Matrix) {
        assert_eq!(b.dims(), (2, 5));
        assert_fpvec_eq!(b,
            mat![1.0, 2.0, 3.0, 4.0, 5.0;
                 10.0, 20.0, 30.0, 40.0, 50.0]);
    }
    fn assert_subview_full(b: &Matrix) {
        assert!(!b.is_subview());
        assert_sub_full(b);
    }

    fn assert_sub_second_col(b: &Matrix) {
        assert_eq!(b.dims(), (2,1));
        assert_eq!(b.get(0,0).unwrap(), 2.0);
        assert_eq!(b.get(1,0).unwrap(), 20.0);
    }
    fn assert_subview_second_col(b: &Matrix) {
        assert!(b.is_subview());
        assert_sub_second_col(b);
    }

    #[test]
    fn test_clone_subm() {
        let a = mat![1.0, 2.0, 3.0, 4.0, 5.0;
                     10.0, 20.0, 30.0, 40.0, 50.0];

        assert_sub_first_row(&a.clone_subm(0, ..).unwrap());
        assert_sub_first_row(&a.clone_subm(0..1, ..).unwrap());
        assert_sub_first_row(&a.clone_subm(0, 0..5).unwrap());
        assert_sub_first_row(&a.clone_subm(0, ..5).unwrap());
        assert_sub_first_row(&a.clone_subm(0, 0..).unwrap());
        assert_sub_first_row(&a.clone_subm(0..1, 0..5).unwrap());

        assert_sub_full(&a.clone_subm(.., ..).unwrap());
        assert_sub_full(&a.clone_subm(0.., ..).unwrap());
        assert_sub_full(&a.clone_subm(..2, ..).unwrap());
        assert_sub_full(&a.clone_subm(0..2, ..).unwrap());
        assert_sub_full(&a.clone_subm(.., 0..).unwrap());
        assert_sub_full(&a.clone_subm(.., ..5).unwrap());
        assert_sub_full(&a.clone_subm(.., 0..5).unwrap());
        assert_sub_full(&a.clone_subm(0..2, 0..5).unwrap());

        assert_sub_second_col(&a.clone_subm(.., 1).unwrap());
        assert_sub_second_col(&a.clone_subm(.., 1..2).unwrap());
        assert_sub_second_col(&a.clone_subm(0..2, 1).unwrap());
        assert_sub_second_col(&a.clone_subm(..2, 1).unwrap());
        assert_sub_second_col(&a.clone_subm(0.., 1).unwrap());
        assert_sub_second_col(&a.clone_subm(0..2, 1..2).unwrap());

        assert_fpvec_eq!(&a.clone_subm(.., 1..3).unwrap(), mat![2.0, 3.0; 20.0, 30.0]);
    }

    #[test]
    fn test_subm() {
        let a = mat![1.0, 2.0, 3.0, 4.0, 5.0; 10.0, 20.0, 30.0, 40.0, 50.0];

        assert_subview_first_row(&a.subm(0, ..).unwrap());
        assert_subview_first_row(&a.subm(0..1, ..).unwrap());
        assert_subview_first_row(&a.subm(0, 0..5).unwrap());
        assert_subview_first_row(&a.subm(0, ..5).unwrap());
        assert_subview_first_row(&a.subm(0, 0..).unwrap());
        assert_subview_first_row(&a.subm(0..1, 0..5).unwrap());

        assert_subview_full(&a.subm(.., ..).unwrap());
        assert_subview_full(&a.subm(0.., ..).unwrap());
        assert_subview_full(&a.subm(..2, ..).unwrap());
        assert_subview_full(&a.subm(0..2, ..).unwrap());
        assert_subview_full(&a.subm(.., 0..).unwrap());
        assert_subview_full(&a.subm(.., ..5).unwrap());
        assert_subview_full(&a.subm(.., 0..5).unwrap());
        assert_subview_full(&a.subm(0..2, 0..5).unwrap());

        assert_subview_second_col(&a.subm(.., 1).unwrap());
        assert_subview_second_col(&a.subm(.., 1..2).unwrap());
        assert_subview_second_col(&a.subm(0..2, 1).unwrap());
        assert_subview_second_col(&a.subm(..2, 1).unwrap());
        assert_subview_second_col(&a.subm(0.., 1).unwrap());
        assert_subview_second_col(&a.subm(0..2, 1..2).unwrap());
    }

    #[test]
    fn test_sub_mul() {
        let a = mat![1.0, 2.0, 3.0, 4.0, 5.0; 6.0, 7.0, 8.0, 9.0, 10.0];

        let b = a.subm(0..1, 0..5).unwrap();
        assert_eq!(b.dims(), (1, 5));
        assert!(b.is_subview());

        let c = &b * mat![2.0; 3.0; 4.0; 5.0; 6.0];
        assert_eq!(c.dims(), (1, 1));
        assert_eq!(c.get(0,0).unwrap(), 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0 + 5.0 * 6.0);
    }

    #[test]
    fn test_sub_add() {
        let a = mat![1.0, 2.0, 3.0, 4.0, 5.0; 10.0, 20.0, 30.0, 40.0, 50.0];

        let b = a.subm(0..1, 0..5).unwrap();
        let c = a.subm(1, ..).unwrap();
        assert_eq!(b.dims(), (1, 5));
        assert_eq!(c.dims(), (1, 5));
        assert!(b.is_subview());
        assert!(c.is_subview());

        let d = &b + &c;
        assert_eq!(d.dims(), (1, 5));
        assert!(!d.is_subview()); // d is a new matrix
        assert_eq!(d.get(0,0).unwrap(), 11.0);
        assert_eq!(d.get(0,1).unwrap(), 22.0);
        assert_eq!(d.get(0,2).unwrap(), 33.0);
        assert_eq!(d.get(0,3).unwrap(), 44.0);
        assert_eq!(d.get(0,4).unwrap(), 55.0);
    }
}
