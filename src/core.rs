//! Core matrix structures and implementations.

use std::f64;
use std::cell::{Ref, RefMut, RefCell};
use std::rc::Rc;
use std::fmt;
use std::ops::Range;

use rand::{self, Rand, Rng};
use rand::distributions::{IndependentSample, Normal};
use rand::distributions::normal::StandardNormal;

use errors::*;

/// Transpose flag; whether or not to treat the matrix as transposed
#[derive(Debug, Clone, Copy)]
pub enum Transpose {
    /// Yes, transpose
    Yes,
    /// No transposition
    No,
}
impl Transpose {
    /// Convert flag to BLAS transposition flag
    #[inline]
    pub fn convert_to_blas(&self) -> ::blas::c::Transpose {
        match *self {
            Transpose::Yes  => ::blas::c::Transpose::Ordinary,
            Transpose::No   => ::blas::c::Transpose::None,
        }
    }
    /// Switch the transposition flag
    #[inline]
    pub fn t(&self) -> Transpose {
        match *self {
            Transpose::Yes  => Transpose::No,
            Transpose::No   => Transpose::Yes,
        }
    }
}

/// Matrix storage structure. Data is stored in a single vector, in column-major format.
#[derive(Debug, Clone)]
pub struct MatrixData {
    values: RefCell<Vec<f64>>,
    rows: usize,
    cols: usize,
}
impl MatrixData {
    /// borrow the underlying matrix data
    pub fn values(&self) -> Ref<Vec<f64>> {
        self.values.borrow()
    }
    /// mutably borrow the underlying matrix data
    pub fn values_mut(&self) -> RefMut<Vec<f64>> {
        self.values.borrow_mut()
    }
}

/// Submatrix range specification; a range from (r_start, c_start) to (r_end, c_end)
#[derive(Debug, Clone)]
pub struct MatrixRange(pub Range<(usize, usize)>);
impl MatrixRange {
    /// Generate a range consisting of the entire underlying matrix
    pub fn full(nrows: usize, ncols: usize) -> MatrixRange {
        MatrixRange((0, 0)..(nrows, ncols))
    }
    fn nrows(&self) -> usize {
        self.0.end.0 - self.0.start.0
    }
    fn ncols(&self) -> usize {
        self.0.end.1 - self.0.start.1
    }
    fn start_row(&self) -> usize {
        self.0.start.0
    }
    fn start_col(&self) -> usize {
        self.0.start.1
    }
}

/// View into matrix data. A single MatrixData structure can back multiple Matrix views, with
/// different range specifications (to denote submatrices) or transpositions.
#[derive(Debug)]
pub struct Matrix {
    /// Reference to the underlying matrix storage structure.
    pub data: Rc<MatrixData>,
    /// The range of values from the underlying matrix storage this matrix shows
    pub view: MatrixRange,
    /// Whether or not to transpose the data in the underlying matrix storage
    pub transposed: Transpose,
}

/// Flag to specify how to symmetrize a matrix (used in to_symmetic method)
pub enum SymmetrizeMethod {
    /// Symmetrize a matrix by copying the lower triangular portion
    CopyLower,
    /// Symmetrize a matrix by copying the upper triangular portion
    CopyUpper,
}

impl Matrix {
    /// Create a matrix (with new underlying matrix storage) from data vector (in column-major
    /// order) with the specificied dimensions
    pub fn from_vec(data: Vec<f64>, nrows: usize, ncols:usize) -> Matrix {
        assert_eq!(data.len(), nrows * ncols);
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(data),
                rows: nrows,
                cols: ncols,
            }),
            view: MatrixRange::full(nrows, ncols),
            transposed: Transpose::No,
        }
    }
    /// Create a matrix (with new underlying matrix storage) consisting entirely of the value 1.0,
    /// with specified dimensions
    pub fn ones(nrows: usize, ncols: usize) -> Matrix {
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(vec![1.0; nrows * ncols]),
                rows: nrows,
                cols: ncols,
            }),
            view: MatrixRange::full(nrows, ncols),
            transposed: Transpose::No,
        }
    }
    /// Create a matrix (with new underlying matrix storage) consisting entirely of the value 0.0,
    /// with specified dimensions
    pub fn zeros(nrows: usize, ncols: usize) -> Matrix {
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(vec![0.0; nrows * ncols]),
                rows: nrows,
                cols: ncols,
            }),
            view: MatrixRange::full(nrows, ncols),
            transposed: Transpose::No,
        }
    }
    /// Create a matrix (with new underlying matrix storage) with the provided data vector as its
    /// diagonal (and zeroes elsewhere)
    pub fn diag(vec: &Vec<f64>) -> Matrix {
        let n = vec.len();
        let mut m = Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(vec![0.0; n * n]),
                rows: n,
                cols: n,
            }),
            view: MatrixRange::full(n, n),
            transposed: Transpose::No,
        };

        for i in 0..n {
            m.set(i, i, vec[i]).expect("invalid indexing");
        }
        m
    }
    /// Create an identity matrix (with new underlying matrix storage) of given size
    pub fn eye(n: usize) -> Matrix {
        Matrix::diag(&vec![1.0; n])
    }
    /// Create a random matrix (with new underlying matrix storage) of given dimensions, with values
    /// between 0.0 and 1.0.
    pub fn rand(nrows: usize, ncols: usize) -> Matrix {
        let mut rng = rand::thread_rng();

        let mut v: Vec<f64> = Vec::new();
        for _ in 0..nrows * ncols {
            v.push(rng.gen());
        }
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(v),
                rows: nrows,
                cols: ncols,
            }),
            view: MatrixRange::full(nrows, ncols),
            transposed: Transpose::No,
        }
    }
    /// Create a random matrix (with new underlying matrix storage) of given dimensions, with values
    /// drawn from a standard normal distribution (mean 0.0, standard deviation 1.0)
    pub fn randsn(nrows: usize, ncols: usize) -> Matrix {
        let mut rng = rand::thread_rng();

        let mut v: Vec<f64> = Vec::new();
        for _ in 0..nrows * ncols {
            v.push(StandardNormal::rand(&mut rng).0);
        }
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(v),
                rows: nrows,
                cols: ncols,
            }),
            view: MatrixRange::full(nrows, ncols),
            transposed: Transpose::No,
        }
    }
    /// Create a random matrix (with new underlying matrix storage) of given dimensions, with values
    /// drawn from a normal distribution with specified mean and standard deviation
    pub fn randn(nrows: usize, ncols: usize, mean: f64, stdev: f64) -> Matrix {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(mean, stdev);

        let mut v: Vec<f64> = Vec::new();
        for _ in 0..nrows * ncols {
            v.push(dist.ind_sample(&mut rng));
        }
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(v),
                rows: nrows,
                cols: ncols,
            }),
            view: MatrixRange::full(nrows, ncols),
            transposed: Transpose::No,
        }

    }

    /// Retrieve the number of rows of this matrix
    pub fn nrows(&self) -> usize {
        match self.transposed {
            Transpose::Yes  => { self.view.ncols() }
            Transpose::No   => { self.view.nrows() }
        }
    }
    /// Retrieve the number of columns of this matrix
    pub fn ncols(&self) -> usize {
        match self.transposed {
            Transpose::Yes  => { self.view.nrows() }
            Transpose::No   => { self.view.ncols() }
        }
    }
    /// Retrieve the dimensions of this matrix
    pub fn dims(&self) -> (usize, usize) {
        match self.transposed {
            Transpose::Yes  => { (self.view.ncols(), self.view.nrows()) }
            Transpose::No   => { (self.view.nrows(), self.view.ncols()) }
        }
    }
    /// Retrieve the minimum dimension (either number of rows or number of columns, whichever is
    /// smaller) of this matrix
    pub fn mindim(&self) -> usize {
        let (m, n) = self.dims();
        if m < n { m } else { n }
    }
    /// Retrieve the maximum dimension (either number of rows or number of columns, whichever is
    /// larger) of this matrix
    pub fn maxdim(&self) -> usize {
        let (m, n) = self.dims();
        if m > n { m } else { n }
    }
    /// Retrieve the number of values in this matrix
    pub fn length(&self) -> usize { self.view.ncols() * self.view.nrows() }
    /// Whether or not this matrix is square
    pub fn is_square(&self) -> bool { self.nrows() == self.ncols() }
    /// Whether or not this matrix is a vector (either row or column)
    pub fn is_vector(&self) -> bool { self.nrows() == 1 || self.ncols() == 1 }
    /// Whether or not this matrix is a row vector (only consists of one row)
    pub fn is_row_vector(&self) -> bool { self.nrows() == 1 }
    /// Whether or not this matrix is a column vector (only consists of one column)
    pub fn is_col_vector(&self) -> bool { self.ncols() == 1 }

    /// Generate a new, transposed, matrix view into the underlying matrix data
    pub fn transpose(&self) -> Matrix {
        Matrix {
            data: self.data.clone(),
            view: self.view.clone(),
            transposed: self.transposed.t()
        }
    }
    /// Short-cut for transpose() method
    #[inline]
    pub fn t(&self) -> Matrix { self.transpose() }

    /// Generate an iterator over the matrix values
    pub fn iter(&self) -> MatrixIter {
        MatrixIter {
            mat: &self,
            current_loc: (0, 0),
        }
    }

    /// Retrieve a specific value from the matrix
    ///
    /// # Failures
    /// Returns `Err` if specified row, column is out of bounds
    pub fn get(&self, r: usize, c: usize) -> Result<f64> {
        self.data.values.borrow().get(self.index(r, c)).map(|&f| f)
            .ok_or(Error::from_kind(ErrorKind::IndexError("index out of bounds")))
    }
    /// Sets a specific value in the matrix
    ///
    /// # Failures
    /// Returns `Err` if specified row, column is out of bounds
    pub fn set(&mut self, r: usize, c: usize, value: f64) -> Result<()> {
        let mut v = self.data.values.borrow_mut();
        let i = self.index(r, c);
        if i >= v.len() {
            return Err(Error::from_kind(ErrorKind::IndexError("index out of bounds")));
        }
        v[i] = value;
        Ok(())
    }

    /// Generate a new matrix (with new underlying matrix data, copied from the given matrices)
    /// which is composed of a horizontal concatenation of the two given matrices. The new matrix
    /// will have the same number of rows, and a number of columns equal to the sum of the number of
    /// columns of the given matrices.
    ///
    /// # Panics
    /// Panics if the two given matrices do not have the same number of rows.
    pub fn hcat(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.nrows(), other.nrows());

        // use the iterators instead of direct access to the data vectors so transposition is
        // handled properly
        let mut data_vec: Vec<f64> = self.iter().collect();
        data_vec.append(&mut other.iter().collect());

        let (nrows, ncols) = (self.nrows(), self.ncols() + other.ncols());
        assert_eq!(data_vec.len(), nrows * ncols);
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(data_vec),
                rows: nrows,
                cols: ncols,
            }),
            view: MatrixRange::full(nrows, ncols),
            transposed: Transpose::No,
        }
    }
    /// Generate a new matrix (with new underlying matrix data, copied from the given matrices)
    /// which is composed of a vertical concatenation of the two given matrices. The new matrix
    /// will have the same number of columns, and a number of rows equal to the sum of the number of
    /// rows of the given matrices.
    ///
    /// # Panics
    /// Panics if the two given matrices do not have the same number of columns.
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

        let (nrows, ncols) = (self.nrows() + other.nrows(), self.ncols());
        assert_eq!(data_vec.len(), nrows * ncols);
        Matrix {
            data: Rc::new(MatrixData {
                values: RefCell::new(data_vec),
                rows: nrows,
                cols: ncols,
            }),
            view: MatrixRange::full(nrows, ncols),
            transposed: Transpose::No,
        }
    }

    /// Returns whether or not this matrix view is a subview (does not show all of the underlying
    /// data)
    pub fn is_subview(&self) -> bool {
        self.nrows() * self.ncols() != self.data.rows * self.data.cols
    }
    /// Pointer to the underlying matrix data
    pub fn data(&self) -> Rc<MatrixData> {
        if self.is_subview() {
            Rc::new(MatrixData {
                values: RefCell::new(self.iter().collect::<Vec<f64>>()),
                rows: self.nrows(),
                cols: self.ncols(),
            })
        } else {
            self.data.clone()
        }
    }

    /// Generate a new matrix (with new underlying data) which is of a symmetric matrix copied
    /// from either the lower triangular portion or the upper triangular portion of this matrix.
    pub fn to_symmetric(&self, method: SymmetrizeMethod) -> Matrix {
        assert!(self.is_square());
        let mut symm = self.clone();
        match method {
            SymmetrizeMethod::CopyUpper => {
                for i in 0..self.nrows() {
                    for j in 0..i {
                        let prev_value = symm.get(j, i).unwrap();
                        symm.set(i, j, prev_value).unwrap();
                    }
                }
            }
            SymmetrizeMethod::CopyLower => {
                let m = self.nrows();
                for i in 0..m {
                    for j in (i + 1)..m {
                        let prev_value = symm.get(j, i).unwrap();
                        symm.set(i, j, prev_value).unwrap();
                    }
                }
            }
        }
        symm
    }

    #[inline]
    fn index(&self, r: usize, c: usize) -> usize {
        match self.transposed {
            Transpose::Yes  => {
                let (r, c) = (self.view.start_col() + r, self.view.start_row() + c);
                self.trindex(c * self.data.cols + r)
            }
            Transpose::No   => {
                let (r, c) = (self.view.start_row() + r, self.view.start_col() + c);
                c * self.data.rows + r
            }
        }
    }
    #[inline]
    fn trindex(&self, index: usize) -> usize {
        (index % self.data.cols) * self.data.rows
            + (index as f32 / self.data.cols as f32).floor() as usize
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                write!(f, "{:+1.5e} ", self.get(i, j).unwrap())?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Clone for Matrix {
    /// Copies all of the data represented by this matrix view into a new matrix (with new
    /// underlying data).
    fn clone(&self) -> Matrix {
        Matrix::from_vec(self.iter().collect(), self.nrows(), self.ncols())
    }
}

/// Matrix iterator
pub struct MatrixIter<'a> {
    mat: &'a Matrix,
    current_loc: (usize, usize),
}
impl<'a> Iterator for MatrixIter<'a> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        if self.current_loc.1 >= self.mat.ncols() { return None }

        let val = self.mat.get(self.current_loc.0, self.current_loc.1).ok();

        self.current_loc.0 += 1;
        if self.current_loc.0 >= self.mat.nrows() {
            self.current_loc.0 = 0;
            self.current_loc.1 += 1;
        }
        val
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
    fn test_get() {
        let a = mat![1, 2; 3, 4];
        assert_eq!(a.dims(), (2, 2));
        assert_eq!(a.get(0, 0).unwrap(), 1.0);
        assert_eq!(a.get(0, 1).unwrap(), 2.0);
        assert_eq!(a.get(1, 0).unwrap(), 3.0);
        assert_eq!(a.get(1, 1).unwrap(), 4.0);
    }
    #[test]
    fn test_set() {
        let mut a = mat![1, 2; 3, 4];
        assert_eq!(a.dims(), (2, 2));
        a.set(1, 0, 5.0).unwrap();

        assert_eq!(a.get(0, 0).unwrap(), 1.0);
        assert_eq!(a.get(0, 1).unwrap(), 2.0);
        assert_eq!(a.get(1, 0).unwrap(), 5.0);
        assert_eq!(a.get(1, 1).unwrap(), 4.0);
    }

    #[test]
    fn test_diag() {
        let a = Matrix::diag(&vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(a.dims(), (5, 5));

        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    assert_eq!(a.get(i, j).unwrap(), i as f64 + 1.0);
                } else {
                    assert_eq!(a.get(i, j).unwrap(), 0.0);
                }
            }
        }
    }
    #[test]
    fn test_eye() {
        let a = Matrix::eye(5);

        assert_eq!(a.dims(), (5, 5));

        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    assert_eq!(a.get(i, j).unwrap(), 1.0);
                } else {
                    assert_eq!(a.get(i, j).unwrap(), 0.0);
                }
            }
        }
    }

    #[test]
    fn test_rand() {
        use subm::CloneSub;

        let (m, n) = (100, 100);
        let a = Matrix::rand(m, n);

        assert_eq!(a.dims(), (m, n));

        assert!(a.iter().fold(f64::NEG_INFINITY, |acc, f| acc.max(f)) < 1.0);
        assert!(a.iter().fold(f64::INFINITY, |acc, f| acc.min(f)) >= 0.0);

        println!("{:#?}", a.clone_subm(0..5, 0..5));
    }

    #[test]
    fn test_randsn() {
        use subm::CloneSub;

        let (m, n) = (100, 100);
        let a = Matrix::randsn(m, n);

        assert_eq!(a.dims(), (m, n));

        let rares = a.iter().fold(0, |acc, f| if f < -3.0 || f > 3.0 { acc + 1 } else { acc });
        // should be ~0.003, give a bit of leeway
        let limit = (0.006 * (m * n) as f64) as usize;
        println!("{:#?} {:#?}", rares, limit);
        assert!(rares < limit);

        println!("{:#?}", a.clone_subm(0..5, 0..5));
    }

    #[test]
    fn test_randn() {
        use subm::CloneSub;

        let (m, n) = (100, 100);
        let (mean, stdev) = (10.0, 3.0);
        let a = Matrix::randn(m, n, mean, stdev);

        assert_eq!(a.dims(), (m, n));

        let rares = a.iter().fold(
            0,
            |acc, f| if f < mean - 3.0 * stdev || f > mean + 3.0 * stdev { acc + 1 } else { acc }
        );
        // should be ~0.003, give a bit of leeway
        let limit = (0.006 * (m * n) as f64) as usize;
        println!("{:#?} {:#?}", rares, limit);
        assert!(rares < limit);

        println!("{:#?}", a.clone_subm(0..5, 0..5));
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
    fn test_symmetrize_lower() {
        let a = mat![1.0, 2.0, 3.0, 4.0;
                     1.0, 2.0, 3.0, 4.0;
                     1.0, 2.0, 3.0, 4.0;
                     1.0, 2.0, 3.0, 4.0];

        let a_symm = a.to_symmetric(SymmetrizeMethod::CopyLower);
        assert_fpvec_eq!(a_symm,
            mat![1.0, 1.0, 1.0, 1.0;
                 1.0, 2.0, 2.0, 2.0;
                 1.0, 2.0, 3.0, 3.0;
                 1.0, 2.0, 3.0, 4.0]);
    }
    #[test]
    fn test_symmetrize_upper() {
        let a = mat![1.0, 2.0, 3.0, 4.0;
                     1.0, 2.0, 3.0, 4.0;
                     1.0, 2.0, 3.0, 4.0;
                     1.0, 2.0, 3.0, 4.0];

        let a_symm = a.to_symmetric(SymmetrizeMethod::CopyUpper);
        assert_fpvec_eq!(a_symm,
            mat![1.0, 2.0, 3.0, 4.0;
                 2.0, 2.0, 3.0, 4.0;
                 3.0, 3.0, 3.0, 4.0;
                 4.0, 4.0, 4.0, 4.0]);
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
}
