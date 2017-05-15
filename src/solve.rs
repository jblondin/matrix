use lapack;
use lapack::c::Layout;

use errors::*;

use Matrix;
use SubMatrix;
use LUDecompose;
use decompose::{c_to_lapack_indexing};

#[inline]
fn max(x: usize, y: usize) -> usize { if x > y { x } else { y } }

#[derive(Debug, Clone)]
pub struct ApproxSoln<T> {
    pub soln: T,
    pub resid: Option<Vec<f64>>,
}

pub trait Solve {
    type Rhs;
    type Output;

    fn solve(&self, b: &Self::Rhs) -> Result<Self::Output>;
    fn solve_exact(&self, b: &Self::Rhs) -> Result<Self::Output>;
    fn solve_symm(&self, b: &Self::Rhs) -> Result<Self::Output>;
    fn solve_approx(&self, b: &Self::Rhs) -> Result<ApproxSoln<Self::Output>>;
    fn inverse(&self) -> Result<Self::Output>;
}

impl Solve for Matrix {
    type Rhs = Matrix;
    type Output = Matrix;

    fn solve(&self, b: &Matrix) -> Result<Matrix> {
        if self.is_square() {
            self.solve_exact(b)
        } else {
            self.solve_approx(b).map(|approx_soln| approx_soln.soln)
        }
    }

    fn solve_exact(&self, b: &Matrix) -> Result<Matrix> {
        if !self.is_square() {
            return Err(Error::from_kind(ErrorKind::SolveError(
                "solve_exact called with non-square matrix".to_string())))
        }
        let (m, n, nrhs) = (self.nrows(), self.ncols(), b.ncols());
        if b.nrows() != m {
            return Err(Error::from_kind(ErrorKind::SolveError(
                "right-hand side nrows must match left-hand matrix nrows".to_string())))
        }

        let (lda, ldb) = (n, n);
        let mut ipiv = vec![0; n];
        let inout = self.clone();
        let soln = b.clone();
        let info = lapack::c::dgesv(Layout::ColumnMajor, n as i32, nrhs as i32,
            &mut inout.data.values.borrow_mut()[..], lda as i32, &mut ipiv[..],
            &mut soln.data.values.borrow_mut()[..], ldb as i32);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::SolveError(
                format!("Matrix solver: Invalid call to dgesv in argument {}", -info))))
        } else if info > 0 {
            Err(Error::from_kind(ErrorKind::SolveError(
                "Matrix solver: matrix is singular ".to_string())))
        } else {
            Ok(soln)
        }
    }

    fn solve_symm(&self, b: &Matrix) -> Result<Matrix> {
        if !self.is_square() {
            return Err(Error::from_kind(ErrorKind::SolveError(
                "solve_symm called with non-square matrix".to_string())))
        }
        let (m, n, nrhs) = (self.nrows(), self.ncols(), b.ncols());
        if b.nrows() != m {
            return Err(Error::from_kind(ErrorKind::SolveError(
                "right-hand side nrows must match left-hand matrix nrows".to_string())))
        }

        let (lda, ldb) = (n, n);
        let mut ipiv = vec![0; n];
        let inout = self.clone();
        let soln = b.clone();
        let info = lapack::c::dsysv(Layout::ColumnMajor, b'U',  n as i32, nrhs as i32,
            &mut inout.data.values.borrow_mut()[..], lda as i32, &mut ipiv[..],
            &mut soln.data.values.borrow_mut()[..], ldb as i32);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::SolveError(
                format!("Symmetric matrix solver: \
                    Invalid call to dsysv in argument {}", -info))))
        } else if info > 0 {
            Err(Error::from_kind(ErrorKind::SolveError(
                "Symmetric matrix solver: matrix is singular ".to_string())))
        } else {
            Ok(soln)
        }
    }

    fn solve_approx(&self, b: &Matrix) -> Result<ApproxSoln<Matrix>> {
        let (m, n, nrhs) = (self.nrows(), self.ncols(), b.ncols());
        if b.nrows() != m {
            return Err(Error::from_kind(ErrorKind::SolveError(
                "right-hand side nrows must match left-hand matrix nrows".to_string())))
        }

        let (lda, ldb) = (m, max(m, n));
        let inout = self.clone();
        let soln = if m >= n {
            b.clone()
        } else {
            b.clone().vcat(&Matrix::zeros(n - m, nrhs))
        };
        let info = lapack::c::dgels(Layout::ColumnMajor, b'N', m as i32, n as i32, nrhs as i32,
            &mut inout.data.values.borrow_mut()[..], lda as i32,
            &mut soln.data.values.borrow_mut()[..], ldb as i32);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::SolveError(
                format!("Approx matrix solver: \
                    Invalid call to dgels in argument {}", -info))))
        } else if info > 0 {
            Err(Error::from_kind(ErrorKind::SolveError(
                "Approx matrix solver: matrix is rank-deficient ".to_string())))
        } else {
            if m > n {
                Ok(ApproxSoln {
                    soln: soln.subm(0..n, ..).unwrap(),
                    resid: {
                        let mut v: Vec<f64> = Vec::new();
                        for j in 0..nrhs {
                            v.push(soln.subm(n..m, j).unwrap().iter().fold(0.0,
                                |acc, f| acc + f * f));
                        }
                        Some(v)
                    },
                })
            } else {
                Ok(ApproxSoln {
                    soln: soln,
                    resid: None
                })
            }
        }
    }

    fn inverse(&self) -> Result<Matrix> {
        if !self.is_square() {
            return Err(Error::from_kind(ErrorKind::SolveError(
                "inverse called with non-square matrix".to_string())))
        }

        let m = self.nrows();
        let lda = m;

        let lu = self.lu()?;
        let inout = lu.lu_data().clone();
        let ipiv = c_to_lapack_indexing(lu.ipiv_data());

        let info = lapack::c::dgetri(Layout::ColumnMajor, m as i32,
            &mut inout.data.values.borrow_mut()[..], lda as i32,
            &ipiv[..]);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::SolveError(
                format!("Matrix inversion: \
                    Invalid call to dgetri in argument {}", -info))))
        } else if info > 0 {
            Err(Error::from_kind(ErrorKind::SolveError(
                "Matrix inversion: matrix is singular ".to_string())))
        } else {
            Ok(inout)
        }

    }
}

pub trait GramSolve {
    type Rhs;
    type Output;

    fn gram_solve(&self, b: &Self::Rhs) -> Result<Self::Output>;
}

impl GramSolve for Matrix {
    type Rhs = Matrix;
    type Output = Matrix;

    fn gram_solve(&self, b: &Matrix) -> Result<Matrix> {
        let udu = self.t() * self;
        udu.solve_symm(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use SymmetrizeMethod;

    macro_rules! assert_error {
        ($res:expr, $err_type:path, $needle:expr, $errtype_str:expr) => {
            assert!($res.is_err());
            let e = $res.unwrap_err();
            println!("{:?}", e.kind());
            match *e.kind() {
                $err_type(ref m) => {
                    assert!(m.find($needle).is_some());
                },
                _ => { panic!(format!("Expected {}, found: {}", $errtype_str, e.kind())) }
            }

        }
    }

    // all should result in singular matrices, but the sum and copyfirst method don't seem to be
    // working for me (probably numerical instability issues)
    #[allow(dead_code)]
    enum SingularMethod {
        Zeros,
        Sum,
        CopyFirst
    }
    fn generate_rank_deficient_matrix(m: usize, n: usize, method: SingularMethod) -> Matrix {
        assert!(m > 1);
        assert!(n > 1);

        let mut a = Matrix::randsn(m, 1);
        let v1_copy = a.clone();
        let mut sum = a.clone();
        for _ in 1..(n - 1) {
            let vi = Matrix::randsn(m, 1);
            a = a.hcat(&vi);
            sum = sum + vi;
        }
        // specify last column based on method chosen
        match method {
            SingularMethod::Zeros       => { a.hcat(&Matrix::zeros(m, 1)) }
            SingularMethod::Sum         => { a.hcat(&sum) }
            SingularMethod::CopyFirst   => { a.hcat(&v1_copy) }
        }
    }
    fn generate_singular_matrix(m: usize, method: SingularMethod) -> Matrix {
        generate_rank_deficient_matrix(m, m, method)
    }

    fn solve_exact_driver(a: &Matrix, b: &Matrix) -> Result<Matrix> {
        a.solve_exact(b).map(|x| {
            assert_eq!(x.dims(), (a.ncols(), b.ncols()));
            x
        })
    }
    #[test]
    fn test_solve_exact() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let x = solve_exact_driver(&a, &b).expect("solve_exact failed unexpectedly");

        assert_eq!(x.dims(), (m, 1));
        println!("a*x\n{}\nb\n{}", &a * &x, &b);
        assert_fpvec_eq!(&a * &x, &b);
    }
    #[test]
    fn test_solve_exact_nonsquare() {
        let (m, n) = (6, 4);
        let a = Matrix::randsn(m, n);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let solve_res = solve_exact_driver(&a, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "non-square", "SolveError");
    }
    #[test]
    fn test_solve_exact_singular() {
        let m = 6;
        let a = generate_singular_matrix(m, SingularMethod::Zeros);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let solve_res = solve_exact_driver(&a, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "singular", "SolveError");
    }
    #[test]
    fn test_solve_exact_invalidrhs() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        println!("{}", a);

        let b = Matrix::randsn(m + 1, 1); // invalud number of rows

        let solve_res = solve_exact_driver(&a, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "right-hand side", "SolveError");
    }

    fn solve_symm_driver(a: &Matrix, b: &Matrix) -> Result<Matrix> {
        a.solve_symm(b).map(|x| {
            assert_eq!(x.dims(), (a.ncols(), b.ncols()));
            x
        })
    }
    #[test]
    fn test_solve_symm() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        let a_symm = a.to_symmetric(SymmetrizeMethod::CopyUpper);
        println!("{}", a_symm);

        let b = Matrix::randsn(m, 1);

        let soln = solve_symm_driver(&a_symm, &b).expect("solve_symm failed unexpectedly");

        assert_eq!(soln.dims(), (m, 1));
        println!("a*x\n{}\nb\n{}", &a_symm * &soln, &b);
        assert_fpvec_eq!(&a_symm * &soln, &b);
    }
    #[test]
    fn test_solve_symm_nonsquare() {
        let (m, n) = (6, 4);
        let a = Matrix::randsn(m, n);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let solve_res = solve_symm_driver(&a, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "non-square", "SolveError");
    }
    #[test]
    fn test_solve_symm_nonsymm() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        println!("a\n{}", a);

        let b = Matrix::randsn(m, 1);

        // should succeed because it never references half the matrix
        let soln = solve_symm_driver(&a, &b).expect("solve_symm failed unexpectedly");
        println!("x\n{}", soln);

        assert_eq!(soln.dims(), (m, 1));

        // a * x != b, since a is the non-symmetric matrix
        println!("a*x\n{}\nb\n{}", &a * &soln, &b);
        assert_fpvec_neq!(&a * &soln, &b);

        // convert a to a symmetric matrix (using upper triangular part since we specify using
        // the upper triangular part of A when calling dsysv) and solution should work
        let a_symm = a.to_symmetric(SymmetrizeMethod::CopyUpper);
        println!("a_symm\n{}", a_symm);
        assert_fpvec_eq!(&a_symm * &soln, &b);
    }
    #[test]
    fn test_solve_symm_singular() {
        let m = 6;
        let a = generate_singular_matrix(m, SingularMethod::Zeros);
        let a_symm = a.to_symmetric(SymmetrizeMethod::CopyUpper);
        println!("{}", a_symm);

        let b = Matrix::randsn(m, 1);

        let solve_res = solve_symm_driver(&a_symm, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "singular", "SolveError");
    }
    #[test]
    fn test_solve_symm_invalidrhs() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        let a_symm = a.to_symmetric(SymmetrizeMethod::CopyUpper);
        println!("{}", a_symm);

        let b = Matrix::randsn(m + 1, 1); // invalud number of rows

        let solve_res = solve_symm_driver(&a_symm, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "right-hand side", "SolveError");
    }

    fn solve_approx_driver(a: &Matrix, b: &Matrix) -> Result<ApproxSoln<Matrix>> {
        a.solve_approx(b).map(|x| {
            assert_eq!(x.soln.dims(), (a.ncols(), b.ncols()));
            x
        })
    }
    #[test]
    fn test_solve_approx_square() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let approx_soln = solve_approx_driver(&a, &b).expect("solve_approx failed unexpectedly");
        assert!(approx_soln.resid.is_none());
        let x = approx_soln.soln;

        assert_eq!(x.dims(), (m, 1));
        println!("a*x\n{}\nb\n{}", &a * &x, &b);
        assert_fpvec_eq!(&a * &x, &b);
    }
    #[test]
    fn test_solve_approx_wide() {
        let (m, n) = (6, 8);
        let a = Matrix::randsn(m, n);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let approx_soln = solve_approx_driver(&a, &b).expect("solve_approx failed unexpectedly");
        assert!(approx_soln.resid.is_none());
        let x = approx_soln.soln;

        assert_eq!(x.dims(), (n, 1));
        println!("a*x\n{}\nb\n{}", &a * &x, &b);
        assert_fpvec_eq!(&a * &x, &b);
    }
    #[test]
    fn test_solve_approx_narrow() {
        let (m, n) = (8, 6);
        let a = Matrix::randsn(m, n);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let approx_soln = solve_approx_driver(&a, &b).expect("solve_approx failed unexpectedly");

        // make sure residual provided by solver matches computed error
        assert!(approx_soln.resid.is_some());
        let mut resid_vec = approx_soln.resid.unwrap();
        assert_eq!(resid_vec.len(), 1);
        let resid = resid_vec.pop().unwrap();
        println!("{}", resid);
        assert!(resid > 0.0);
        let x = approx_soln.soln;

        assert_eq!(x.dims(), (n, 1));
        println!("a*x\n{}\nb\n{}", &a * &x, &b);
        let ax = &a * &x;
        assert_eq!(ax.dims(), (m, 1));
        let mut sumsq = 0.0;
        for i in 0..m {
            let err = ax.get(i, 0).unwrap() - b.get(i, 0).unwrap();
            sumsq += err * err;
        }
        println!("resid:{} sumsq:{} diff:{}", resid, sumsq, (sumsq - resid).abs());
        assert!((sumsq - resid).abs() < 1e-8);
    }
    #[test]
    fn test_solve_approx_rank_deficient() {
        let (m, n) = (8, 6);
        let a = generate_rank_deficient_matrix(m, n, SingularMethod::Zeros);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let solve_res = solve_approx_driver(&a, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "rank-deficient", "SolveError");
    }
    #[test]
    fn test_solve_approx_invalidrhs() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        println!("{}", a);

        let b = Matrix::randsn(m + 1, 1);

        let solve_res = solve_approx_driver(&a, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "right-hand side", "SolveError");
    }

    fn solve_driver(a: &Matrix, b: &Matrix) -> Result<Matrix> {
        a.solve(b).map(|x| {
            assert_eq!(x.dims(), (a.ncols(), b.ncols()));
            x
        })
    }
    #[test]
    fn test_solve_square() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let x = solve_driver(&a, &b).expect("solve failed unexpectedly");

        assert_eq!(x.dims(), (m, 1));
        println!("a*x\n{}\nb\n{}", &a * &x, &b);
        assert_fpvec_eq!(&a * &x, &b);
    }
    #[test]
    fn test_solve_wide() {
        let (m, n) = (6, 8);
        let a = Matrix::randsn(m, n);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let x = solve_driver(&a, &b).expect("solve failed unexpectedly");

        assert_eq!(x.dims(), (n, 1));
        println!("a*x\n{}\nb\n{}", &a * &x, &b);
        assert_fpvec_eq!(&a * &x, &b);
    }
    #[test]
    fn test_solve_narrow() {
        let (m, n) = (8, 6);
        let a = Matrix::randsn(m, n);
        println!("{}", a);

        let b = Matrix::randsn(m, 1);

        let x = solve_driver(&a, &b).expect("solve failed unexpectedly");

        // this call doesn't provide the residual, so make an extra call to the approx driver to
        // get it
        let approx_soln = solve_approx_driver(&a, &b).expect("solve_approx failed unexpectedly");

        assert!(approx_soln.resid.is_some());
        let mut resid_vec = approx_soln.resid.unwrap();
        assert_eq!(resid_vec.len(), 1);
        let resid = resid_vec.pop().unwrap();
        println!("{}", resid);
        assert!(resid > 0.0);

        // ok, now we have the resid and we can compare it to the result of solve()
        assert_eq!(x.dims(), (n, 1));
        println!("a*x\n{}\nb\n{}", &a * &x, &b);
        let ax = &a * &x;
        assert_eq!(ax.dims(), (m, 1));
        let mut sumsq = 0.0;
        for i in 0..m {
            let err = ax.get(i, 0).unwrap() - b.get(i, 0).unwrap();
            sumsq += err * err;
        }
        println!("resid:{} sumsq:{} diff:{}", resid, sumsq, (sumsq - resid).abs());
        assert!((sumsq - resid).abs() < 1e-8);
    }
    #[test]
    fn test_solve_invalidrhs() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        println!("{}", a);

        let b = Matrix::randsn(m + 1, 1);

        let solve_res = solve_driver(&a, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "right-hand side", "SolveError");
    }

    fn inverse_driver(a: &Matrix) -> Result<Matrix> {
        a.inverse().map(|x| {
            assert_eq!(x.dims(), a.dims());
            x
        })
    }
    #[test]
    fn test_inverse() {
        let m = 6;
        let a = Matrix::randsn(m, m);
        println!("a\n{}", a);

        let a_inverse = inverse_driver(&a).expect("inverse failed unexpectedly");
        println!("a_inverse\n{}\na*a_inverse\n{}", a_inverse, &a * &a_inverse);

        assert_fpvec_eq!(&a * &a_inverse, Matrix::eye(6), 1e-8);
    }
    #[test]
    fn test_inverse_nonsquare() {
        let (m, n) = (8, 6);
        let a = Matrix::randsn(m, n);
        println!("{}", a);

        let inverse_res = inverse_driver(&a);

        assert_error!(inverse_res, ErrorKind::SolveError, "non-square", "SolveError");
    }
    #[test]
    fn test_inverse_singular() {
        let m = 6;
        let a = generate_singular_matrix(m, SingularMethod::Zeros);
        println!("{}", a);

        let inverse_res = inverse_driver(&a);

        assert_error!(inverse_res, ErrorKind::SolveError, "singular", "SolveError");
    }


    fn gram_solve_driver(a: &Matrix, b: &Matrix) -> Result<Matrix> {
        a.gram_solve(b).map(|x| {
            assert_eq!(x.dims(), (a.ncols(), b.ncols()));
            x
        })
    }

    #[test]
    fn test_gram_solve() {
        let a = Matrix::ones(5, 1).hcat(
            &Matrix::from_vec(vec![0.45642, 0.86603, 0.38062, 0.62465, 0.15748], 5, 1));
        let b = Matrix::from_vec(vec![0.886446, 0.096545], 2, 1);

        let x = gram_solve_driver(&a, &b).expect("gram_solve failed");

        let expected_soln = mat![0.78168; -1.21599];
        println!("x\n{}\nexpected\n{}", x, expected_soln);
        assert_fpvec_eq!(x, expected_soln, 1e-5);
        println!("a'*a*x\n{}\nb\n{}", a.t() * &a * &x, &b);
        assert_fpvec_eq!(a.t() * &a * &x, &b);
    }
    #[test]
    fn test_gram_solve_singular() {
        let a = generate_singular_matrix(6, SingularMethod::Zeros);
        println!("{}", a);
        let b = Matrix::randsn(6, 1);

        let solve_res = gram_solve_driver(&a, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "singular", "SolveError");
    }
    #[test]
    fn test_gram_solve_invalidrhs() {
        let (m, nrhs) = (6, 1);
        let a = Matrix::randsn(m, m);
        let b = Matrix::randsn(m + 1, nrhs); // incorrect size

        let solve_res = gram_solve_driver(&a, &b);

        assert_error!(solve_res, ErrorKind::SolveError, "right-hand side", "SolveError");
    }
}
