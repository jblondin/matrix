use lapack;
use lapack::c::Layout;

use errors::*;

use Matrix;

#[inline]
fn min(x: usize, y: usize) -> usize { if x < y { x } else { y } }

pub trait Compose<T> {
    fn compose(&self) -> T;
}

pub fn lapack_to_c_indexing(v: &Vec<i32>) -> Vec<usize> {
    v.iter().map(|&i| i as usize - 1).collect()
}
pub fn c_to_lapack_indexing(v: &Vec<usize>) -> Vec<i32> {
    v.iter().map(|&u| u as i32 + 1).collect()
}

#[derive(Debug, Clone)]
pub struct LU<T, P> {
    lu: T,
    ipiv: P,
}
impl LU<Matrix, Vec<usize>> {
    pub fn l(&self) -> Matrix {
        let (m, n) = self.lu.dims();

        let mut l = Matrix::zeros(m, min(m, n));
        for i in 0..m {
            for j in 0..min(i + 1, n) {
                if i == j {
                    l.set(i, j, 1.0).unwrap();
                } else {
                    l.set(i, j, self.lu.get(i, j).unwrap()).unwrap();
                }
            }
        }
        l
    }
    pub fn u(&self) -> Matrix {
        let (m, n) = self.lu.dims();

        let mindim = min(m, n);
        let mut u = Matrix::zeros(min(m, n), n);
        for i in 0..mindim {
            for j in i..n {
                u.set(i, j, self.lu.get(i, j).unwrap()).unwrap();
            }
        }
        u
    }
    pub fn p(&self) -> Matrix {
        let m = self.lu.nrows();

        // mutate ipiv into permutation vector
        let mut p: Vec<usize> = Vec::new();
        for i in 0..m { p.push(i); }
        if p.len() > 0 {
            for i in (0..p.len() - 1).rev() {
                if i < self.ipiv.len() {
                    p.swap(self.ipiv[i], i);
                }
            }
        }

        // form into permutation matrix
        let mut pmat = Matrix::zeros(m, m);
        for i in 0..m {
            pmat.set(i, p[i], 1.0).unwrap();
        }
        pmat
    }
    pub fn lu_data(&self) -> &Matrix { &self.lu }
    pub fn ipiv_data(&self) -> &Vec<usize> { &self.ipiv }
    pub fn lu_data_mut(&mut self) -> &mut Matrix { &mut self.lu }
    pub fn ipiv_data_mut(&mut self) -> &mut Vec<usize> { &mut self.ipiv }
}
impl Compose<Matrix> for LU<Matrix, Vec<usize>> {
    fn compose(&self) -> Matrix {
        let mut lu = self.l() * self.u();
        // permute in place instead of generating permutation matrix
        for i in (0..lu.nrows()).rev() {
            if i < self.ipiv.len() && self.ipiv[i] != i {
                for j in 0..lu.ncols() {
                    let valueij= lu.get(i, j).unwrap();
                    let valueipivj = lu.get(self.ipiv[i], j).unwrap();
                    lu.set(self.ipiv[i], j, valueij).unwrap();
                    lu.set(i, j, valueipivj).unwrap();
                }
            }
        }
        lu
    }
}

pub trait LUDecompose: Sized {
    type PermutationStore;

    fn lu(&self) -> Result<LU<Self, Self::PermutationStore>>;
}

impl LUDecompose for Matrix {
    type PermutationStore = Vec<usize>;

    fn lu(&self) -> Result<LU<Matrix, Vec<usize>>> {

        let (m, n) = self.dims();
        let lda = m;

        let lu = self.clone();
        let mut ipiv: Vec<i32> = vec![0; min(m, n)];
        let lu_data = lu.data();
        let info = lapack::c::dgetrf(Layout::ColumnMajor, m as i32, n as i32,
            &mut lu_data.values_mut()[..], lda as i32,
            &mut ipiv[..]);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::DecompositionError(
                format!("LU Decomposition: Invalid call to dgetrf in argument {}", -info))))
        } else {
            Ok(LU {
                lu: lu,
                ipiv: lapack_to_c_indexing(&ipiv),
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct QR<T, S> {
    qr: T,
    tau: S,
}
impl QR<Matrix, Vec<f64>> {
    fn q(&self) -> Matrix {
        let (m,n) = self.qr.dims();
        let mut q = Matrix::eye(m);
        for k in 0..min(m, n) {
            let mut v = Matrix::zeros(m, 1);
            v.set(k, 0, 1.0).unwrap();
            for i in (k + 1)..m {
                v.set(i, 0, self.qr.get(i, k).unwrap()).unwrap();
            }
            let h = Matrix::eye(m) - self.tau[k] * &v * &v.t();
            q = q * h;
        }
        q
    }

    fn r(&self) -> Matrix {
        let (m, n) = self.qr.dims();
        let mut r = Matrix::zeros(m, n);
        for i in 0..min(m, n) {
            for j in i..n {
                r.set(i, j, self.qr.get(i, j).unwrap()).unwrap();
            }
        }
        r
    }
}
impl Compose<Matrix> for QR<Matrix, Vec<f64>> {
    fn compose(&self) -> Matrix {
        &self.q() * &self.r()
    }
}

pub trait QRDecompose: Sized {
    type TauStore;

    fn qr(&self) -> Result<QR<Self, Self::TauStore>>;
}

impl QRDecompose for Matrix {
    type TauStore = Vec<f64>;

    fn qr(&self) -> Result<QR<Matrix, Vec<f64>>> {

        let (m, n) = (self.nrows(), self.ncols());
        let lda = m;

        let mut qr = QR {
            qr: self.clone(),
            tau: vec![0.0; min(m, n)],
        };

        let qr_data = qr.qr.data();
        let info = lapack::c::dgeqrfp(Layout::ColumnMajor, m as i32, n as i32,
            &mut qr_data.values_mut()[..], lda as i32,
            &mut qr.tau[..]);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::DecompositionError(
                format!("QR Decomposition: Invalid call to dgeqrfp in argument {}", -info))))
        } else {
            Ok(qr)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cholesky<T> {
    a: T,
}
impl Cholesky<Matrix> {
    pub fn l(&self) -> Matrix {
        let m = self.a.nrows();
        let mut l = Matrix::zeros(m, m);

        for i in 0..m {
            for j in 0..i + 1 {
                l.set(i, j, self.a.get(i, j).unwrap()).unwrap();
            }
        }

        l
    }
}
impl Compose<Matrix> for Cholesky<Matrix> {
    fn compose(&self) -> Matrix {
        let l = self.l();
        &l * l.t()
    }
}

pub trait CholeskyDecompose: Sized {
    fn chol(&self) -> Result<Cholesky<Matrix>>;
}
impl CholeskyDecompose for Matrix {
    fn chol(&self) -> Result<Cholesky<Matrix>> {

        let (m, n) = self.dims();
        if m != n {
            return Err(Error::from_kind(ErrorKind::DecompositionError(
                "Cholesky decomposition only available for square matrices".to_string())))
        }
        let lda = n;

        let chol = Cholesky {
            a: self.clone()
        };

        let chol_data = chol.a.data();
        let info = lapack::c::dpotrf(Layout::ColumnMajor, b'L', n as i32,
            &mut chol_data.values_mut()[..], lda as i32);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::DecompositionError(
                format!("Cholesky factorization: Invalid call to dpotrf in argument {}", -info))))
        } else if info > 0 {
            Err(Error::from_kind(ErrorKind::DecompositionError(
                format!("Cholesky factorization: Matrix not symmetric positive definite \
                (leading minor of order {} not positive definite)", info))))
        } else {
            Ok(chol)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Eigen<T, U> {
    eigs: (U, U), // real and imaginary parts
    vl: Option<T>,
    vr: Option<T>,
}
impl Eigen<Matrix, Vec<f64>> {
    pub fn eigenvalues(&self) -> (&Vec<f64>, &Vec<f64>) {
        (self.eigenvalues_real(), self.eigenvalues_imag())
    }
    pub fn has_complex_eigenvalues(&self) -> bool {
        self.eigenvalues_imag().iter().fold(0.0, |acc, &f| if acc != f { 1.0 } else { 0.0 }) != 0.0
    }
    pub fn eigenvalues_real(&self) -> &Vec<f64> {
        &self.eigs.0
    }
    pub fn eigenvalues_imag(&self) -> &Vec<f64> {
        &self.eigs.1
    }
    pub fn eigenvectors_left(&self) -> Option<&Matrix> {
        self.vl.as_ref()
    }
    pub fn eigenvectors_right(&self) -> Option<&Matrix> {
        self.vr.as_ref()
    }
}
impl Compose<Matrix> for Eigen<Matrix, Vec<f64>> {
    fn compose(&self) -> Matrix {
        assert!(!self.has_complex_eigenvalues());
        assert!(self.vl.is_some());
        assert!(self.vr.is_some());
        self.eigenvectors_right().unwrap() * Matrix::diag(self.eigenvalues_real())
            * self.eigenvectors_left().unwrap().t()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EigenOptions {
    BothEigenvectors,
    LeftEigenvectorsOnly,
    RightEigenvectorsOnly,
    EigenvaluesOnly
}
pub trait EigenDecompose: Sized {
    type EigenvalueStore;

    fn eigen(&self, opts: EigenOptions) -> Result<Eigen<Self, Self::EigenvalueStore>>;
}
impl EigenDecompose for Matrix {
    type EigenvalueStore = Vec<f64>;

    fn eigen(&self, opts: EigenOptions) -> Result<Eigen<Matrix, Vec<f64>>> {
        let (m, n) = self.dims();
        if m != n {
            return Err(Error::from_kind(ErrorKind::DecompositionError(
                "Eigendecomposition only available for square matrices".to_string())))
        }

        let jobvl = match opts {
            EigenOptions::BothEigenvectors => { b'V' }
            EigenOptions::LeftEigenvectorsOnly => { b'V' }
            _ => { b'N' }
        };
        let jobvr = match opts {
            EigenOptions::BothEigenvectors => { b'V' }
            EigenOptions::RightEigenvectorsOnly => { b'V' }
            _ => { b'N' }
        };

        let inout = self.clone();
        let lda = m;

        let mut wr = vec![0.0; n];
        let mut wi = vec![0.0; n];

        let ldvl = if jobvl == b'V' { n } else { 1 };
        let vl = Matrix::zeros(ldvl, n);

        let ldvr = if jobvr == b'V' { n } else { 1 };
        let vr = Matrix::zeros(ldvr, n);

        let (inout_data, vl_data, vr_data) = (inout.data(), vl.data(), vr.data());
        let info = lapack::c::dgeev(Layout::ColumnMajor, jobvl, jobvr, n as i32,
            &mut inout_data.values_mut()[..], lda as i32,
            &mut wr[..], &mut wi[..],
            &mut vl_data.values_mut()[..], ldvl as i32,
            &mut vr_data.values_mut()[..], ldvr as i32);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::DecompositionError(
                format!("Eigendecomposition: \
                    Invalid call to dgeev in argument {}", -info))))
        } else if info > 0 {
            Err(Error::from_kind(ErrorKind::DecompositionError(
                format!("Eigendecomposition: did not converge starting at element {}", info))))
        } else {
            Ok(Eigen {
                eigs: (wr, wi),
                vl: if jobvl == b'V' { Some(vl) } else { None },
                vr: if jobvr == b'V' { Some(vr) } else { None },
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct SVD<T, U> {
    u: T,
    sigma: U,
    v: T,
}
impl SVD<Matrix, Vec<f64>> {
    pub fn u(&self) -> &Matrix {
        &self.u
    }
    pub fn sigmavec(&self) -> &Vec<f64> {
        &self.sigma
    }
    pub fn sigmamat(&self) -> Matrix {
        let (m, n) = (self.u.ncols(), self.v.nrows());
        let mut sigma = Matrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                if i == j {
                    sigma.set(i, j, self.sigma[i]).unwrap();
                }
            }
        }
        sigma
    }
    pub fn v(&self) -> &Matrix {
        &self.v
    }
}
impl Compose<Matrix> for SVD<Matrix, Vec<f64>> {
    fn compose(&self) -> Matrix {
        &self.u * self.sigmamat() * &self.v.t()
    }
}

pub trait SingularValueDecompose: Sized {
    type SingularValueStore;

    fn svd(&self) -> Result<SVD<Self, Self::SingularValueStore>>;
}

impl SingularValueDecompose for Matrix {
    type SingularValueStore = Vec<f64>;

    fn svd(&self) -> Result<SVD<Matrix, Vec<f64>>> {

        let (m,n) = (self.nrows(), self.ncols());
        let (lda, ldu, ldvt) = (m, m, n);

        let u = Matrix::zeros(ldu, m);
        let vt = Matrix::zeros(ldvt, n);

        let mindim = min(m, n);

        let input = self.clone();
        let mut singular_values = vec![0.0; mindim];

        let (input_data, u_data, vt_data) = (input.data(), u.data(), vt.data());
        let info = lapack::c::dgesdd(Layout::ColumnMajor, b'A', m as i32, n as i32,
            &mut input_data.values_mut()[..], lda as i32, &mut singular_values[..],
            &mut u_data.values_mut()[..], ldu as i32,
            &mut vt_data.values_mut()[..], ldvt as i32);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::DecompositionError(
                format!("Singular Value Decomposition: \
                    Invalid call to dgesdd in argument {}", -info))))
        } else if info > 0 {
            Err(Error::from_kind(ErrorKind::DecompositionError(
                "Singular Value Decomposition: did not converge".to_string())))
        } else {
            Ok(SVD {
                u: u,
                sigma: singular_values,
                v: vt.t().clone()
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{self, Rng};
    use std::cmp::Ordering;

    use subm::SubMatrix;

    fn lu_test_driver(m: usize, n: usize) {
        let a = Matrix::randsn(m, n);

        let lu = a.lu().expect("LU decomposition failed");
        println!("lu\n{}\nl\n{}\nu\n{}\np\n{}", lu.lu, lu.l(), lu.u(), lu.p());
        println!("a\n{}\na_lu\n{}", a, &lu.l() * &lu.u());

        // make sure L is lower trapezoidal
        let l = lu.l();
        for i in 0..m {
            for j in (i + 1)..min(m, n) {
                assert_eq!(l.get(i, j).unwrap(), 0.0);
            }
        }

        // make sure U is upper trapezoidal
        let u = lu.u();
        for i in 1..min(m, n) {
            for j in 0..min(i, n) {
                assert_eq!(u.get(i, j).unwrap(), 0.0);
            }
        }

        // make sure P is a valid permutation matrix
        let p = lu.p();
        let sum = p.iter().fold(0.0, |acc, f| acc + f);
        assert_eq!(sum, m as f64);
        for i in 0..min(m, n) {
            let mut sum_row = 0.0;
            for j in 0..p.ncols() { sum_row += p.get(i, j).unwrap(); }
            assert_eq!(sum_row, 1.0);
        }
        for j in 0..min(m, n) {
            let mut sum_col = 0.0;
            for i in 0..p.nrows() { sum_col += p.get(i, j).unwrap(); }
            assert_eq!(sum_col, 1.0);
        }

        // make sure if composes
        let a_composed = lu.compose();
        assert_fpvec_eq!(a, a_composed);
        println!("a_composed\n{}", a_composed);

        // also try composing with full permutation matrix
        let a_composedfullpm = &lu.p() * &lu.l() * &lu.u();
        assert_fpvec_eq!(a, a_composedfullpm);
        println!("a_composedfullpm\n{}", a_composedfullpm);

    }

    #[test]
    fn test_lu_square() {
        lu_test_driver(6, 6);
    }
    #[test]
    fn test_lu_wide() {
        lu_test_driver(6, 8);
    }
    #[test]
    fn test_lu_narrow() {
        lu_test_driver(8, 6);
    }

    fn qr_test_driver(m: usize, n: usize) {
        let a = Matrix::randsn(m, n);
        println!("a:\n{}", a);

        let qr = a.qr().expect("QR decomposition failed");
        let q = qr.q();
        let r = qr.r();
        println!("q:\n{}\nr:\n{}\n", q, r);

        // make sure Q is unitary
        assert_fpvec_eq!(Matrix::eye(m), &q * &q.t(), 1e-5);
        assert_fpvec_eq!(Matrix::eye(m), &q.t() * &q, 1e-5);

        // make sure R is upper trapezoidal
        for i in 1..m {
            for j in 0..min(i, n) {
                assert_eq!(r.get(i, j).unwrap(), 0.0);
            }
        }

        // make sure QR decomposition recomposes properly
        let a_composed = qr.compose();
        println!("a_composed:\n{}", a_composed);
        assert_fpvec_eq!(a, a_composed);

        // also try manually
        let a_composedmanual = &q * &r;
        println!("a_composedmanual:\n{}", a_composedmanual);
        assert_fpvec_eq!(a, a_composedmanual);
    }

    #[test]
    fn test_qr_square() {
        qr_test_driver(6, 6);
    }
    #[test]
    fn test_qr_wide() {
        qr_test_driver(6, 8);
    }
    #[test]
    fn test_qr_narrow() {
        qr_test_driver(8, 6);
    }

    fn cholesky_test_driver(a: Matrix) -> Result<()> {
        let (m, n) = a.dims();
        let chol_res = a.chol();

        match chol_res {
            Ok(chol)   => {
                // if chol completed, A must be square
                assert_eq!(m, n);

                // ensure L is lower triangular
                let l = chol.l();
                println!("{}", l);
                assert_eq!(l.dims(), (m, m));
                for i in 0..m {
                    for j in (i + 1)..m {
                        assert_eq!(l.get(i, j).unwrap(), 0.0);
                    }
                }

                // make sure the cholesky decomposition recomposes properly
                let a_composed = chol.compose();
                println!("diff: {}", &a - &a_composed);
                assert_fpvec_eq!(a, a_composed);

                // also try to compose manually
                let a_composedmanual = &l * l.t();
                println!("diff: {}", &a - &a_composedmanual);
                assert_fpvec_eq!(a, a_composedmanual);

                Ok(())
            }
            Err(e)  => { Err(e) }
        }
    }

    fn rand_symm_psd(m: usize) -> Matrix {
        let a_nonpsd = Matrix::randsn(m, m);
        &a_nonpsd * a_nonpsd.t() + m as f64 * Matrix::eye(m)
    }

    #[test]
    fn test_chol() {
        let m = 6;
        let a = rand_symm_psd(m);

        assert!(cholesky_test_driver(a).is_ok());
    }

    #[test]
    fn test_chol_nonsquare() {
        let (m, n) = (8, 6);
        let a_nonsquare = Matrix::randsn(m, n);

        let chol_res = cholesky_test_driver(a_nonsquare);
        assert!(chol_res.is_err());
        let e = chol_res.unwrap_err();
        println!("{:?}", e.kind());
        match *e.kind() {
            ErrorKind::DecompositionError(ref m) => {
                assert!(m.find("square").is_some());
            },
            _ => { panic!(format!("Expected DecompositionError, found: {:?}", e.kind())) }
        }
    }

    #[test]
    fn test_chol_nonposdef() {
        let m = 6;
        let mut a = rand_symm_psd(m);

        // set a value on the diagonal to negative, which will ensure matrix is not PSD
        let prev_value = a.get(3, 3).unwrap();
        a.set(3, 3, -1.0 * prev_value).unwrap();

        let chol_res = cholesky_test_driver(a);
        assert!(chol_res.is_err());
        let e = chol_res.unwrap_err();
        println!("{:?}", e.kind());
        match *e.kind() {
            ErrorKind::DecompositionError(ref m) => {
                assert!(m.find("positive definite").is_some());
            },
            _ => { panic!(format!("Expected DecompositionError, found: {:?}", e.kind())) }
        }

    }

    fn check_vl(a: &Matrix, vl: &Matrix, eigs: &(&Vec<f64>, &Vec<f64>)) {
        assert!(vl.is_square());
        assert_eq!(vl.dims(), a.dims());
        for j in 0..vl.ncols() {
            let v = vl.subm(.., j).unwrap();
            assert_fpvec_eq!(v.t() * a, eigs.0[j] * v.t());
        }
    }
    fn check_vr(a: &Matrix, vr: &Matrix, eigs: &(&Vec<f64>, &Vec<f64>)) {
        assert!(vr.is_square());
        assert_eq!(vr.dims(), a.dims());
        for j in 0..vr.ncols() {
            let v = vr.subm(.., j).unwrap();
            assert_fpvec_eq!(a * &v, eigs.0[j] * &v);
        }
    }
    fn eigen_test_driver(a: Matrix, opts: EigenOptions) -> Result<Eigen<Matrix, Vec<f64>>> {
        let (m, n) = a.dims();
        let eigen = a.eigen(opts);

        match eigen {
            Ok(eigen)   => {
                // if eigen completed, A must be square
                assert_eq!(m, n);

                match opts {
                    EigenOptions::BothEigenvectors => {
                        let vl = eigen.eigenvectors_left().unwrap();
                        let vr = eigen.eigenvectors_right().unwrap();
                        let eigs = eigen.eigenvalues();

                        // not dealing with complex eigenvalues currently
                        assert!(!eigen.has_complex_eigenvalues());

                        // check if eigenvalues / vectors are valid
                        check_vl(&a, &vl, &eigs);
                        check_vr(&a, &vr, &eigs);

                        // make sure the eigendecomposition recomposes properly
                        let a_composed = eigen.compose();
                        println!("diff: {}", &a - &a_composed);
                        assert_fpvec_eq!(a, a_composed);

                        // also try to compose manually
                        let a_composedmanual = vr * Matrix::diag(&eigs.0) * vl.t();
                        println!("diff: {}", &a - &a_composedmanual);
                        assert_fpvec_eq!(a, a_composedmanual);

                    }
                    EigenOptions::LeftEigenvectorsOnly => {
                        let vl = eigen.eigenvectors_left().unwrap();
                        assert!(eigen.eigenvectors_right().is_none());
                        check_vl(&a, &vl, &eigen.eigenvalues());
                    }
                    EigenOptions::RightEigenvectorsOnly => {
                        let vr = eigen.eigenvectors_right().unwrap();
                        assert!(eigen.eigenvectors_left().is_none());
                        check_vr(&a, &vr, &eigen.eigenvalues());
                    }
                    EigenOptions::EigenvaluesOnly => {
                        assert!(eigen.eigenvectors_left().is_none());
                        assert!(eigen.eigenvectors_right().is_none());
                    }
                }

                Ok(eigen)
            }
            Err(e)  => { Err(e) }
        }
    }

    #[test]
    fn test_eigen_both_eigenvectors() {
        let (m, n) = (8, 6);
        let a = Matrix::randsn(m, n);
        let b = a.t() * a;
        println!("{}", b);

        eigen_test_driver(b, EigenOptions::BothEigenvectors).unwrap();
    }
    #[test]
    fn test_eigen_left_eigenvectors() {
        let (m, n) = (6, 8);
        let a = Matrix::randsn(m, n);
        let b = a.t() * a;

        eigen_test_driver(b, EigenOptions::LeftEigenvectorsOnly).unwrap();
    }
    #[test]
    fn test_eigen_right_eigenvectors() {
        let (m, n) = (6, 8);
        let a = Matrix::randsn(m, n);
        let b = a.t() * a;

        eigen_test_driver(b, EigenOptions::RightEigenvectorsOnly).unwrap();
    }
    #[test]
    fn test_eigen_no_eigenvectors() {
        let (m, n) = (6, 8);
        let a = Matrix::randsn(m, n);
        let b = a.t() * a;

        eigen_test_driver(b, EigenOptions::EigenvaluesOnly).unwrap();
    }
    #[test]
    fn test_eigenvalues() {
        let (m, n) = (8, 6);
        let a = Matrix::randsn(m, n);
        let b = a.t() * &a;
        println!("{}", (&b - b.t()).iter().fold(0.0, |acc, f| acc + f));

        let eigen = b.eigen(EigenOptions::EigenvaluesOnly).expect("Eigendecomposition failed");
        let svd = a.svd().expect("SVD failed");

        // eigenvalues of A'A should be squares of singular values of A
        assert!(!eigen.has_complex_eigenvalues());
        let mut eigenvalues = eigen.eigenvalues_real().clone();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut svs_sqrd = svd.sigma.iter().map(|f| f * f).collect::<Vec<f64>>();
        svs_sqrd.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("eigen: {:?}", eigenvalues);
        println!("svs_sqrd: {:?}", svs_sqrd);
        println!("diff: {:?}", eigenvalues.iter().zip(&svs_sqrd).map(|(e, s)| e - s)
            .collect::<Vec<f64>>());
        assert_fpvec_eq!(eigenvalues, svs_sqrd);
    }
    #[test]
    fn test_eigen_nonsquare() {
        let (m, n) = (8, 6);
        let a_nonsquare = Matrix::randsn(m, n);

        let eigen_res = eigen_test_driver(a_nonsquare, EigenOptions::BothEigenvectors);
        assert!(eigen_res.is_err());
        let e = eigen_res.unwrap_err();
        println!("{:?}", e.kind());
        match *e.kind() {
            ErrorKind::DecompositionError(ref m) => {
                assert!(m.find("square").is_some());
            },
            _ => { panic!(format!("Expected DecompositionError, found: {:?}", e.kind())) }
        }
    }


    fn svd_test_driver(u: Matrix, sigma_diag: Vec<f64>, v: Matrix) {
        let (m, n) = (u.nrows(), v.nrows());
        assert_eq!(u.nrows(), u.ncols());
        assert_eq!(v.nrows(), v.ncols());
        assert_eq!(sigma_diag.len(), min(m, n));

        // make sure input U is unitary
        assert_fpvec_eq!(Matrix::eye(m), &u * &u.t(), 1e-5);
        assert_fpvec_eq!(Matrix::eye(m), &u.t() * &u, 1e-5);

        // make sure input V is unitary
        assert_fpvec_eq!(Matrix::eye(n), &v * &v.t(), 1e-5);
        assert_fpvec_eq!(Matrix::eye(n), &v.t() * &v, 1e-5);

        let mut sigma = Matrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                if i == j {
                    sigma.set(i, j, sigma_diag[i]).unwrap();
                }
            }
        }

        // compose the matrix
        let mat = &u * &sigma * &v.t();
        println!("mat:\n{}", mat);

        // decompose it back to get the singular values
        let svd = mat.svd().expect("SVD failed");
        println!("{:#?}", svd);

        // make sure singular values are identical
        assert_fpvec_eq!(svd.sigma, sigma_diag);
        println!("orig sigma: {:?}\ncomputed sigma: {:?}", sigma_diag, svd.sigma);

        // make sure return U and V matrices are unitary
        assert_fpvec_eq!(Matrix::eye(m), &svd.u * &svd.u.t(), 1e-5);
        assert_fpvec_eq!(Matrix::eye(m), &svd.u.t() * &svd.u, 1e-5);
        assert_fpvec_eq!(Matrix::eye(n), &svd.v * &svd.v.t(), 1e-5);
        assert_fpvec_eq!(Matrix::eye(n), &svd.v.t() * &svd.v, 1e-5);

        // make sure the SVD recomposes properly
        let mat_composed = svd.compose();
        println!("diff: {}", &mat - &mat_composed);
        assert_fpvec_eq!(mat, mat_composed);

        // also try to compose manually
        let mat_composedmanual = &svd.u * &svd.sigmamat() * &svd.v.t();
        println!("diff: {}", &mat - &mat_composedmanual);
        assert_fpvec_eq!(mat, mat_composedmanual);
    }

    fn generate_singular_values(m: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();

        let mut sigma_diag: Vec<f64> = vec![];
        for _ in 0..m {
            sigma_diag.push(rng.gen());
        }
        (&mut sigma_diag).sort_by(|&x, &y| {
            if x == y { Ordering::Equal }
            else if x < y { Ordering::Greater }
            else { Ordering::Less }
        });
        sigma_diag
    }

    #[test]
    fn test_svd_square() {

        let u = mat![-0.498315,  0.128598, -0.301661, -0.427864,  0.155238,  0.661044;
                     -0.091854, -0.497697,  0.140542, -0.738935,  0.099853, -0.410016;
                      0.285827,  0.161231, -0.737995, -0.048587,  0.503824, -0.302443;
                      0.506196,  0.116019,  0.492270, -0.158451,  0.587474,  0.343138;
                      0.636511, -0.152503, -0.279146, -0.285588, -0.552441,  0.326989;
                     -0.012949, -0.820395, -0.156077,  0.402342,  0.248546,  0.280661];

        let v = mat![-0.187616,  0.280013,  0.193183,  0.828045, -0.337414,  0.222637;
                      0.429073, -0.416271, -0.064585,  0.358640, -0.194242, -0.687088;
                     -0.598013,  0.427246, -0.155716, -0.098051, -0.070811, -0.648819;
                     -0.495611, -0.559239,  0.658957, -0.047464,  0.028993, -0.065597;
                     -0.153013, -0.251152, -0.292822, -0.275479, -0.848193,  0.180126;
                     -0.392472, -0.435827, -0.643656,  0.312992,  0.350907,  0.143626];

        svd_test_driver(u, generate_singular_values(6), v);
    }

    #[test]
    fn test_svd_narrow() {

        let u = mat![-0.498315,  0.128598, -0.301661, -0.427864,  0.155238,  0.661044;
                     -0.091854, -0.497697,  0.140542, -0.738935,  0.099853, -0.410016;
                      0.285827,  0.161231, -0.737995, -0.048587,  0.503824, -0.302443;
                      0.506196,  0.116019,  0.492270, -0.158451,  0.587474,  0.343138;
                      0.636511, -0.152503, -0.279146, -0.285588, -0.552441,  0.326989;
                     -0.012949, -0.820395, -0.156077,  0.402342,  0.248546,  0.280661];

        let v =  mat![-0.725341,  0.527853, -0.441127, -0.025669;
                      -0.597937, -0.367570,  0.514698,  0.492392;
                      -0.340529, -0.442671,  0.078282, -0.825805;
                      -0.019763, -0.624745, -0.731003,  0.273747];

        svd_test_driver(u, generate_singular_values(4), v);
    }

    #[test]
    fn test_svd_wide() {

        let u =  mat![-0.725341,  0.527853, -0.441127, -0.025669;
                      -0.597937, -0.367570,  0.514698,  0.492392;
                      -0.340529, -0.442671,  0.078282, -0.825805;
                      -0.019763, -0.624745, -0.731003,  0.273747];

        let v = mat![-0.498315,  0.128598, -0.301661, -0.427864,  0.155238,  0.661044;
                     -0.091854, -0.497697,  0.140542, -0.738935,  0.099853, -0.410016;
                      0.285827,  0.161231, -0.737995, -0.048587,  0.503824, -0.302443;
                      0.506196,  0.116019,  0.492270, -0.158451,  0.587474,  0.343138;
                      0.636511, -0.152503, -0.279146, -0.285588, -0.552441,  0.326989;
                     -0.012949, -0.820395, -0.156077,  0.402342,  0.248546,  0.280661];

        svd_test_driver(u, generate_singular_values(4), v);
    }

}
