use lapack;
use lapack::c::Layout;

use Matrix;

#[inline]
fn min(x: usize, y: usize) -> usize { if x < y { x } else { y } }

pub trait Compose<T> {
    fn compose(&self) -> T;
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

    fn lu(&self) -> LU<Self, Self::PermutationStore>;
}

impl LUDecompose for Matrix {
    type PermutationStore = Vec<usize>;

    fn lu(&self) -> LU<Matrix, Vec<usize>> {

        let (m, n) = self.dims();
        let lda = m;

        let lu = self.clone();
        let mut ipiv: Vec<i32> = vec![0; min(m, n)];
        lapack::c::dgetrf(Layout::ColumnMajor, m as i32, n as i32,
            &mut lu.data.values.borrow_mut()[..], lda as i32,
            &mut ipiv[..]);

        LU {
            lu: lu,
            ipiv: ipiv.iter().map(|&i| i as usize - 1).collect(),
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

    fn qr(&self) -> QR<Self, Self::TauStore>;
}

impl QRDecompose for Matrix {
    type TauStore = Vec<f64>;

    fn qr(&self) -> QR<Matrix, Vec<f64>> {

        let (m, n) = (self.nrows(), self.ncols());
        let lda = m;

        let mut qr = QR {
            qr: self.clone(),
            tau: vec![0.0; min(m, n)],
        };

        lapack::c::dgeqrfp(Layout::ColumnMajor, m as i32, n as i32,
            &mut qr.qr.data.values.borrow_mut()[..], lda as i32,
            &mut qr.tau[..]);

        qr
    }
}

#[derive(Debug, Clone)]
pub struct SVD<T, U> {
    pub u: T,
    pub sigma: U,
    pub v: T,
}
pub trait SingularValueDecompose: Sized {
    type SingularValueStore;

    fn svd(&self) -> SVD<Self, Self::SingularValueStore>;
}

impl SingularValueDecompose for Matrix {
    type SingularValueStore = Vec<f64>;

    fn svd(&self) -> SVD<Matrix, Vec<f64>> {

        let (m,n) = (self.nrows(), self.ncols());
        let (lda, ldu, ldvt) = (m, m, n);

        let u = Matrix::zeros(ldu, m);
        let vt = Matrix::zeros(ldvt, n);

        let mindim = min(m, n);

        let input = self.clone();
        let mut singular_values = vec![0.0; mindim];

        lapack::c::dgesdd(Layout::ColumnMajor, b'A', m as i32, n as i32,
            &mut input.data.values.borrow_mut()[..], lda as i32, &mut singular_values[..],
            &mut u.data.values.borrow_mut()[..], ldu as i32,
            &mut vt.data.values.borrow_mut()[..], ldvt as i32);

        SVD {
            u: u,
            sigma: singular_values,
            v: vt.t().clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{self, Rng};
    use std::cmp::Ordering;

    fn lu_test_driver(m: usize, n: usize) {
        let a = Matrix::randsn(m, n);

        let lu = a.lu();
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

        let qr = a.qr();
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

    #[test]
    fn test_svd() {
        let mut rng = rand::thread_rng();

        let u = mat![-0.498315,  0.128598, -0.301661, -0.427864,  0.155238,  0.661044;
                     -0.091854, -0.497697,  0.140542, -0.738935,  0.099853, -0.410016;
                      0.285827,  0.161231, -0.737995, -0.048587,  0.503824, -0.302443;
                      0.506196,  0.116019,  0.492270, -0.158451,  0.587474,  0.343138;
                      0.636511, -0.152503, -0.279146, -0.285588, -0.552441,  0.326989;
                     -0.012949, -0.820395, -0.156077,  0.402342,  0.248546,  0.280661];
        // make sure input U is unitary
        assert_fpvec_eq!(Matrix::eye(6), &u * &u.t(), 1e-5);
        assert_fpvec_eq!(Matrix::eye(6), &u.t() * &u, 1e-5);

        let mut sigma_diag: Vec<f64> = vec![];
        for _ in 0..6 {
            sigma_diag.push(rng.gen());
        }
        (&mut sigma_diag).sort_by(|&x, &y| {
            if x == y { Ordering::Equal }
            else if x < y { Ordering::Greater }
            else { Ordering::Less }
        });
        let sigma = Matrix::diag(&sigma_diag);
        println!("{:#?}", sigma);

        let v = mat![-0.187616,  0.280013,  0.193183,  0.828045, -0.337414,  0.222637;
                      0.429073, -0.416271, -0.064585,  0.358640, -0.194242, -0.687088;
                     -0.598013,  0.427246, -0.155716, -0.098051, -0.070811, -0.648819;
                     -0.495611, -0.559239,  0.658957, -0.047464,  0.028993, -0.065597;
                     -0.153013, -0.251152, -0.292822, -0.275479, -0.848193,  0.180126;
                     -0.392472, -0.435827, -0.643656,  0.312992,  0.350907,  0.143626];
        // make sure input V is unitary
        assert_fpvec_eq!(Matrix::eye(6), &v * &v.t(), 1e-5);
        assert_fpvec_eq!(Matrix::eye(6), &v.t() * &v, 1e-5);

        // compose the matrix
        let m = &u * &sigma * &v.t();
        println!("m:\n{}", m);

        // decompose it back to get the singular values
        let svd = m.svd();
        println!("{:#?}", svd);

        // make sure singular values are identical
        assert_fpvec_eq!(svd.sigma, sigma_diag);

        // make sure return U and V matrices are unitary
        assert_fpvec_eq!(Matrix::eye(6), &svd.u * &svd.u.t(), 1e-5);
        assert_fpvec_eq!(Matrix::eye(6), &svd.u.t() * &svd.u, 1e-5);
        assert_fpvec_eq!(Matrix::eye(6), &svd.v * &svd.v.t(), 1e-5);
        assert_fpvec_eq!(Matrix::eye(6), &svd.v.t() * &svd.v, 1e-5);

        // make sure the SVD recomposes properly
        let m_composed = &svd.u * Matrix::diag(&svd.sigma) * &svd.v.t();

        println!("diff: {}", &m - &m_composed);
        assert_fpvec_eq!(m, m_composed);
    }
}
