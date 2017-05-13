use lapack;
use lapack::c::Layout;

use Matrix;

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

        let min = |x: usize , y: usize| { if x < y { x } else { y } };
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
