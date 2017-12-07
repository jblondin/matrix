use std::f64;

use lapack;
use lapack::c::Layout;

use Matrix;
use EigenDecompose;
use EigenOptions;
use SingularValueDecompose;
use solve::Solve;

use errors::*;

/// Valid vector norm types
#[derive(Debug, Clone, Copy)]
pub enum Norm {
    /// L1 (absolute value) norm
    L1,
    /// L2 (Euclidean) norm
    L2,
    /// L2 norm, squared (to avoid potentially expensive square root operations)
    L2Sqrd,
    /// Ininity (maximum) norm
    Inf,
    /// Negative infinity (minimum) norm
    NegInf,
    /// General p-norm with specified coefficient
    P(f64),
}
/// Trait providing vector norms
pub trait VectorNorm {
    /// Compute the specified norm type on the vector.
    ///
    /// #Panics
    /// Panics if implementing type is not actually a vector.
    fn norm(&self, norm_type: Norm) -> f64;
}

impl VectorNorm for Matrix {
    fn norm(&self, norm_type: Norm) -> f64 {
        assert!(self.is_vector());
        self.entrywise_norm(norm_type)
    }
}

/// Valid matrix norm types
#[derive(Debug, Clone, Copy)]
pub enum MatNorm {
    //induced norms
    /// L1-Induced matrix norm
    InducedL1,
    /// L2-Induced matrix norm
    InducedL2,
    /// Infinity-induced matrix norm
    InducedInf,
    /// Spectral norm (identical to InducedL2 norm)
    Spectral,

    //entrywise
    /// General entrywise matrix norm, using specified vector norm
    Entrywise(Norm),
    /// L_{2,1} norm (sum of Euclidean norms of the columns of the matrix)
    L21,
    /// L_{p, q} norm
    Lpq(f64, f64),

    /// Frobenius norm (identical to Entrywise(L2) or Lpq(2,2))
    Frobenius,
    /// Max norm (identifier to Entrywise(Inf))
    Max,

    // schatten norms
    /// General Schatten norm with specified vector norm
    Schatten(Norm),
    /// Nuclear norm (identical to Schatten(L1)
    Nuclear,

}
/// Provides method for computing matrix norms.
pub trait MatrixNorm {
    /// Compute the specified matrix norm type
    fn matrix_norm(&self, norm_type: MatNorm) -> f64;
}

impl Matrix {
    fn induced_l1_norm(&self) -> f64 {
        self.map_columns(|mat| mat.norm(Norm::L1)).iter().fold(0.0, |acc, f| acc.max(f))
    }
    fn induced_l2_norm(&self) -> f64 {
        let svd = self.svd().expect("Extracting singular values failed while computing L2 norm");
        svd.sigmavec().iter().fold(f64::NEG_INFINITY, |acc, &f| acc.max(f))
    }
    fn induced_inf_norm(&self) -> f64 {
        self.map_rows(|mat| mat.norm(Norm::L1)).iter().fold(0.0, |acc, f| acc.max(f))
    }
    fn entrywise_norm(&self, norm_type: Norm) -> f64 {
        match norm_type {
            Norm::L1     => {
                self.iter().fold(0.0, |acc, f| acc + f.abs())
            }
            Norm::L2     => {
                self.iter().fold(0.0, |acc, f| acc + f * f).sqrt()
            }
            Norm::L2Sqrd => {
                self.iter().fold(0.0, |acc, f| acc + f * f)
            }
            Norm::Inf    => {
                self.iter().fold(0.0, |acc, f| f.abs().max(acc))
            }
            Norm::NegInf => {
                self.iter().fold(f64::INFINITY, |acc, f| f.abs().min(acc))
            }
            Norm::P(p)   => {
                assert!(p >= 1.0);
                self.iter().fold(0.0, |acc, f| acc + f.abs().powf(p)).powf(1.0 / p)
            }
        }
    }
    fn l21_norm(&self) -> f64 {
        self.map_columns(|mat| mat.norm(Norm::L2)).norm(Norm::L1)
    }
    fn lpq_norm(&self, p: f64, q: f64) -> f64 {
        assert!(p >= 1.0 && q >= 1.0);
        self.map_columns(|mat| mat.norm(Norm::P(p))).norm(Norm::P(q))
    }
    fn schatten_norm(&self, norm_type: Norm) -> f64 {
        let svd = self.svd().expect("Extracting singular values failed while computing \
            Schatten norm");
        let sigmas = svd.sigmavec();
        Matrix::from_vec(sigmas.clone(), sigmas.len(), 1).norm(norm_type)
    }
}
impl MatrixNorm for Matrix {
    fn matrix_norm(&self, norm_type: MatNorm) -> f64 {
        match norm_type {
            MatNorm::InducedL1       => { self.induced_l1_norm() }
            MatNorm::InducedL2       => { self.induced_l2_norm() }
            MatNorm::InducedInf      => { self.induced_inf_norm() }
            MatNorm::Spectral        => { self.induced_l2_norm() }
            MatNorm::Entrywise(nt)   => { self.entrywise_norm(nt) }
            MatNorm::L21             => { self.l21_norm() }
            MatNorm::Lpq(p, q)       => { self.lpq_norm(p, q) }
            MatNorm::Frobenius       => { self.entrywise_norm(Norm::L2) }
            MatNorm::Max             => { self.entrywise_norm(Norm::Inf) }
            MatNorm::Schatten(nt)    => { self.schatten_norm(nt) }
            MatNorm::Nuclear         => { self.schatten_norm(Norm::L1) }
        }
    }
}

impl Matrix {
    fn rcond_inner(&self, nt: MatNorm) -> Result<f64> {
        if !self.is_square() {
            return Err(Error::from_kind(ErrorKind::CondError(
                "rcond called with non-square matrix".to_string())))
        }

        let norm_spec = match nt {
            MatNorm::InducedL1 => { b'1' }
            MatNorm::InducedInf => { b'I' }
            _ => { panic!("rcond_inner only allows InducedL1 or InducedInf MatNorm"); }
        };
        let norm_a = self.matrix_norm(nt);

        let m = self.nrows();
        let lda = m;

        let lu = self.clone();
        let mut ipiv: Vec<i32> = vec![0; m];
        let lu_data = lu.data();
        let info = lapack::c::dgetrf(Layout::ColumnMajor, m as i32, m as i32,
            &mut lu_data.values_mut()[..], lda as i32,
            &mut ipiv[..]);

        if info < 0 {
            return Err(Error::from_kind(ErrorKind::CondError(
                format!("rcond: Invalid call to dgetrf in argument {}", -info))))
        };

        let mut rcond = 0.0;
        let info = lapack::c::dgecon(Layout::ColumnMajor, norm_spec, m as i32,
            &lu_data.values()[..], lda as i32, norm_a, &mut rcond);

        if info < 0 {
            Err(Error::from_kind(ErrorKind::CondError(
                format!("rcond: Invalid call to dgecon in argument {}", -info))))
        } else {
            Ok(rcond)
        }
    }
    /// Computes the reciprocal of the L1-norm condition number of the matrix
    pub fn rcond(&self) -> Result<f64> {
        self.rcond_inner(MatNorm::InducedL1)
    }
    /// Computes the recpirocal of the Inf-norm condition number of the matrix
    pub fn rcond_inf(&self) -> Result<f64> {
        self.rcond_inner(MatNorm::InducedInf)
    }

    /// Computes the L2-norm condition number of this matrix
    pub fn cond(&self) -> Result<f64> {
        self.cond_nt(MatNorm::InducedL2)
    }

    /// Computes the condition number using the specified matrix norm
    pub fn cond_nt(&self, norm_type: MatNorm) -> Result<f64> {
        match norm_type {
            MatNorm::InducedL2 | MatNorm::Spectral => {
                // SVD, max sv / min sv
                let svd = self.svd()?;
                let (max_sv, min_sv) = svd.sigmavec().iter().fold(
                    (f64::NEG_INFINITY, f64::INFINITY),
                    |acc, f| (f.max(acc.0), f.min(acc.1))
                );
                Ok(max_sv / min_sv)

            }
            MatNorm::InducedL1 => {
                self.rcond().map(|f| 1.0 / f)
            }
            MatNorm::InducedInf => {
                self.rcond_inf().map(|f| 1.0 / f)
            }
            MatNorm::Frobenius | MatNorm::Entrywise(Norm::L2) => {
                // cond = sqrt(trace(A'*A)) * sqrt(trace(Ainv'*Ainv))
                // B = A' * A
                // Binv = inv(A' * A) = Ainv * Ainv'
                // trace(Ainv' * Ainv) = trace(Ainv * Ainv'), so tr(Binv) = tr(Ainv' * Ainv)
                // cond = sqrt(tr(B)) * sqrt(tr(Binv))
                // eigendecomposition B = Pinv * D * P
                // Binv = inv(Pinv * D * P) = Pinv * Dinv * P
                // Dinv = 1.0 / D
                // cond = sqrt(sum(diag(D))) * sqrt(sum(1.0 / diag(D)))
                let b = self.t() * self;
                let eigendec = b.eigen(EigenOptions::EigenvaluesOnly)?;
                assert!(!eigendec.has_complex_eigenvalues());

                let eigs = eigendec.eigenvalues_real();
                Ok(eigs.iter().fold(0.0, |acc, f| acc + f).sqrt() *
                    eigs.iter().fold(0.0, |acc, f| acc + 1.0 / f).sqrt())
            }
            _ => {
                Ok(self.matrix_norm(norm_type) * self.inverse()?.matrix_norm(norm_type))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use SubMatrix;


    fn precomp_vector() -> Matrix {
        mat![0.77491879, 0.97650547, 0.54472260, -0.20757379, 0.72747111, -0.43498534]
    }
    fn vecnorm_driver_precomputed(rowvec: Matrix, expected: f64, norm_type: Norm) {
        assert!(rowvec.is_row_vector());
        let n = rowvec.norm(norm_type);
        println!("type:{:?} {} ?= {}", norm_type, n, expected);
        assert_fp_eq!(n, expected);

        let colvec = rowvec.t();
        assert!(colvec.is_col_vector());
        let n = colvec.norm(norm_type);
        println!("type:{:?} {} ?= {}", norm_type, n, expected);
        assert_fp_eq!(n, expected);
    }

    #[test]
    fn test_l1norm() {
        vecnorm_driver_precomputed(precomp_vector(), 3.6661771, Norm::L1);
    }
    #[test]
    fn test_l2norm() {
        vecnorm_driver_precomputed(precomp_vector(), 1.6162605, Norm::L2);
    }
    #[test]
    fn test_l2sqrdnorm() {
        vecnorm_driver_precomputed(precomp_vector(), 2.6122981, Norm::L2Sqrd);
    }
    #[test]
    fn test_infnorm() {
        vecnorm_driver_precomputed(precomp_vector(), 0.97650547, Norm::Inf);
    }
    #[test]
    fn test_neginfnorm() {
        vecnorm_driver_precomputed(precomp_vector(), 0.20757379, Norm::NegInf);
    }
    #[test]
    fn test_pnorm() {
        vecnorm_driver_precomputed(precomp_vector(), 2.1036332, Norm::P(1.5));
        vecnorm_driver_precomputed(precomp_vector(), 1.3919007, Norm::P(2.5));
        vecnorm_driver_precomputed(precomp_vector(), 1.2670964, Norm::P(3.0));
        vecnorm_driver_precomputed(precomp_vector(), 1.0399278, Norm::P(6.0));
        vecnorm_driver_precomputed(precomp_vector(), 0.97650566, Norm::P(50.0));
        vecnorm_driver_precomputed(precomp_vector(), 0.97650547, Norm::P(1.0e4));

        let a = precomp_vector();
        vecnorm_driver_precomputed(a.clone(), a.norm(Norm::L1), Norm::P(1.0));
        vecnorm_driver_precomputed(a.clone(), a.norm(Norm::L2), Norm::P(2.0));
        vecnorm_driver_precomputed(a.clone(), a.norm(Norm::Inf), Norm::P(1.0e4));
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_norm_nonvec() {
        let a = mat![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
        a.norm(Norm::L2);
    }
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_norm_pnorm_fail() {
        let a = mat![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
        a.norm(Norm::P(0.5));
    }

    #[test]
    fn test_matnorm_inducedl1() {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        assert_eq!(a.matrix_norm(MatNorm::InducedL1), 18.0);
    }
    #[test]
    fn test_matnorm_inducedl2() {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        assert_fp_eq!(a.matrix_norm(MatNorm::InducedL2), 15.007944);
        assert_eq!(a.matrix_norm(MatNorm::InducedL2), a.matrix_norm(MatNorm::Spectral));
    }
    #[test]
    fn test_matnorm_inducedinf() {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        assert_eq!(a.matrix_norm(MatNorm::InducedInf), 24.0);
    }

    #[test]
    fn test_matnorm_entrywisel1() {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        assert_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::L1)), 45.0);
    }
    #[test]
    fn test_matnorm_entrywisel2() {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::L2)), 16.881943);
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::L2)),
            a.matrix_norm(MatNorm::Frobenius));
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::L2)),
            a.matrix_norm(MatNorm::Lpq(2.0, 2.0)));
    }
    #[test]
    fn test_matnorm_entrywisel2sqrd() {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::L2Sqrd)), 285.0);
    }
    #[test]
    fn test_matnorm_entrywiseinf() {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::Inf)), 9.0);
    }
    #[test]
    fn test_matnorm_entrywisep() {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];

        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::P(1.5))), 23.103503);
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::P(2.5))), 14.141152);
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::P(3.0))), 12.651490);
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::P(6.0))), 9.9636801);
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::P(50.0))), 9.0004984);
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::P(100.0))), 9.0);

        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::P(1.0))),
            a.matrix_norm(MatNorm::Entrywise(Norm::L1)));
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::P(2.0))),
            a.matrix_norm(MatNorm::Entrywise(Norm::L2)));
        assert_fp_eq!(a.matrix_norm(MatNorm::Entrywise(Norm::P(100.0))),
            a.matrix_norm(MatNorm::Entrywise(Norm::Inf)));
    }

    #[test]
    fn test_matnorm_l21() {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        let mn = a.matrix_norm(MatNorm::L21);
        assert_fp_eq!(mn, 28.992661);
        assert_fp_eq!(mn, a.subm(.., 0).unwrap().norm(Norm::L2)
            + a.subm(.., 1).unwrap().norm(Norm::L2)
            + a.subm(.., 2).unwrap().norm(Norm::L2));
    }
    fn lpq_norm_driver(p: f64, q: f64, expected: f64) {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        let mn = a.matrix_norm(MatNorm::Lpq(p, q));
        assert_fp_eq!(mn, expected);
        let b = mat![a.subm(.., 0).unwrap().norm(Norm::P(p));
                     a.subm(.., 1).unwrap().norm(Norm::P(p));
                     a.subm(.., 2).unwrap().norm(Norm::P(p))];
        assert_fp_eq!(mn, b.norm(Norm::P(q)));
    }
    #[test]
    fn test_matnorm_lpq() {
        lpq_norm_driver(1.5, 2.5, 17.410244);
        lpq_norm_driver(2.5, 1.5, 18.815084);
        lpq_norm_driver(1.5, 15.0, 13.078451);
        lpq_norm_driver(15.0, 1.5, 16.685414);
        lpq_norm_driver(1.5, 2.5, 17.410244);
    }
    #[test]
    fn test_matnorm_l21_lpq() {
        let a = Matrix::randsn(6, 6);
        assert_fp_eq!(a.matrix_norm(MatNorm::L21), a.matrix_norm(MatNorm::Lpq(2.0, 1.0)));

    }

    fn schatten_driver(nt: Norm, expected: f64) -> Matrix {
        let a = mat![-1.0, 2.0, 3.0; 4.0, -5.0, 6.0; 7.0, 8.0, -9.0];
        let a_sigma = mat![15.007944068, 6.90641433855, 3.47319101380];
        let mn = a.matrix_norm(MatNorm::Schatten(nt));
        println!("{} {}", mn, expected);
        assert_fp_eq!(mn, expected);
        println!("{} {}", mn, a_sigma.norm(nt));
        assert_fp_eq!(mn, a_sigma.norm(nt));
        a
    }
    #[test]
    fn test_matnorm_schattenl1() {
        let a = schatten_driver(Norm::L1, 25.387549);
        assert_fp_eq!(a.matrix_norm(MatNorm::Nuclear),
            a.matrix_norm(MatNorm::Schatten(Norm::L1)));
    }
    #[test]
    fn test_matnorm_schattenl2() {
        let a = schatten_driver(Norm::L2, 16.881943);
        assert_fp_eq!(a.matrix_norm(MatNorm::Schatten(Norm::L2)),
            a.matrix_norm(MatNorm::Frobenius));
    }
    #[test]
    fn test_matnorm_schattenl2sqrd() {
        schatten_driver(Norm::L2Sqrd, 285.0);
    }
    #[test]
    fn test_matnorm_schatteninf() {
        let a = schatten_driver(Norm::Inf, 15.007944);
        assert_fp_eq!(a.matrix_norm(MatNorm::Schatten(Norm::Inf)),
            a.matrix_norm(MatNorm::Spectral));
    }
    #[test]
    fn test_matnorm_schattenneginf() {
        schatten_driver(Norm::NegInf, 3.473191);
    }
    #[test]
    fn test_matnorm_schattenp() {
        schatten_driver(Norm::P(1.5), 18.991547);
        schatten_driver(Norm::P(2.5), 15.977532);
        schatten_driver(Norm::P(3.0), 15.538494);
        schatten_driver(Norm::P(6.0), 15.031987);
        schatten_driver(Norm::P(50.0), 15.007944);
        let a = schatten_driver(Norm::P(100.0), 15.007944);

        assert_fp_eq!(a.matrix_norm(MatNorm::Schatten(Norm::P(1.0))),
            a.matrix_norm(MatNorm::Schatten(Norm::L1)));
        assert_fp_eq!(a.matrix_norm(MatNorm::Schatten(Norm::P(2.0))),
            a.matrix_norm(MatNorm::Schatten(Norm::L2)));
        assert_fp_eq!(a.matrix_norm(MatNorm::Schatten(Norm::P(100.0))),
            a.matrix_norm(MatNorm::Schatten(Norm::Inf)));
    }

    #[test]
    fn test_rcond_l1() {
        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0];
        let acond = a.rcond().unwrap();
        println!("{}", acond);

        assert_fp_eq!(acond, 1.0 / 55.5);
    }
    #[test]
    fn test_rcond_inf() {
        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0];
        let acond = a.rcond_inf().unwrap();
        println!("{}", acond);

        assert_fp_eq!(acond, 1.0 / 54.0);
    }
    #[test]
    fn test_rcond_nonsquare() {
        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0; -10.0, -11.0, 12.0];
        let acond_res = a.rcond();

        assert!(acond_res.is_err());
        let e = acond_res.unwrap_err();
        println!("{:?}", e.kind());
        match *e.kind() {
            ErrorKind::CondError(ref m) => {
                assert!(m.find("non-square").is_some());
            },
            _ => { panic!(format!("Expected CondError, found: {:?}", e.kind())) }
        }
    }
    #[test]
    fn test_cond_fro() {
        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0; -10.0, -11.0, 12.0];
        let acond = a.cond_nt(MatNorm::Frobenius).unwrap();
        assert_fp_eq!(acond, 43.39603897);
        assert_fp_eq!(acond, a.cond_nt(MatNorm::Entrywise(Norm::L2)).unwrap());

        let b = a.t() * &a;
        let bcond = b.cond_nt(MatNorm::Frobenius).unwrap();
        assert_fp_eq!(bcond, 1538.9530598);
        assert_fp_eq!(bcond, b.cond_nt(MatNorm::Entrywise(Norm::L2)).unwrap());
    }
    #[test]
    fn test_cond_spectral() {
        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0; -10.0, -11.0, 12.0];
        let acond = a.cond_nt(MatNorm::InducedL2).unwrap();
        println!("{}", acond);
        assert_fp_eq!(acond, 38.59574132);
        assert_fp_eq!(acond, a.cond_nt(MatNorm::Spectral).unwrap());
    }
    #[test]
    fn test_cond_inf() {
        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0; -10.0, -11.0, 12.0];
        let b = a.t() * &a;
        let bcond = b.cond_nt(MatNorm::InducedInf).unwrap();
        println!("{}", bcond);
        assert_fp_eq!(bcond, 1812.4096386);
        assert_fp_eq!(bcond, 1.0 / b.rcond_inf().unwrap());

        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0];
        let acond = a.cond_nt(MatNorm::InducedInf).unwrap();
        assert_fp_eq!(acond, 54.0);
    }
    #[test]
    fn test_cond_l1() {
        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0; -10.0, -11.0, 12.0];
        let b = a.t() * &a;
        let bcond = b.cond_nt(MatNorm::InducedL1).unwrap();
        println!("{}", bcond);
        assert_fp_eq!(bcond, 1812.4096386);
        assert_fp_eq!(bcond, 1.0 / b.rcond().unwrap());

        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0];
        let acond = a.cond_nt(MatNorm::InducedL1).unwrap();
        assert_fp_eq!(acond, 55.5);
    }
    #[test]
    fn test_cond_other() {
        let a = mat![-1.0, 2.0, -3.0; 4.0, 5.0, -6.0; -7.0, 8.0, -9.0; -10.0, -11.0, 12.0];
        let b = a.t() * &a;

        let bcond = b.cond_nt(MatNorm::Entrywise(Norm::L1)).unwrap();
        println!("{}", bcond);
        assert_fp_eq!(bcond, 8394.3105756);

        let bcond = b.cond_nt(MatNorm::Entrywise(Norm::P(2.5))).unwrap();
        println!("{}", bcond);
        assert_fp_eq!(bcond, 1123.3761173);
    }
}
