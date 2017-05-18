use Matrix;

#[derive(Debug, Clone, Copy)]
pub enum Norm {
    L1,
    L2,
    L2Sqrd,
    Inf,
    P(f64),
}
pub trait VectorNorm {
    fn norm(&self, norm_type: Norm) -> f64;
}

impl VectorNorm for Matrix {
    fn norm(&self, norm_type: Norm) -> f64 {
        assert!(self.is_vector());

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
            Norm::P(p)   => {
                assert!(p >= 1.0);
                self.iter().fold(0.0, |acc, f| acc + f.abs().powf(p)).powf(1.0 / p)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
