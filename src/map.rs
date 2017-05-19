use Matrix;
use SubMatrix;

impl Matrix {
    pub fn map<F>(&self, f: F) -> Matrix where F: Fn(f64) -> f64 {
        Matrix::from_vec(self.iter().map(f).collect(), self.nrows(), self.ncols())
    }

    pub fn map_columns<F>(&self, f: F) -> Matrix where F: Fn(Matrix) -> f64 {
        let mut v: Vec<f64> = Vec::new();
        for c in 0..self.ncols() {
            v.push(f(self.subm(.., c).unwrap()));
        }
        Matrix::from_vec(v, 1, self.ncols())
    }

    pub fn map_rows<F>(&self, f: F) -> Matrix where F: Fn(Matrix) -> f64 {
        let mut v: Vec<f64> = Vec::new();
        for r in 0..self.nrows() {
            v.push(f(self.subm(r, ..).unwrap()));
        }
        Matrix::from_vec(v, self.nrows(), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map() {
        let (m, n) = (3, 4);
        let a = mat![1.0, 1.0, 1.0, 1.0; 2.0, 2.0, 2.0, 2.0; 3.0, 3.0, 3.0, 3.0];
        let b = a.map(|f| f.powf(1.5));
        assert_eq!(b.dims(), (m, n));

        for j in 0..n {
            assert_eq!(b.get(0, j).unwrap(), 1.0);
        }
        for j in 0..n {
            assert_fp_eq!(b.get(1, j).unwrap(), 2.8284271);
        }
        for j in 0..n {
            assert_fp_eq!(b.get(2, j).unwrap(), 5.1961524);
        }

    }

    #[test]
    fn test_map_columns() {
        let a = mat![1.0, 2.0, 3.0, 4.0; 10.0, 20.0, 30.0, 40.0; 100.0, 200.0, 300.0, 400.0];
        let b = a.map_columns(|mat| mat.iter().fold(0.0, |acc, f| acc + f));
        assert_eq!(b.dims(), (1, 4));
        assert!(b.is_row_vector());

        assert_eq!(b.get(0, 0).unwrap(), 111.0);
        assert_eq!(b.get(0, 1).unwrap(), 222.0);
        assert_eq!(b.get(0, 2).unwrap(), 333.0);
        assert_eq!(b.get(0, 3).unwrap(), 444.0);
    }

    #[test]
    fn test_map_rows() {
        let a = mat![1.0, 2.0, 3.0, 4.0; 10.0, 20.0, 30.0, 40.0; 100.0, 200.0, 300.0, 400.0];
        let b = a.map_rows(|mat| mat.iter().fold(0.0, |acc, f| acc + f));
        assert_eq!(b.dims(), (3, 1));
        assert!(b.is_col_vector());

        assert_eq!(b.get(0, 0).unwrap(), 10.0);
        assert_eq!(b.get(1, 0).unwrap(), 100.0);
        assert_eq!(b.get(2, 0).unwrap(), 1000.0);
    }
}
