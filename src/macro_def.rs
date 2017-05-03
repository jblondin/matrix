// Example:
// let a = mat![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
// assert_eq!(a.dims(), (3, 4));
macro_rules! mat {
    [$( $( $x:expr ),* );*] => {{
        let mut v: Vec<Vec<f64>> = Vec::new();
        let mut nrows = 0;
        let mut ncols = 0;
        $(
            v.push(Vec::new());
            let mut tmp_ncols = 0;
            $(
                v[nrows].push($x as f64);
                tmp_ncols += 1;
            )*
            if nrows > 0 {
                assert_eq!(ncols, tmp_ncols);
            }
            ncols = tmp_ncols;
            nrows += 1;
        )*

        let mut data: Vec<f64> = Vec::new();
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(v[i][j]);
            }
        }

        Matrix::from_vec(data, nrows, ncols)
    }}
}

#[cfg(test)]
mod tests {
    use Matrix;

    #[test]
    fn test_macro() {
        let a = mat![1, 2, 3, 4; 5, 6, 7, 8];
        assert_eq!(a.dims(), (2, 4));

        assert_eq!(a.get(0, 0).unwrap(), 1.0);
        assert_eq!(a.get(0, 1).unwrap(), 2.0);
        assert_eq!(a.get(0, 2).unwrap(), 3.0);
        assert_eq!(a.get(0, 3).unwrap(), 4.0);
        assert_eq!(a.get(1, 0).unwrap(), 5.0);
        assert_eq!(a.get(1, 1).unwrap(), 6.0);
        assert_eq!(a.get(1, 2).unwrap(), 7.0);
        assert_eq!(a.get(1, 3).unwrap(), 8.0);
        println!("{:#?}", a);

        let a = mat![1, 2; 3.0, 4; 5.5, 6];
        assert_eq!(a.dims(), (3, 2));

        assert_eq!(a.get(0, 0).unwrap(), 1.0);
        assert_eq!(a.get(0, 1).unwrap(), 2.0);
        assert_eq!(a.get(1, 0).unwrap(), 3.0);
        assert_eq!(a.get(1, 1).unwrap(), 4.0);
        assert_eq!(a.get(2, 0).unwrap(), 5.5);
        assert_eq!(a.get(2, 1).unwrap(), 6.0);
        println!("{:#?}", a);

    }
}
