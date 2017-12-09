// Set of tests that should mirror the examples in README

#[macro_use] extern crate wee_matrix as matrix;

use matrix::Matrix;

#[test]
fn test_creation() {
    let a = mat![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
    assert_eq!(a.dims(), (3, 4));

    // the vector in from_vec assumes column-major order
    let b = Matrix::from_vec(vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0],
        3, 4);
    assert_eq!(a.iter().collect::<Vec<_>>(), b.iter().collect::<Vec<_>>());
}

#[test]
fn test_ones_zeros() {
    let a = Matrix::ones(5, 4);
    assert_eq!(a.dims(), (5, 4));
    assert_eq!(a.get(0, 0).unwrap(), 1.0);
    assert_eq!(a.get(0, 3).unwrap(), 1.0);
    assert_eq!(a.get(4, 2).unwrap(), 1.0);
}

#[test]
fn test_vcat_hcat() {
    let a = Matrix::rand(3, 2);
    let b = Matrix::rand(2, 2);
    let a_b = a.vcat(&b);
    assert_eq!(a_b.dims(), (5, 2));

    let c = Matrix::rand(3, 3);
    let a_c = a.hcat(&c);
    assert_eq!(a_c.dims(), (3, 5));
}

#[test]
fn test_ops() {
    let a = Matrix::ones(2, 2);
    let b = Matrix::ones(2, 2);
    let c = &a + &b;
    assert_eq!(c.dims(), (2, 2));
    assert_eq!(c.get(0, 0).unwrap(), 2.0);

    let d = &c * Matrix::zeros(2, 2);
    assert_eq!(d.dims(), (2, 2));
    assert_eq!(d.get(0, 0).unwrap(), 0.0);
}
