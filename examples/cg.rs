extern crate rustimization;

use rustimization::cg_minimizer::CG;

const TRUE_MIN: [f64; 2] = [-1., 2.];

fn f(x: &[f64]) -> f64 {
    (x[0] - TRUE_MIN[0]).powi(4) + (x[1] - TRUE_MIN[1]).powi(4)
}

fn g(x: &[f64]) -> Vec<f64> {
    let mut v = Vec::with_capacity(2);
    v.push(4. * (x[0] - TRUE_MIN[0]).powi(3));
    v.push(4. * (x[1] - TRUE_MIN[1]).powi(3));
    v
}

fn main() {
    let mut x = [0.; 2];
    CG::new(&mut x, |x| Ok((f(x), g(x))))
        .minimize()
        .expect("Failed to minimize");
    println!("The minimum found is at {:?}", x);
    println!("This should be close to {:?}", TRUE_MIN);
}
