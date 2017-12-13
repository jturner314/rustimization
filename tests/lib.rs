extern crate rustimization;
use rustimization::lbfgsb_minimizer::Lbfgsb;
#[test]
fn test() {
    let f = |x: &[f64]| (x[0] + 4.0).powf(2.0);
    let g = |x: &[f64]| vec![2.0 * (x[0] + 4.0)];
    let mut x = vec![40.0f64];
    {
        let mut fmin = Lbfgsb::new(&mut x, |x| Ok((f(x), g(x))));
        fmin.set_upper_bound(0, 100.0);
        //fmin.set_lower_bound(0,10.0);
        fmin.set_verbosity(-1);
        fmin.max_iteration(100);
        fmin.minimize();
    }
    println!("{:?}", x);
}
