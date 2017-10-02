use cg_minimizer::CG;
use lbfgsb_minimizer::Lbfgsb;

#[allow(non_camel_case_types)]
pub enum MinMethod {
    L_BFGS_B,
    CG,
}

pub struct Funcmin<'a, F>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    x: &'a mut [f64],
    f: F,
    tol: f64,
    verbose: bool,
    method: MinMethod,
    max_iter: u32,
}

impl<'a, F> Funcmin<'a, F>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    // constructor requres three mendatory parameter which is the initial
    // solution, function and the gradient function
    pub fn new(
        xvec: &'a mut [f64],
        func: F,
        m: MinMethod,
    ) -> Self {
        Funcmin {
            x: xvec,
            f: func,
            tol: 1.0e-7,
            max_iter: 10000,
            verbose: false,
            method: m,
        }
    }

    // pub fn minimize(&mut self) {
    //     let ver = if self.verbose { 0 } else { 1 };
    //     {
    //         match self.method {
    //             MinMethod::L_BFGS_B => {
    //                 let mut minf = Lbfgsb::new(&mut self.x, self.f);
    //                 minf.set_verbosity(ver);
    //                 minf.set_tolerance(self.tol);
    //                 minf.max_iteration(self.max_iter);
    //                 minf.minimize();
    //             }
    //             MinMethod::CG => {
    //                 let mut minf = CG::new(&mut self.x, self.f);
    //                 minf.set_verbosity(vec![ver, 0]);
    //                 minf.set_tolerance(self.tol);
    //                 minf.max_iteration(self.max_iter);
    //                 minf.minimize();
    //             }
    //         }
    //     }
    // }

    pub fn set_tolerance(&mut self, t: f64) {
        self.tol = t;
    }

    // set max iteration
    pub fn max_iteration(&mut self, i: u32) {
        self.max_iter = i;
    }

    pub fn set_verbosity(&mut self, b: bool) {
        self.verbose = b;
    }
}
