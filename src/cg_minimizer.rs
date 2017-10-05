use libc::{c_double, c_int};
use cg::step;

#[derive(Debug)]
pub enum Success {
    Converged,
    ReachedMaxIter,
}

#[derive(Debug)]
pub enum Error {
    /// Improper input parameters.
    BadInput,
    /// Descent was not obtained.
    NoDescent,
    /// Line search failure.
    LineSearch,
}

#[derive(Clone, Copy, Debug)]
enum Method {
    /// Fletcher–Reeves
    FletcherReeves = 1,
    /// Polak–Ribière
    PolakRibiere = 2,
    /// Positive Polak–Ribière (β = max(β,0))
    PositivePolakRibiere = 3,
}

pub struct CG<'a, F>
where
    F: Fn(&[c_double]) -> (c_double, Vec<c_double>),
{
    n: c_int,
    x: &'a mut [c_double],
    d: Vec<c_double>,
    gold: Vec<c_double>,
    f: F,
    eps: c_double,
    w: Vec<c_double>,
    iflag: c_int,
    irest: c_int,
    iprint: [c_int; 2],
    method: Method,
    gtol: c_double,
    finish: c_int,
    max_iter: Option<usize>,
}

impl<'a, F> CG<'a, F>
where
    F: Fn(&[c_double]) -> (c_double, Vec<c_double>),
{
    // constructor requres three mendatory parameter which is the initial
    // solution, function and the gradient function
    pub fn new(xvec: &'a mut [c_double], func: F) -> Self {
        let len = xvec.len();
        // creating CG struct
        CG {
            n: len as i32,
            x: xvec,
            d: vec![0.; len],
            f: func,
            eps: 1e-7,
            w: vec![0.; len],
            gold: vec![0.; len],
            iprint: [-1, 0],
            iflag: 0,
            irest: 0,
            method: Method::FletcherReeves,
            gtol: 1e-5,                // `scipy.optimize.fmin_cg` default
            max_iter: Some(200 * len), // `scipy.optimize.fmin_cg` default
            finish: 0,
        }
    }

    // this function will start the optimization algorithm
    pub fn minimize(&mut self) -> Result<Success, Error> {
        let (mut fval, mut gval) = (self.f)(self.x);
        let mut iter_num = 0;
        loop {
            if self.max_iter.is_some() && iter_num >= self.max_iter.unwrap() {
                return Ok(Success::ReachedMaxIter);
            }
            step(
                self.n,
                &mut self.x,
                fval,
                &gval,
                &mut self.d,
                &mut self.gold,
                &self.iprint,
                &mut self.w,
                self.eps,
                &mut self.iflag,
                self.irest,
                self.method as i32,
                self.finish,
            );
            iter_num += 1;
            match self.iflag {
                -3 => {
                    return Err(Error::BadInput);
                }
                -2 => {
                    return Err(Error::NoDescent);
                }
                -1 => {
                    return Err(Error::LineSearch);
                }
                0 => {
                    return Ok(Success::Converged);
                }
                1 => {
                    // geting the function and gradient value
                    let vals = (self.f)(self.x);
                    fval = vals.0;
                    gval = vals.1;
                }
                2 => if gval.iter().all(|g| g.abs() <= self.gtol) {
                    self.finish = 1;
                },
                iflag => panic!("Unknown iflag value {}", iflag),
            }
        }
    }

    // set max iteration
    pub fn max_iteration(&mut self, i: usize) {
        self.max_iter = Some(i);
    }

    // set restart
    pub fn set_restart(&mut self, b: bool) {
        self.irest = b as i32;
    }

    // set verbosity
    // vec[0] < 0 : NO OUTPUT IS GENERATED
    // vec[0] = 0 : OUTPUT ONLY AT FIRST AND LAST ITERATION
    // vec[0] > 0 : OUTPUT EVERY IPRINT(1) ITERATIONS
    // vec[1]     : SPECIFIES THE TYPE OF OUTPUT GENERATED;
    // vec[1] = 0 : NO ADDITIONAL INFORMATION PRINTED
    // vec[1] = 1 : INITIAL X AND GRADIENT VECTORS PRINTED
    // vec[1] = 2 : X VECTOR PRINTED EVERY ITERATION
    // vec[1] = 3 : X VECTOR AND GRADIENT VECTOR PRINTED
    pub fn set_verbosity(&mut self, v: [i32; 2]) {
        self.iprint = v;
    }

    // set tolernace
    pub fn set_tolerance(&mut self, t: f64) {
        self.eps = t;
    }
}
