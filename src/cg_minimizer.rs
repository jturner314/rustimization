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
    iprint: Vec<c_int>,
    method: c_int,
    finish: c_int,
    max_iter: Option<u32>,
}

impl<'a, F> CG<'a, F>
where
    F: Fn(&[c_double]) -> (c_double, Vec<c_double>),
{
    // constructor requres three mendatory parameter which is the initial
    // solution, function and the gradient function
    pub fn new(xvec: &'a mut [c_double], func: F) -> Self {
        let len = xvec.len() as i32;
        // creating CG struct
        CG {
            n: len,
            x: xvec,
            d: vec![0.0f64; len as usize],
            f: func,
            eps: 1.0e-7,
            w: vec![0.0f64; len as usize],
            gold: vec![0.0f64; len as usize],
            iprint: vec![1, 3],
            iflag: 0,
            irest: 0,
            method: 1,
            max_iter: Some(10000),
            finish: 0,
        }
    }

    // this function will start the optimization algorithm
    pub fn minimize(&mut self) -> Result<Success, Error> {
        let (mut fval, mut gval) = (self.f)(self.x);
        let icall = 0;
        loop {
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
                self.method,
                self.finish,
            );
            if self.max_iter.is_some() && icall > self.max_iter.unwrap() {
                return Ok(Success::ReachedMaxIter);
            }
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
                2 => {
                    // termination check
                    self.finish = 1;
                    let tlev = self.eps * (1.0 + fval.abs());
                    for l in gval.iter().take(self.x.len()) {
                        if *l > tlev {
                            self.finish = 0;
                            break;
                        }
                    }
                }
                iflag => panic!("Unknown iflag value {}", iflag),
            }
        }
    }

    // set max iteration
    pub fn max_iteration(&mut self, i: u32) {
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
    pub fn set_verbosity(&mut self, v: Vec<i32>) {
        self.iprint = v;
    }

    // set tolernace
    pub fn set_tolerance(&mut self, t: f64) {
        self.eps = t;
    }
}
