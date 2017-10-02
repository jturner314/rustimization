use libc::{c_char, c_double, c_int};
use std::ffi::CStr;
use lbfgsb::step;
use string::stringfy;

pub enum Success {
    /// The projected gradient is sufficiently small.
    ProjectedGradient,
    Converged,
    ReachedMaxIter,
}

pub enum Error {
    /// The routine has detected an error in the input parameters.
    BadInput,
    /// The routine has terminated abnormally without being able to satisfy the
    /// termination conditions. `x` contains the best approximation found.
    AbnormalExit {
        /// f(x)
        f_x: c_double,
        /// g(x)
        g_x: Vec<c_double>,
    },
}

pub struct Lbfgsb<'a, F>
where
    F: Fn(&[c_double]) -> (c_double, Vec<c_double>),
{
    n: c_int,
    m: c_int,
    x: &'a mut [c_double],
    l: Vec<c_double>,
    u: Vec<c_double>,
    nbd: Vec<c_int>,
    f: F,
    factr: c_double,
    pgtol: c_double,
    wa: Vec<c_double>,
    iwa: Vec<c_int>,
    task: Vec<c_char>,
    iprint: c_int,
    csave: Vec<c_char>,
    lsave: Vec<c_int>,
    isave: Vec<c_int>,
    dsave: Vec<c_double>,
    max_iter: Option<u32>,
}

impl<'a, F> Lbfgsb<'a, F>
where
    F: Fn(&[c_double]) -> (c_double, Vec<c_double>),
{
    // constructor requres three mendatory parameter which is the initial
    // solution, function and the gradient function
    pub fn new(xvec: &'a mut [c_double], func: F) -> Self {
        let len = xvec.len() as i32;
        Lbfgsb {
            n: len,
            m: 5,
            x: xvec,
            l: vec![0.0f64; len as usize],
            u: vec![0.0f64; len as usize],
            nbd: vec![0; len as usize],
            f: func,
            factr: 0.0e0,
            pgtol: 0.0e0,
            wa: vec![0.0f64; (2 * 5 * len + 11 * 5 * 5 + 5 * len + 8 * 5) as usize],
            iwa: vec![0; 3 * len as usize],
            task: vec![0; 60],
            iprint: -1,
            csave: vec![0; 60],
            lsave: vec![0, 0, 0, 0],
            isave: vec![0; 44],
            dsave: vec![0.0f64; 29],
            max_iter: None,
        }
    }

    // this function will start the optimization algorithm
    pub fn minimize(&mut self) -> Result<Success, Error> {
        let mut fval = 0.0f64;
        let mut gval = vec![0.0f64; self.x.len()];
        // converting fortran string "START"
        stringfy(&mut self.task);
        loop {
            step(
                self.n,
                self.m,
                self.x,
                &self.l,
                &self.u,
                &self.nbd,
                fval,
                &gval,
                self.factr,
                self.pgtol,
                &mut self.wa,
                &mut self.iwa,
                &mut self.task,
                self.iprint,
                &mut self.csave,
                &mut self.lsave,
                &mut self.isave,
                &mut self.dsave,
            );
            // converting to rust string
            let tsk = unsafe { CStr::from_ptr(self.task.as_ptr()).to_string_lossy() };
            if tsk.starts_with("FG") {
                let vals = (self.f)(self.x);
                fval = vals.0;
                gval = vals.1;
            }
            if tsk.starts_with("NEW_X") && self.max_iter.is_none()
                && self.dsave[11] <= 1.0e-10 * (1.0e0 + fval.abs())
            {
                return Ok(Success::ProjectedGradient);
            }
            if self.max_iter.is_some() && self.iteration() >= self.max_iter.unwrap() as i32 {
                return Ok(Success::ReachedMaxIter);
            }
            if tsk.starts_with("CONV") {
                return Ok(Success::Converged);
            }
            if tsk.starts_with("ERROR") {
                return Err(Error::BadInput);
            }
            if tsk.starts_with("ABNO") {
                return Err(Error::AbnormalExit {
                    f_x: fval,
                    g_x: gval,
                });
            }
        }
    }

    fn iteration(&self) -> c_int {
        self.isave[29]
    }

    // this function is used to set lower bounds to a variable
    pub fn set_lower_bound(&mut self, index: usize, value: f64) {
        if self.nbd[index] == 1 || self.nbd[index] == 2 {
            println!("variable already has Lower Bound");
        } else {
            let temp = self.nbd[index] - 1;
            self.nbd[index] = if temp < 0 { -temp } else { temp };
            self.l[index] = value;
        }
    }

    // this function is used to set upper bounds to a variable
    pub fn set_upper_bound(&mut self, index: usize, value: f64) {
        if self.nbd[index] == 3 || self.nbd[index] == 2 {
            println!("variable already has Lower Bound");
        } else {
            self.nbd[index] = 3 - self.nbd[index];
            self.u[index] = value;
        }
    }

    // set the verbosity level
    pub fn set_verbosity(&mut self, l: i32) {
        self.iprint = l;
    }

    // set termination tolerance
    // 1.0e12 for low accuracy
    // 1.0e7  for moderate accuracy
    // 1.0e1  for extremely high accuracy
    pub fn set_termination_tolerance(&mut self, t: f64) {
        self.factr = t;
    }

    // set tolerance of projection gradient
    pub fn set_tolerance(&mut self, t: f64) {
        self.pgtol = t;
    }

    // set max iteration
    pub fn max_iteration(&mut self, i: u32) {
        self.max_iter = Some(i);
    }

    // set maximum number of variable metric corrections
    // The range  3 <= m <= 20 is recommended
    pub fn set_matric_correction(&mut self, m: i32) {
        self.m = m;
    }
}
