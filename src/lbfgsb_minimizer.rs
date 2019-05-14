use lbfgsb::step;
use libc::{c_char, c_double, c_int};
use std::error;
use std::ffi::CStr;
use std::fmt;
use string::stringfy;

#[derive(Debug)]
pub enum Success {
    Converged,
    ReachedMaxIter {
        /// f(x) and g(x)
        f_g_x: (c_double, Vec<c_double>),
    },
}

#[derive(Debug)]
pub enum Error {
    /// The routine has detected an error in the input parameters.
    BadInput,
    /// The routine has terminated abnormally without being able to satisfy the
    /// termination conditions. `x` contains the best approximation found.
    AbnormalExit {
        /// f(x) and g(x)
        f_g_x: (c_double, Vec<c_double>),
    },
    /// The objective function returned an error.
    ObjectiveFn(Box<error::Error + Send + Sync + 'static>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::BadInput => write!(f, "error in input parameters"),
            Error::AbnormalExit { .. } => {
                write!(f, "abnormal exit without satisfying termination conditions")
            }
            Error::ObjectiveFn(err) => write!(f, "error in objective function: {}", err),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::BadInput | Error::AbnormalExit { .. } => None,
            Error::ObjectiveFn(err) => Some(&**err),
        }
    }
}

const CSAVE_LEN: usize = 60;
const LSAVE_LEN: usize = 4;
const ISAVE_LEN: usize = 44;
const DSAVE_LEN: usize = 29;

pub struct Lbfgsb<'a, F>
where
    F: FnMut(
        &[c_double],
    ) -> Result<(c_double, Vec<c_double>), Box<error::Error + Send + Sync + 'static>>,
{
    /// Number of variables.
    n: c_int,
    /// Number of corrections used in the limited memory matrix. Values of m <
    /// 3 are not recommended, and large values of m can result in excessive
    /// computing time. The range 3 <= m <= 20 is recommended.
    m: c_int,
    /// Initial guess and then, on successful exit, best solution found.
    x: &'a mut [c_double],
    /// Lower bounds on variables.
    l: Vec<c_double>,
    /// Upper bounds on variables.
    u: Vec<c_double>,
    /// Types of bounds on variables.
    nbd: Vec<c_int>,
    /// Objective and gradient function.
    f: F,
    /// Tolerance in function value termination test. Iteration will stop when
    /// `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch`
    /// where `epsmch` is the machine precision.
    ///
    /// Typical values for factr on a computer with 15 digits of accuracy in
    /// double precision are `factr` =
    ///
    /// * 1.e12 for low accuracy;
    /// * 1.e7  for moderate accuracy;
    /// * 1.e1  for extremely high accuracy.
    ///
    /// The user can suppress this termination test by setting `factr` to `0.`.
    factr: c_double,
    /// Tolerance in projected gradient termination test. Iteration
    /// will stop when `max{|proj g_i | i = 1, ..., n} <= pgtol`
    /// where `proj g_i` is the `i`th component of the projected gradient.
    /// The user can suppress this termination test by setting `pgtol` to `0.`.
    pgtol: c_double,
    /// Working array for the routine.
    iwa: Vec<c_int>,
    task: Vec<c_char>,
    /// Controls the frequency and type of output generated:
    ///
    /// * `iprint<0` no output is generated;
    /// * `iprint=0` print only one line at the last iteration;
    /// * `0<iprint<99` print also `f` and `|proj g|` every `iprint` iterations;
    /// * `iprint=99` print details of every iteration except n-vectors;
    /// * `iprint=100` print also the changes of active set and final `x`;
    /// `iprint>100` print details of every iteration including `x` and `g`;
    /// When `iprint > 0`, the file `iterate.dat` will be created to summarize the iteration.
    iprint: c_int,
    /// Working array for the routine.
    csave: [c_char; CSAVE_LEN],
    /// Working array for the routine.
    // TODO: is c_int correct here?
    lsave: [c_int; LSAVE_LEN],
    /// Working array for the routine.
    isave: [c_int; ISAVE_LEN],
    /// Working array for the routine.
    dsave: [c_double; DSAVE_LEN],
    max_iter: Option<u32>,
}

impl<'a, F> Lbfgsb<'a, F>
where
    F: FnMut(
        &[c_double],
    ) -> Result<(c_double, Vec<c_double>), Box<error::Error + Send + Sync + 'static>>,
{
    // constructor requres three mendatory parameter which is the initial
    // solution, function and the gradient function
    pub fn new(xvec: &'a mut [c_double], func: F) -> Self {
        let n: usize = xvec.len();
        Lbfgsb {
            n: n as i32,
            m: 10, // `scipy.optimize.fmin_l_bfgs_b` default,
            x: xvec,
            l: vec![0.; n],
            u: vec![0.; n],
            nbd: vec![0; n],
            f: func,
            factr: 1e7,  // `scipy.optimize.fmin_l_bfgs_b` default
            pgtol: 1e-5, // `scipy.optimize.fmin_l_bfgs_b` default
            iwa: vec![0; 3 * n],
            task: vec![0; 60],
            iprint: -1,
            csave: [0; CSAVE_LEN],
            lsave: [0; LSAVE_LEN],
            isave: [0; ISAVE_LEN],
            dsave: [0.; DSAVE_LEN],
            max_iter: Some(15000), // `scipy.optimize.fmin_l_bfgs_b` default
        }
    }

    // this function will start the optimization algorithm
    pub fn minimize(&mut self) -> Result<Success, Error> {
        // Create the working array for the routine here because we need to know the latest `m`
        let n = self.x.len();
        let m = self.m as usize;
        let mut wa: Vec<c_double> = vec![0.; 2 * m * n + 5 * n + 11 * m * m + 8 * m];

        let mut fval = 0.0f64;
        let mut gval = vec![0.0f64; n];
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
                &mut wa,
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
                let vals = (self.f)(self.x).map_err(|err| Error::ObjectiveFn(err))?;
                fval = vals.0;
                gval = vals.1;
            }
            if tsk.starts_with("NEW_X") {
                if self.max_iter.is_some() && self.iteration() >= self.max_iter.unwrap() as i32 {
                    return Ok(Success::ReachedMaxIter {
                        f_g_x: (fval, gval),
                    });
                }
                // TODO: add callback to check convergence here
            }
            if tsk.starts_with("CONV") {
                return Ok(Success::Converged);
            }
            if tsk.starts_with("ERROR") {
                return Err(Error::BadInput);
            }
            if tsk.starts_with("ABNO") {
                return Err(Error::AbnormalExit {
                    f_g_x: (fval, gval),
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
