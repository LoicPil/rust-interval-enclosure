
use inari::{interval, Interval};
use matplotlib::pyplot::subplots;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul, Sub};

// ============================================================
// Configuration
// ============================================================

const ORDER: usize = 10;

const T0: f64 = 0.0;
const TF: f64 = 3.0;

const H: f64 = 0.01;

const N_STEPS: usize = ((TF - T0) / H) as usize;

// Lorenz parameters
const SIGMA: f64 = 10.0;
const RHO: f64 = 28.0;
const BETA: f64 = 8.0 / 3.0;

// Initial interval
const Y1_LO: f64 = -8.001;
const Y1_HI: f64 = -7.998;

const Y2_LO: f64 = 7.998;
const Y2_HI: f64 = 8.001;

const Y3_LO: f64 = 26.998;
const Y3_HI: f64 = 27.001;

// ============================================================
// Polynomial
// ============================================================

#[derive(Clone, Debug)]
pub struct Polynomial {
    coeffs: HashMap<Vec<usize>, f64>,
    degree: usize,
    dimension: usize,
}

impl Polynomial {
    pub fn new(dimension: usize, degree: usize) -> Self {
        Self {
            coeffs: HashMap::new(),
            degree,
            dimension,
        }
    }

    pub fn get(&self, exponents: &[usize]) -> f64 {
        *self.coeffs.get(exponents).unwrap_or(&0.0)
    }

    pub fn set(&mut self, exponents: &[usize], value: f64) {
        assert_eq!(exponents.len(), self.dimension);

        if value.abs() < 1e-14 {
            self.coeffs.remove(exponents);
        } else {
            self.coeffs.insert(exponents.to_vec(), value);
        }
    }

    pub fn constant(
        dimension: usize,
        degree: usize,
        value: f64,
    ) -> Self {
        let mut p = Self::new(dimension, degree);

        p.set(&vec![0; dimension], value);

        p
    }

    pub fn variable(
        dimension: usize,
        degree: usize,
        variable: usize,
        coefficient: f64,
    ) -> Self {
        assert!(variable < dimension);

        let mut p = Self::new(dimension, degree);

        let mut exponents = vec![0; dimension];

        exponents[variable] = 1;

        p.set(&exponents, coefficient);

        p
    }

    pub fn total_degree(exponents: &[usize]) -> usize {
        exponents.iter().sum()
    }

    // --------------------------------------------------------
    // Interval evaluation
    // --------------------------------------------------------

    pub fn evaluate(
        &self,
        domain: &[Interval],
    ) -> Interval {
        assert_eq!(domain.len(), self.dimension);

        let mut result =
            interval!(0.0, 0.0).unwrap();

        for (exponents, coefficient) in &self.coeffs {
            let mut term =
                interval!(*coefficient, *coefficient)
                    .unwrap();

            for (i, exponent) in
                exponents.iter().enumerate()
            {
                if *exponent > 0 {
                    term *= domain[i]
                        .powi(*exponent as i32);
                }
            }

            result += term;
        }

        result
    }

    // --------------------------------------------------------
    // Split into low and high order terms
    // --------------------------------------------------------

    pub fn split(
        &self,
        order: usize,
    ) -> (Polynomial, Polynomial) {
        let mut low =
            Polynomial::new(self.dimension, order);

        let mut high =
            Polynomial::new(self.dimension, self.degree);

        for (exponents, coefficient) in
            &self.coeffs
        {
            if Self::total_degree(exponents) <= order {
                low.set(exponents, *coefficient);
            } else {
                high.set(exponents, *coefficient);
            }
        }

        (low, high)
    }

    // --------------------------------------------------------
    // Point evaluation
    // --------------------------------------------------------

    pub fn sample(&self, point: &[f64]) -> f64 {
        assert_eq!(point.len(), self.dimension);

        let mut result = 0.0;

        for (exponents, coefficient) in
            &self.coeffs
        {
            let mut term = *coefficient;

            for (x, exponent) in
                point.iter().zip(exponents.iter())
            {
                term *= x.powi(*exponent as i32);
            }

            result += term;
        }

        result
    }

    // --------------------------------------------------------
    // Terms
    // --------------------------------------------------------

    pub fn terms(
        &self,
    ) -> Vec<(&Vec<usize>, &f64)> {
        let mut terms: Vec<_> =
            self.coeffs.iter().collect();

        terms.sort_by_key(|(e, _)| {
            Self::total_degree(e)
        });

        terms
    }
}

// ============================================================
// Polynomial arithmetic
// ============================================================

impl Add for Polynomial {
    type Output = Polynomial;

    fn add(
        mut self,
        other: Polynomial,
    ) -> Polynomial {
        assert_eq!(
            self.dimension,
            other.dimension
        );

        assert_eq!(self.degree, other.degree);

        for (exponents, coefficient) in
            other.coeffs
        {
            let value =
                self.get(&exponents) + coefficient;

            self.set(&exponents, value);
        }

        self
    }
}

impl Sub for Polynomial {
    type Output = Polynomial;

    fn sub(
        mut self,
        other: Polynomial,
    ) -> Polynomial {
        assert_eq!(
            self.dimension,
            other.dimension
        );

        assert_eq!(self.degree, other.degree);

        for (exponents, coefficient) in
            other.coeffs
        {
            let value =
                self.get(&exponents) - coefficient;

            self.set(&exponents, value);
        }

        self
    }
}

impl Mul for Polynomial {
    type Output = Polynomial;

    fn mul(
        self,
        other: Polynomial,
    ) -> Polynomial {
        assert_eq!(
            self.dimension,
            other.dimension
        );

        let degree =
            self.degree + other.degree;

        let mut result =
            Polynomial::new(
                self.dimension,
                degree,
            );

        for (e1, c1) in &self.coeffs {
            for (e2, c2) in &other.coeffs {
                let exponents: Vec<usize> =
                    e1.iter()
                        .zip(e2.iter())
                        .map(|(a, b)| a + b)
                        .collect();

                let value =
                    result.get(&exponents)
                        + c1 * c2;

                result.set(&exponents, value);
            }
        }

        result
    }
}

// ============================================================
// Taylor Model
// ============================================================

#[derive(Clone, Debug)]
pub struct TaylorModel {
    pub polynomial: Polynomial,
    pub remainder: Interval,
    pub domain: Vec<Interval>,
    pub order: usize,
}

impl TaylorModel {
    pub fn constant(
        value: f64,
        dimension: usize,
        order: usize,
        domain: Vec<Interval>,
    ) -> Self {
        Self {
            polynomial:
                Polynomial::constant(
                    dimension,
                    order,
                    value,
                ),

            remainder:
                interval!(0.0, 0.0).unwrap(),

            domain,
            order,
        }
    }

    pub fn variable(
        variable: usize,
        coefficient: f64,
        dimension: usize,
        order: usize,
        domain: Vec<Interval>,
    ) -> Self {
        Self {
            polynomial:
                Polynomial::variable(
                    dimension,
                    order,
                    variable,
                    coefficient,
                ),

            remainder:
                interval!(0.0, 0.0).unwrap(),

            domain,
            order,
        }
    }

    pub fn from_interval(
        value: Interval,
        dimension: usize,
        order: usize,
        domain: Vec<Interval>,
    ) -> Self {
        Self {
            polynomial:
                Polynomial::constant(
                    dimension,
                    order,
                    0.0,
                ),

            remainder: value,

            domain,
            order,
        }
    }

    pub fn range(&self) -> Interval {
        self.polynomial
            .evaluate(&self.domain)
            + self.remainder
    }

    pub fn polynomial_range(&self) -> Interval {
        self.polynomial
            .evaluate(&self.domain)
    }

    pub fn powi(
        &self,
        exponent: usize,
    ) -> TaylorModel {
        if exponent == 0 {
            return TaylorModel::constant(
                1.0,
                self.polynomial.dimension,
                self.order,
                self.domain.clone(),
            );
        }

        let mut result = self.clone();

        for _ in 1..exponent {
            result =
                result * self.clone();
        }

        result
    }

    pub fn sample(
        &self,
        point: &[f64],
    ) -> f64 {
        self.polynomial.sample(point)
    }
}

// ============================================================
// Taylor Model arithmetic
// ============================================================

impl Add for TaylorModel {
    type Output = TaylorModel;

    fn add(
        self,
        other: TaylorModel,
    ) -> TaylorModel {
        assert_eq!(
            self.domain,
            other.domain
        );

        assert_eq!(
            self.order,
            other.order
        );

        TaylorModel {
            polynomial:
                self.polynomial
                    + other.polynomial,

            remainder:
                self.remainder
                    + other.remainder,

            domain: self.domain,

            order: self.order,
        }
    }
}

impl Sub for TaylorModel {
    type Output = TaylorModel;

    fn sub(
        self,
        other: TaylorModel,
    ) -> TaylorModel {
        assert_eq!(
            self.domain,
            other.domain
        );

        assert_eq!(
            self.order,
            other.order
        );

        TaylorModel {
            polynomial:
                self.polynomial
                    - other.polynomial,

            remainder:
                self.remainder
                    - other.remainder,

            domain: self.domain,

            order: self.order,
        }
    }
}

impl Mul for TaylorModel {
    type Output = TaylorModel;

    fn mul(
        self,
        other: TaylorModel,
    ) -> TaylorModel {
        assert_eq!(
            self.domain,
            other.domain
        );

        assert_eq!(
            self.order,
            other.order
        );

        let order = self.order;

        // Full polynomial product.
        let product =
            self.polynomial.clone()
                * other.polynomial.clone();

        // Keep terms up to ORDER.
        let (low, high) =
            product.split(order);

        // Range of discarded terms.
        let high_range =
            high.evaluate(&self.domain);

        let p_range =
            self.polynomial
                .evaluate(&self.domain);

        let q_range =
            other.polynomial
                .evaluate(&self.domain);

        // (P + E)(Q + F)
        //
        // = PQ + PF + QE + EF
        //
        // PQ = low + high
        //
        // Therefore everything not represented
        // by low goes into the remainder.

        let error =
            high_range
                + p_range * other.remainder
                + q_range * self.remainder
                + self.remainder
                    * other.remainder;

        TaylorModel {
            polynomial: low,

            remainder: error,

            domain: self.domain,

            order,
        }
    }
}

// ============================================================
// Display
// ============================================================

impl fmt::Display for TaylorModel {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let names =
            ["x", "y", "z", "t"];

        let terms =
            self.polynomial.terms();

        if terms.is_empty() {
            write!(f, "0")?;
        } else {
            let mut first = true;

            for (exponents, coefficient)
                in terms
            {
                if !first {
                    if *coefficient >= 0.0 {
                        write!(
                            f,
                            " + {:.6}",
                            coefficient
                        )?;
                    } else {
                        write!(
                            f,
                            " - {:.6}",
                            coefficient.abs()
                        )?;
                    }
                } else {
                    write!(
                        f,
                        "{:.6}",
                        coefficient
                    )?;

                    first = false;
                }

                for (i, exponent)
                    in exponents.iter().enumerate()
                {
                    if *exponent == 0 {
                        continue;
                    }

                    write!(
                        f,
                        "*{}",
                        names[i]
                    )?;

                    if *exponent > 1 {
                        write!(
                            f,
                            "^{}",
                            exponent
                        )?;
                    }
                }
            }
        }

        write!(
            f,
            " + E, E = {}",
            self.remainder
        )
    }
}

// ============================================================
// Lorenz system
//
// y1' = sigma (y2 - y1)
//
// y2' = (rho - y3)y1 - y2
//
// y3' = y1 y2 - beta y3
// ============================================================

fn lorenz(
    y1: TaylorModel,
    y2: TaylorModel,
    y3: TaylorModel,
) -> (
    TaylorModel,
    TaylorModel,
    TaylorModel,
) {
    let domain =
        y1.domain.clone();

    let sigma =
        TaylorModel::constant(
            SIGMA,
            3,
            ORDER,
            domain.clone(),
        );

    let rho =
        TaylorModel::constant(
            RHO,
            3,
            ORDER,
            domain.clone(),
        );

    let beta =
        TaylorModel::constant(
            BETA,
            3,
            ORDER,
            domain.clone(),
        );

    // y1' = sigma * (y2 - y1)
    let f1 =
        sigma * (y2.clone() - y1.clone());

    // y2' = (rho - y3)y1 - y2
    let f2 =
        (rho - y3.clone())
            * y1.clone()
            - y2.clone();

    // y3' = y1*y2 - beta*y3
    let f3 =
        y1.clone() * y2.clone()
            - beta * y3.clone();

    (f1, f2, f3)
}

// ============================================================
// One Euler step
//
// y_{n+1} = y_n + h f(y_n)
//
// This is deliberately simple.
// No QR preconditioning.
// No shrink wrapping.
// ============================================================

fn euler_step(
    y1: TaylorModel,
    y2: TaylorModel,
    y3: TaylorModel,
    h: f64,
) -> (
    TaylorModel,
    TaylorModel,
    TaylorModel,
) {
    let domain =
        y1.domain.clone();

    let h_tm =
        TaylorModel::constant(
            h,
            3,
            ORDER,
            domain,
        );

    let (f1, f2, f3) =
        lorenz(
            y1.clone(),
            y2.clone(),
            y3.clone(),
        );

    let new_y1 =
        y1 + h_tm.clone() * f1;

    let new_y2 =
        y2 + h_tm.clone() * f2;

    let new_y3 =
        y3 + h_tm * f3;

    (new_y1, new_y2, new_y3)
}

// ============================================================
// Interval Euler propagation
// ============================================================
//
// This is the basic Taylor-model ODE experiment.
//
// Important:
// the initial uncertainty is represented by the
// Taylor-model variables:
//
// y1 = center + radius*x
// y2 = center + radius*y
// y3 = center + radius*z
//
// with (x,y,z) in [-1,1]^3.
//
// ============================================================

fn solve_lorenz() -> Vec<(
    f64,
    Interval,
    Interval,
    Interval,
)> {
    let domain = vec![
        interval!(-1.0, 1.0).unwrap(),
        interval!(-1.0, 1.0).unwrap(),
        interval!(-1.0, 1.0).unwrap(),
    ];

    // Initial intervals
    let y1_mid =
        0.5 * (Y1_LO + Y1_HI);

    let y1_rad =
        0.5 * (Y1_HI - Y1_LO);

    let y2_mid =
        0.5 * (Y2_LO + Y2_HI);

    let y2_rad =
        0.5 * (Y2_HI - Y2_LO);

    let y3_mid =
        0.5 * (Y3_LO + Y3_HI);

    let y3_rad =
        0.5 * (Y3_HI - Y3_LO);

    // Initial Taylor models:
    //
    // y1 = midpoint + radius*x
    // y2 = midpoint + radius*y
    // y3 = midpoint + radius*z

    let mut y1 =
        TaylorModel::constant(
            y1_mid,
            3,
            ORDER,
            domain.clone(),
        )
        + TaylorModel::variable(
            0,
            y1_rad,
            3,
            ORDER,
            domain.clone(),
        );

    let mut y2 =
        TaylorModel::constant(
            y2_mid,
            3,
            ORDER,
            domain.clone(),
        )
        + TaylorModel::variable(
            1,
            y2_rad,
            3,
            ORDER,
            domain.clone(),
        );

    let mut y3 =
        TaylorModel::constant(
            y3_mid,
            3,
            ORDER,
            domain.clone(),
        )
        + TaylorModel::variable(
            2,
            y3_rad,
            3,
            ORDER,
            domain,
        );

    let mut result =
        Vec::with_capacity(N_STEPS + 1);

    result.push((
        T0,
        y1.range(),
        y2.range(),
        y3.range(),
    ));

    for step in 0..N_STEPS {
        (y1, y2, y3) =
            euler_step(
                y1,
                y2,
                y3,
                H,
            );

        let t =
            T0 + (step + 1) as f64 * H;

        let r1 = y1.range();
        let r2 = y2.range();
        let r3 = y3.range();

        result.push((
            t,
            r1,
            r2,
            r3,
        ));

        if step % 50 == 0 {
            println!(
                "t = {:.3} | \
                 y1 = [{:.6e}, {:.6e}] | \
                 y2 = [{:.6e}, {:.6e}] | \
                 y3 = [{:.6e}, {:.6e}]",
                t,
                r1.inf(),
                r1.sup(),
                r2.inf(),
                r2.sup(),
                r3.inf(),
                r3.sup(),
            );
        }
    }

    result
}

// ============================================================
// Plot
// ============================================================

fn plot_solution(
    result: &[(
        f64,
        Interval,
        Interval,
        Interval,
    )],
) -> Result<
    (),
    Box<dyn std::error::Error>,
> {
    let ts: Vec<f64> =
        result
            .iter()
            .map(|(t, _, _, _)| *t)
            .collect();

    let y1_lo: Vec<f64> =
        result
            .iter()
            .map(|(_, y1, _, _)| y1.inf())
            .collect();

    let y1_hi: Vec<f64> =
        result
            .iter()
            .map(|(_, y1, _, _)| y1.sup())
            .collect();

    let y2_lo: Vec<f64> =
        result
            .iter()
            .map(|(_, _, y2, _)| y2.inf())
            .collect();

    let y2_hi: Vec<f64> =
        result
            .iter()
            .map(|(_, _, y2, _)| y2.sup())
            .collect();

    let y3_lo: Vec<f64> =
        result
            .iter()
            .map(|(_, _, _, y3)| y3.inf())
            .collect();

    let y3_hi: Vec<f64> =
        result
            .iter()
            .map(|(_, _, _, y3)| y3.sup())
            .collect();

    let (fig, [[mut ax1, mut ax2], [mut ax3, mut ax4]]) =
        subplots()?;

    // --------------------------------------------------------
    // y1
    // --------------------------------------------------------

    ax1.xy(&ts, &y1_lo)
        .fmt("-")
        .color([0.0, 0.6, 0.0])
        .label("lower")
        .plot();

    ax1.xy(&ts, &y1_hi)
        .fmt("-")
        .color([0.0, 0.6, 0.0])
        .label("upper")
        .plot();

    ax1.set_title("Lorenz: y1");
    ax1.set_xlabel("t");
    ax1.set_ylabel("y1");
    ax1.grid();

    // --------------------------------------------------------
    // y2
    // --------------------------------------------------------

    ax2.xy(&ts, &y2_lo)
        .fmt("-")
        .color([0.0, 0.0, 1.0])
        .label("lower")
        .plot();

    ax2.xy(&ts, &y2_hi)
        .fmt("-")
        .color([0.0, 0.0, 1.0])
        .label("upper")
        .plot();

    ax2.set_title("Lorenz: y2");
    ax2.set_xlabel("t");
    ax2.set_ylabel("y2");
    ax2.grid();

    // --------------------------------------------------------
    // y3
    // --------------------------------------------------------

    ax3.xy(&ts, &y3_lo)
        .fmt("-")
        .color([0.7, 0.0, 0.7])
        .label("lower")
        .plot();

    ax3.xy(&ts, &y3_hi)
        .fmt("-")
        .color([0.7, 0.0, 0.7])
        .label("upper")
        .plot();

    ax3.set_title("Lorenz: y3");
    ax3.set_xlabel("t");
    ax3.set_ylabel("y3");
    ax3.grid();

    // --------------------------------------------------------
    // All components
    // --------------------------------------------------------

    ax4.xy(&ts, &y1_lo)
        .fmt("-")
        .label("y1")
        .plot();

    ax4.xy(&ts, &y2_lo)
        .fmt("-")
        .label("y2")
        .plot();

    ax4.xy(&ts, &y3_lo)
        .fmt("-")
        .label("y3")
        .plot();

    ax4.set_title("Lorenz system");
    ax4.set_xlabel("t");
    ax4.set_ylabel("y");
    ax4.grid();

    ax4.legend(
        std::iter::empty(),
    );

    // Your Rust matplotlib binding supports
    // save().to_file(), so save directly as PDF.

    fig.save()
        .to_file("lorenz_taylor_model.pdf")?;

    Ok(())
}

// ============================================================
// Main
// ============================================================

fn main() -> Result<
    (),
    Box<dyn std::error::Error>,
> {
    println!();
    println!(
        "=========================================="
    );

    println!(
        "Taylor Model - Lorenz System"
    );

    println!(
        "=========================================="
    );

    println!(
        "Taylor order : {}",
        ORDER
    );

    println!(
        "Time interval: [{}, {}]",
        T0, TF
    );

    println!(
        "Step size    : {}",
        H
    );

    println!(
        "Steps        : {}",
        N_STEPS
    );

    println!();

    println!(
        "Initial y1 = [{}, {}]",
        Y1_LO, Y1_HI
    );

    println!(
        "Initial y2 = [{}, {}]",
        Y2_LO, Y2_HI
    );

    println!(
        "Initial y3 = [{}, {}]",
        Y3_LO, Y3_HI
    );

    println!();

    println!(
        "No QR preconditioning."
    );

    println!(
        "No shrink wrapping."
    );

    println!(
        "Using Taylor-model arithmetic."
    );

    println!();

    let result =
        solve_lorenz();

    println!();
    println!(
        "=========================================="
    );

    println!(
        "Final enclosure at t = {}",
        TF
    );

    let (_, y1, y2, y3) =
        result.last().unwrap();

    println!(
        "y1 = [{:.15e}, {:.15e}]",
        y1.inf(),
        y1.sup()
    );

    println!(
        "y2 = [{:.15e}, {:.15e}]",
        y2.inf(),
        y2.sup()
    );

    println!(
        "y3 = [{:.15e}, {:.15e}]",
        y3.inf(),
        y3.sup()
    );

    println!(
        "width(y1) = {:.6e}",
        y1.wid()
    );

    println!(
        "width(y2) = {:.6e}",
        y2.wid()
    );

    println!(
        "width(y3) = {:.6e}",
        y3.wid()
    );

    println!(
        "=========================================="
    );

    plot_solution(&result)?;

    println!();
    println!(
        "PDF written to: \
         lorenz_taylor_model.pdf"
    );

    Ok(())
}
