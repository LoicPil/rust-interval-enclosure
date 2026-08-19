use inari::{Interval, IntervalError, const_interval, interval};
use nalgebra::{DMatrix, SymmetricEigen};

const ONE: Interval = const_interval!(1.0, 1.0);
const ZERO: Interval = const_interval!(0.0, 0.0);
const TWO: Interval = const_interval!(2.0, 2.0);
const FOUR: Interval = const_interval!(4.0, 4.0);
const SIX: Interval = const_interval!(6.0, 6.0);
const TWELVE: Interval = const_interval!(12.0, 12.0);
const TWENTY_FOUR: Interval = const_interval!(24.0, 24.0);
const TWO_THOUSAND_EIGHT_HUNDRED_EIGHTY: Interval = const_interval!(2880.0, 2880.0);

/// Midpoint rule with certified second-derivative error term.
///
/// Per subinterval [x_i, x_i+h]:
///
/// h f((x_i+x_{i+1})/2) + h^3/24 * f''([x_i,x_{i+1}])
pub fn midpoint_certified<F, FPP>(
    f: F,
    f_pp: FPP,
    a: f64,
    b: f64,
    n: u32,
) -> Result<Interval, IntervalError>
where
    F: Fn(Interval) -> Interval,
    FPP: Fn(Interval) -> Interval,
{
    let ia = interval!(a, a)?;
    let ib = interval!(b, b)?;
    let i_n = interval!(n as f64, n as f64)?;

    let h = (ib - ia) / i_n;
    let mut acc = ZERO;

    for i in 0..n {
        let ii = interval!(i as f64, i as f64)?;

        let xi = ia + ii * h;
        let xi1 = ia + (ii + ONE) * h;
        let mid = (xi + xi1) / TWO;

        let point_value = f(mid) * h;

        let sub = interval!(xi.inf(), xi1.sup())?;

        let coeff = h.powi(3) / TWENTY_FOUR;
        let error_term = f_pp(sub) * coeff;

        acc += point_value + error_term;
    }

    Ok(acc)
}

/// Trapezoidal rule with certified second-derivative error term.
///
/// Per subinterval [x_i,x_i+h]:
///
/// h/2 (f(x_i)+f(x_{i+1}))
///     - h^3/12 * f''([x_i,x_{i+1}])
pub fn trapezoidal_certified<F, FPP>(
    f: F,
    f_pp: FPP,
    a: f64,
    b: f64,
    n: u32,
) -> Result<Interval, IntervalError>
where
    F: Fn(Interval) -> Interval,
    FPP: Fn(Interval) -> Interval,
{
    let ia = interval!(a, a)?;
    let ib = interval!(b, b)?;
    let i_n = interval!(n as f64, n as f64)?;

    let h = (ib - ia) / i_n;
    let mut acc = ZERO;

    for i in 0..n {
        let ii = interval!(i as f64, i as f64)?;

        let xi = ia + ii * h;
        let xi1 = ia + (ii + ONE) * h;

        let point_value = (h / TWO) * (f(xi) + f(xi1));

        let sub = interval!(xi.inf(), xi1.sup())?;

        let coeff = h.powi(3) / TWELVE;
        let error_term = f_pp(sub) * coeff;

        acc += point_value - error_term;
    }

    Ok(acc)
}

/// Simpson's rule with certified fourth-derivative error term.
///
/// Per subinterval [x_i,x_i+h]:
///
/// h/6 (f(x_i) + 4f(mid) + f(x_{i+1}))
///     - h^5/2880 * f''''([x_i,x_{i+1}])
pub fn simpson_certified<F, F4>(
    f: F,
    f_4: F4,
    a: f64,
    b: f64,
    n: u32,
) -> Result<Interval, IntervalError>
where
    F: Fn(Interval) -> Interval,
    F4: Fn(Interval) -> Interval,
{
    let ia = interval!(a, a)?;
    let ib = interval!(b, b)?;
    let i_n = interval!(n as f64, n as f64)?;

    let h = (ib - ia) / i_n;
    let mut acc = ZERO;

    for i in 0..n {
        let ii = interval!(i as f64, i as f64)?;

        let xi = ia + ii * h;
        let xi1 = ia + (ii + ONE) * h;
        let mid = (xi + xi1) / TWO;

        let point_value = (h / SIX) * (f(xi) + FOUR * f(mid) + f(xi1));

        let sub = interval!(xi.inf(), xi1.sup())?;

        let coeff = h.powi(5) / TWO_THOUSAND_EIGHT_HUNDRED_EIGHTY;
        let error_term = f_4(sub) * coeff;

        acc += point_value - error_term;
    }

    Ok(acc)
}

/// Generic Gaussian quadrature rule on [-1,1].
pub trait GaussianRule {
    fn order(&self) -> usize;
    fn nodes(&self) -> &[f64];
    fn weights(&self) -> &[f64];
    fn c_n(&self, a: f64, b: f64) -> f64;

    fn transported(&self, a: f64, b: f64) -> Result<(Vec<Interval>, Vec<Interval>), IntervalError> {
        let half = interval!((b - a) / 2.0, (b - a) / 2.0)?;
        let mid = interval!((a + b) / 2.0, (a + b) / 2.0)?;

        let nodes = self
            .nodes()
            .iter()
            .map(|&s| {
                let s = interval!(s, s).unwrap();
                mid + half * s
            })
            .collect();

        let weights = self
            .weights()
            .iter()
            .map(|&w| {
                let w = interval!(w, w).unwrap();
                half * w
            })
            .collect();

        Ok((nodes, weights))
    }
}

/// Generic Gaussian quadrature with certified 2n-th derivative error term.
pub fn gaussian_certified<R, F, F2N>(
    rule: &R,
    f: F,
    f_2n: F2N,
    a: f64,
    b: f64,
    n: u32,
) -> Result<Interval, IntervalError>
where
    R: GaussianRule,
    F: Fn(Interval) -> Interval,
    F2N: Fn(Interval) -> Interval,
{
    let ia = interval!(a, a)?;
    let ib = interval!(b, b)?;
    let i_n = interval!(n as f64, n as f64)?;

    let h = (ib - ia) / i_n;
    let two_n = 2 * rule.order();

    let mut fact_2n = 1.0_f64;

    for k in 1..=two_n {
        fact_2n *= k as f64;
    }

    let mut acc = ZERO;

    for i in 0..n {
        let ii = interval!(i as f64, i as f64)?;

        let xi = ia + ii * h;
        let xi1 = ia + (ii + ONE) * h;

        let (nodes, weights) = rule.transported(xi.inf(), xi1.sup())?;

        let mut point_value = ZERO;

        for (&x, &w) in nodes.iter().zip(weights.iter()) {
            point_value += w * f(x);
        }

        let sub = interval!(xi.inf(), xi1.sup())?;

        let c_n = rule.c_n(xi.inf(), xi1.sup());

        let coeff = interval!(c_n / fact_2n, c_n / fact_2n)?;

        let error_term = f_2n(sub) * coeff;

        acc += point_value + error_term;
    }

    Ok(acc)
}

/// Gauss-Legendre quadrature rule:
/// nodes and weights on [-1,1] via Golub-Welsch.
pub struct GaussLegendreRule {
    order: usize,
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl GaussLegendreRule {
    pub fn new(order: usize) -> Self {
        assert!(order >= 1);

        let beta0 = 2.0;

        let mut jacobi = DMatrix::<f64>::zeros(order, order);

        for k in 1..order {
            let kf = k as f64;

            let beta_k = kf.powi(2) / (4.0 * kf.powi(2) - 1.0);

            let off = beta_k.sqrt();

            jacobi[(k, k - 1)] = off;
            jacobi[(k - 1, k)] = off;
        }

        let eig = SymmetricEigen::new(jacobi);

        let mut pairs: Vec<(f64, f64)> = (0..order)
            .map(|i| {
                let node = eig.eigenvalues[i];

                let v0 = eig.eigenvectors[(0, i)];

                (node, beta0 * v0 * v0)
            })
            .collect();

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Self {
            order,
            nodes: pairs.iter().map(|p| p.0).collect(),
            weights: pairs.iter().map(|p| p.1).collect(),
        }
    }
}

impl GaussianRule for GaussLegendreRule {
    fn order(&self) -> usize {
        self.order
    }

    fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn c_n(&self, a: f64, b: f64) -> f64 {
        let n = self.order as f64;
        let half = (b - a) / 2.0;

        let mut fact_n = 1.0_f64;

        for k in 1..=self.order {
            fact_n *= k as f64;
        }

        let mut fact_2n = 1.0_f64;

        for k in 1..=(2 * self.order) {
            fact_2n *= k as f64;
        }

        let k_n_inv = 2f64.powf(n) * fact_n * fact_n / fact_2n;

        half.powf(2.0 * n + 1.0) * (2.0 / (2.0 * n + 1.0)) * k_n_inv * k_n_inv
    }
}

/// Gauss-Legendre quadrature with certified 2n-th
/// derivative error term.
pub fn gauss_legendre_certified<F, F2N>(
    rule: &GaussLegendreRule,
    f: F,
    f_2n: F2N,
    a: f64,
    b: f64,
    n: u32,
) -> Result<Interval, IntervalError>
where
    F: Fn(Interval) -> Interval,
    F2N: Fn(Interval) -> Interval,
{
    gaussian_certified(rule, f, f_2n, a, b, n)
}
