use inari::{Interval, interval};
use nalgebra::{DMatrix, SymmetricEigen};

/// Midpoint rule with certified second-derivative error term.
/// Per subinterval [x_i, x_i+h]:
///   h f((x_i+x_i+h)/2) + h^3 / 24 * f''([x_i,x_i+h])
pub fn midpoint_certified<F, FPP>(f: F, f_pp: FPP, a: f64, b: f64, n: u32) -> Interval
where
    F: Fn(f64) -> f64,
    FPP: Fn(Interval) -> Interval,
{
    let h = (b - a) / n as f64;
    let mut acc = interval!(0.0, 0.0).unwrap();

    for i in 0..n {
        let xi = a + (i as f64) * h;
        let xi1 = xi + h;
        let mid = (xi + xi1) / 2.0;

        let point_value = f(mid) * h;
        let sub = interval!(xi, xi1).unwrap();

        let coeff = interval!(h.powi(3) / 24.0, h.powi(3) / 24.0).unwrap();

        let error_term = f_pp(sub) * coeff;
        let point_interval = interval!(point_value, point_value).unwrap();

        acc += point_interval + error_term;
    }

    acc
}

/// Trapezoidal rule with certified second-derivative error term.
/// Per subinterval [x_i,x_i+h]:
///   h/2 (f(x_i)+f(x_i+h)) - h^3/12 * f''([x_i,x_i+h])
pub fn trapezoidal_certified<F, FPP>(f: F, f_pp: FPP, a: f64, b: f64, n: u32) -> Interval
where
    F: Fn(f64) -> f64,
    FPP: Fn(Interval) -> Interval,
{
    let h = (b - a) / n as f64;
    let mut acc = interval!(0.0, 0.0).unwrap();

    for i in 0..n {
        let xi = a + (i as f64) * h;
        let xi1 = xi + h;

        let point_value = (h / 2.0) * (f(xi) + f(xi1));

        let sub = interval!(xi, xi1).unwrap();

        let coeff = interval!(h.powi(3) / 12.0, h.powi(3) / 12.0).unwrap();

        let error_term = f_pp(sub) * coeff;
        let point_interval = interval!(point_value, point_value).unwrap();

        acc = acc + point_interval - error_term;
    }

    acc
}

/// Simpson's rule with certified fourth-derivative error term.
/// Per subinterval [x_i,x_i+h]:
///   h/6 (f(x_i)+4f(mid)+f(x_i+h))
///   - h^5/2880 * f''''([x_i,x_i+h])
pub fn simpson_certified<F, F4>(f: F, f_4: F4, a: f64, b: f64, n: u32) -> Interval
where
    F: Fn(f64) -> f64,
    F4: Fn(Interval) -> Interval,
{
    let h = (b - a) / n as f64;
    let mut acc = interval!(0.0, 0.0).unwrap();

    for i in 0..n {
        let xi = a + (i as f64) * h;
        let xi1 = xi + h;
        let mid = (xi + xi1) / 2.0;

        let point_value = (h / 6.0) * (f(xi) + 4.0 * f(mid) + f(xi1));

        let sub = interval!(xi, xi1).unwrap();

        let coeff = interval!(h.powi(5) / 2880.0, h.powi(5) / 2880.0).unwrap();

        let error_term = f_4(sub) * coeff;
        let point_interval = interval!(point_value, point_value).unwrap();

        acc = acc + point_interval - error_term;
    }

    acc
}

/// Generic Gaussian quadrature rule on [-1,1].
pub trait GaussianRule {
    fn order(&self) -> usize;
    fn nodes(&self) -> &[f64];
    fn weights(&self) -> &[f64];
    fn c_n(&self, a: f64, b: f64) -> f64;

    fn transported(&self, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
        let half = (b - a) / 2.0;
        let mid = (a + b) / 2.0;

        let nodes = self.nodes().iter().map(|&s| half * s + mid).collect();

        let weights = self.weights().iter().map(|&w| half * w).collect();

        (nodes, weights)
    }
}

/// Generic Gaussian quadrature with certified 2n-th derivative error term.
pub fn gaussian_certified<R, F, F2N>(rule: &R, f: F, f_2n: F2N, a: f64, b: f64, n: u32) -> Interval
where
    R: GaussianRule,
    F: Fn(f64) -> f64,
    F2N: Fn(Interval) -> Interval,
{
    let h = (b - a) / n as f64;
    let two_n = 2 * rule.order();

    let mut fact_2n = 1.0_f64;
    for k in 1..=two_n {
        fact_2n *= k as f64;
    }

    let mut acc = interval!(0.0, 0.0).unwrap();

    for i in 0..n {
        let xi = a + (i as f64) * h;
        let xi1 = xi + h;

        let (nodes, weights) = rule.transported(xi, xi1);

        let point_value: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * f(x))
            .sum();

        let sub = interval!(xi, xi1).unwrap();

        let c_n = rule.c_n(xi, xi1);

        let coeff = interval!(c_n / fact_2n, c_n / fact_2n).unwrap();

        let error_term = f_2n(sub) * coeff;

        let point_interval = interval!(point_value, point_value).unwrap();

        acc = acc + point_interval + error_term;
    }

    acc
}

/// Gauss-Legendre quadrature rule: nodes/weights on [-1,1] via Golub-Welsch.
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

/// Gauss-Legendre quadrature with certified 2n-th derivative error term.
pub fn gauss_legendre_certified<F, F2N>(
    rule: &GaussLegendreRule,
    f: F,
    f_2n: F2N,
    a: f64,
    b: f64,
    n: u32,
) -> Interval
where
    F: Fn(f64) -> f64,
    F2N: Fn(Interval) -> Interval,
{
    gaussian_certified(rule, f, f_2n, a, b, n)
}
