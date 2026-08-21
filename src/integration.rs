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

fn pairwise_sum(intervals: &[Interval]) -> Interval {
    match intervals.len() {
        0 => ZERO,
        1 => intervals[0],
        _ => {
            let mid = intervals.len() / 2;
            pairwise_sum(&intervals[..mid]) + pairwise_sum(&intervals[mid..])
        }
    }
}

/// Computes a certified midpoint quadrature enclosure.
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
    let mut contributions = Vec::with_capacity(n as usize);

    for i in 0..n {
        let ii = interval!(i as f64, i as f64)?;
        let ii1 = interval!((i + 1) as f64, (i + 1) as f64)?;

        let xi = if i == 0 { ia } else { ia + ii * h };
        let xi1 = if i + 1 == n { ib } else { ia + ii1 * h };

        let mid = (xi + xi1) / TWO;

        let point_value = f(mid) * h;

        let sub = interval!(xi.inf(), xi1.sup())?;

        let coeff = h.powi(3) / TWENTY_FOUR;
        let error_term = f_pp(sub) * coeff;

        contributions.push(point_value + error_term);
    }

    Ok(pairwise_sum(&contributions))
}

/// Computes a certified trapezoidal quadrature enclosure.
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
    let mut contributions = Vec::with_capacity(n as usize);

    for i in 0..n {
        let ii = interval!(i as f64, i as f64)?;
        let ii1 = interval!((i + 1) as f64, (i + 1) as f64)?;

        let xi = if i == 0 { ia } else { ia + ii * h };
        let xi1 = if i + 1 == n { ib } else { ia + ii1 * h };

        let point_value = (h / TWO) * (f(xi) + f(xi1));

        let sub = interval!(xi.inf(), xi1.sup())?;

        let coeff = h.powi(3) / TWELVE;
        let error_term = f_pp(sub) * coeff;

        contributions.push(point_value - error_term);
    }

    Ok(pairwise_sum(&contributions))
}

/// Computes a certified Simpson quadrature enclosure.
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
    let mut contributions = Vec::with_capacity(n as usize);

    for i in 0..n {
        let ii = interval!(i as f64, i as f64)?;
        let ii1 = interval!((i + 1) as f64, (i + 1) as f64)?;

        let xi = if i == 0 { ia } else { ia + ii * h };
        let xi1 = if i + 1 == n { ib } else { ia + ii1 * h };

        let mid = (xi + xi1) / TWO;

        let point_value = (h / SIX) * (f(xi) + FOUR * f(mid) + f(xi1));

        let sub = interval!(xi.inf(), xi1.sup())?;

        let coeff = h.powi(5) / TWO_THOUSAND_EIGHT_HUNDRED_EIGHTY;
        let error_term = f_4(sub) * coeff;

        contributions.push(point_value - error_term);
    }

    Ok(pairwise_sum(&contributions))
}

/// Defines a Gaussian quadrature rule.
pub trait GaussianRule {
    fn order(&self) -> usize;

    fn nodes(&self) -> &[f64];

    fn weights(&self) -> &[f64];

    fn c_n(&self, a: Interval, b: Interval) -> Interval;

    /// Transports the rule from `[-1, 1]` to `[a, b]`.
    /// laisser les poids sur -1 1 et juste multiplier a  la fin par la taille
    fn transported(
        &self,
        a: Interval,
        b: Interval,
    ) -> Result<(Vec<Interval>, Vec<Interval>), IntervalError> {
        let half = (b - a) / TWO;
        let mid = (a + b) / TWO;

        let nodes = self
            .nodes()
            .iter()
            .map(|&s| {
                let s = interval!(s.next_down(), s.next_up()).unwrap();
                mid + half * s
            })
            .collect();

        let weights = self
            .weights()
            .iter()
            .map(|&w| interval!(w.next_down(), w.next_up()).unwrap())
            .collect();

        Ok((nodes, weights))
    }
}

/// Computes a certified Gaussian quadrature enclosure.
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
    let a = interval!(a, a)?;
    let b = interval!(b, b)?;
    let i_n = interval!(n as f64, n as f64)?;

    let h = (b - a) / i_n;
    let two_n = 2 * rule.order();

    let mut fact_2n = ONE;

    for k in 1..=two_n {
        fact_2n = fact_2n * interval!(k as f64, k as f64)?; // dois je faire k.next_down(), k.next_up() ?
    }

    let mut contributions = Vec::with_capacity(n as usize);

    for i in 0..n {
        let ii = interval!(i as f64, i as f64)?;
        let ii1 = interval!((i + 1) as f64, (i + 1) as f64)?;

        let xi = if i == 0 { a } else { a + ii * h };
        let xi1 = if i + 1 == n { b } else { a + ii1 * h };

        let (nodes, weights) = rule.transported(xi, xi1)?;

        let mut point_terms = Vec::with_capacity(nodes.len());

        for (&x, &w) in nodes.iter().zip(weights.iter()) {
            point_terms.push(w * f(x));
        }

        let sub_half = (xi1 - xi) / TWO;

        let point_value = pairwise_sum(&point_terms) * sub_half;

        let sub = interval!(xi.inf(), xi1.sup())?;

        let c_n = rule.c_n(xi, xi1);
        let coeff = c_n / fact_2n;

        let derivative = f_2n(sub);
        let error_term = derivative * coeff;

        contributions.push(point_value + error_term);
    }

    Ok(pairwise_sum(&contributions))
}

/// Stores a Gauss-Legendre quadrature rule.
pub struct GaussLegendreRule {
    order: usize,
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl GaussLegendreRule {
    /// Constructs a Gauss-Legendre rule.
    ///
    /// DLMF values are used for orders 5, 10 and 20.
    /// Other orders are computed using the Golub-Welsch algorithm.
    pub fn new(order: usize) -> Self {
        assert!(order >= 1);

        match order {
            5 => Self::from_dlmf(5),
            10 => Self::from_dlmf(10),
            20 => Self::from_dlmf(20),
            _ => Self::from_golub_welsch(order),
        }
    }

    fn from_golub_welsch(order: usize) -> Self {
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

    fn from_dlmf(order: usize) -> Self {
        let (nodes, weights): (&[f64], &[f64]) = match order {
            5 => (
                &[
                    -0.9061798459386640,
                    -0.5384693101056831,
                    0.0,
                    0.5384693101056831,
                    0.9061798459386640,
                ],
                &[
                    0.2369268850561891,
                    0.4786286704993665,
                    0.5688888888888889,
                    0.4786286704993665,
                    0.2369268850561891,
                ],
            ),
            10 => (
                &[
                    -0.9739065285171717,
                    -0.8650633666889845,
                    -0.6794095682990244,
                    -0.4333953941292472,
                    -0.1488743389816312,
                    0.1488743389816312,
                    0.4333953941292472,
                    0.6794095682990244,
                    0.8650633666889845,
                    0.9739065285171717,
                ],
                &[
                    0.0666713443086881,
                    0.1494513491505806,
                    0.2190863625159820,
                    0.2692667193099963,
                    0.2955242247147529,
                    0.2955242247147529,
                    0.2692667193099963,
                    0.2190863625159820,
                    0.1494513491505806,
                    0.0666713443086881,
                ],
            ),
            20 => (
                &[
                    -0.9931285991850949,
                    -0.9639719272779138,
                    -0.9122344282513259,
                    -0.8391169718222188,
                    -0.7463319064601508,
                    -0.6360536807265150,
                    -0.5108670019508271,
                    -0.3737060887154196,
                    -0.2277858511416451,
                    -0.0765265211334973,
                    0.0765265211334973,
                    0.2277858511416451,
                    0.3737060887154196,
                    0.5108670019508271,
                    0.6360536807265150,
                    0.7463319064601508,
                    0.8391169718222188,
                    0.9122344282513259,
                    0.9639719272779138,
                    0.9931285991850949,
                ],
                &[
                    0.0176140071391521,
                    0.0406014298190927,
                    0.0626720483341091,
                    0.0832767415767047,
                    0.1019301198172404,
                    0.1181945319615184,
                    0.1316886384491766,
                    0.1420961093183821,
                    0.1491729864726037,
                    0.1527533871307259,
                    0.1527533871307259,
                    0.1491729864726037,
                    0.1420961093183821,
                    0.1316886384491766,
                    0.1181945319615184,
                    0.1019301198172404,
                    0.0832767415767047,
                    0.0626720483341091,
                    0.0406014298190927,
                    0.0176140071391521,
                ],
            ),
            _ => unreachable!(),
        };

        Self {
            order,
            nodes: nodes.to_vec(),
            weights: weights.to_vec(),
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

    fn c_n(&self, a: Interval, b: Interval) -> Interval {
        let n = self.order;

        let half = (b - a) / TWO;

        let mut fact_n = ONE;

        for k in 1..=n {
            fact_n = fact_n * interval!(k as f64, k as f64).unwrap();
        }

        let mut fact_2n = ONE;

        for k in 1..=(2 * n) {
            fact_2n = fact_2n * interval!(k as f64, k as f64).unwrap();
        }

        let two_pow_n = TWO.powi(n as i32);

        let k_n_inv = two_pow_n * fact_n * fact_n / fact_2n;

        let exponent = (2 * n + 1) as i32;

        half.powi(exponent)
            * (TWO / interval!((2 * n + 1) as f64, (2 * n + 1) as f64).unwrap())
            * k_n_inv
            * k_n_inv
    }
}

/// Computes a certified Gauss-Legendre quadrature enclosure.
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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::hausdorff_nested;
    use std::f64::consts::PI;

    #[test]
    fn test_constant() {
        let rule = GaussLegendreRule::new(5);

        let result = gauss_legendre_certified(
            &rule,
            |_| interval!(1.0, 1.0).unwrap(),
            |_| interval!(0.0, 0.0).unwrap(),
            0.0,
            1.0,
            1,
        )
        .unwrap();

        println!("result = {}", result);

        assert!(result.contains(1.0));
    }

    #[test]
    fn test_linear() {
        let rule = GaussLegendreRule::new(5);

        let result =
            gauss_legendre_certified(&rule, |x| x, |_| interval!(0.0, 0.0).unwrap(), 0.0, 1.0, 1)
                .unwrap();

        println!("result = {}", result);

        assert!(result.contains(0.5));
        assert!(result.wid() <= 5e-16);
    }

    #[test]
    fn test_quadratic() {
        let rule = GaussLegendreRule::new(5);

        let result = gauss_legendre_certified(
            &rule,
            |x| x.sqr(),
            |_| interval!(0.0, 0.0).unwrap(),
            0.0,
            1.0,
            1,
        )
        .unwrap();

        assert!(result.contains(1.0 / 3.0));
    }

    #[test]
    fn test_cubic() {
        let rule = GaussLegendreRule::new(5);

        let result = gauss_legendre_certified(
            &rule,
            |x| x.powi(3),
            |_| interval!(0.0, 0.0).unwrap(),
            0.0,
            1.0,
            1,
        )
        .unwrap();

        assert!(result.contains(1.0 / 4.0));
    }

    #[test]
    fn test_x5() {
        let rule = GaussLegendreRule::new(5);

        let result = gauss_legendre_certified(
            &rule,
            |x| x.powi(5),
            |_| interval!(0.0, 0.0).unwrap(),
            0.0,
            1.0,
            1,
        )
        .unwrap();

        println!("result = {}", result);

        assert!(result.contains(1.0 / 6.0));
    }

    #[test]
    fn test_x6() {
        let rule = GaussLegendreRule::new(5);

        let result = gauss_legendre_certified(
            &rule,
            |x| x.powi(6),
            |_| interval!(0.0, 0.0).unwrap(),
            0.0,
            1.0,
            1,
        )
        .unwrap();

        println!("result = {}", result);

        assert!(result.contains(1.0 / 7.0));
    }

    #[test]
    fn test_x10() {
        let rule = GaussLegendreRule::new(5);

        let result = gauss_legendre_certified(
            &rule,
            |x| x.powi(10),
            |_| interval!(3628800.0, 3628800.0).unwrap(),
            0.0,
            1.0,
            1,
        )
        .unwrap();

        println!("result = {}", result);

        assert!(result.contains(1.0 / 11.0));
    }

    #[test]
    fn test_exponential() {
        let rule = GaussLegendreRule::new(5);

        let result =
            gauss_legendre_certified(&rule, |x| x.exp(), |x| x.exp(), 0.0, 1.0, 1).unwrap();

        assert!(result.contains(std::f64::consts::E - 1.0));
    }

    #[test]
    fn test_sine() {
        let rule = GaussLegendreRule::new(5);

        let result =
            gauss_legendre_certified(&rule, |x| x.sin(), |x| -x.sin(), 0.0, PI, 1).unwrap();

        assert!(result.contains(2.0));
    }

    #[test]
    fn test_transport() {
        let rule = GaussLegendreRule::new(5);
        let a = interval!(0.0, 0.0).unwrap();
        let b = interval!(1.0, 1.0).unwrap();
        let (nodes, weights) = rule.transported(a, b).unwrap();

        let half = (b - a) / TWO;

        let result = half
            * pairwise_sum(
                &nodes
                    .iter()
                    .zip(weights.iter())
                    .map(|(&x, &w)| w * x.powi(5))
                    .collect::<Vec<_>>(),
            );

        println!("result = {}", result);
        assert!(result.contains(1.0 / 6.0));
    }

    #[test]
    fn test_midpoint_constant() {
        let result = midpoint_certified(
            |_| interval!(1.0, 1.0).unwrap(),
            |_| interval!(0.0, 0.0).unwrap(),
            0.0,
            1.0,
            10,
        )
        .unwrap();
        assert!(result.contains(1.0));
    }

    #[test]
    fn test_midpoint_quadratic() {
        let result =
            midpoint_certified(|x| x * x, |_| interval!(2.0, 2.0).unwrap(), 0.0, 1.0, 10).unwrap();

        assert!(result.contains(1.0 / 3.0));
    }

    #[test]
    fn test_trapezoidal_constant() {
        let result = trapezoidal_certified(
            |_| interval!(1.0, 1.0).unwrap(),
            |_| interval!(0.0, 0.0).unwrap(),
            0.0,
            1.0,
            10,
        )
        .unwrap();

        assert!(result.contains(1.0));
    }

    #[test]
    fn test_trapezoidal_quadratic() {
        let result =
            trapezoidal_certified(|x| x * x, |_| interval!(2.0, 2.0).unwrap(), 0.0, 1.0, 10)
                .unwrap();

        assert!(result.contains(1.0 / 3.0));
    }

    #[test]
    fn test_simpson_constant() {
        let result = simpson_certified(
            |_| interval!(1.0, 1.0).unwrap(),
            |_| interval!(0.0, 0.0).unwrap(),
            0.0,
            1.0,
            10,
        )
        .unwrap();

        assert!(result.contains(1.0));
    }

    #[test]
    fn test_simpson_quartic() {
        let result = simpson_certified(
            |x| x.powi(4),
            |_| interval!(24.0, 24.0).unwrap(),
            0.0,
            1.0,
            10,
        )
        .unwrap();

        assert!(result.contains(1.0 / 5.0));
    }
    #[test]
    fn test_golub_welsch_nodes_within_1ulp_of_dlmf() {
        for &order in &[5usize, 10, 20] {
            let gw = GaussLegendreRule::from_golub_welsch(order);
            let dlmf = GaussLegendreRule::from_dlmf(order);

            for i in 0..order {
                let gw_node = gw.nodes[i];
                let dlmf_node = dlmf.nodes[i];

                let lo = gw_node.next_down();
                let hi = gw_node.next_up();

                let abs_err = (gw_node - dlmf_node).abs();

                assert!(
                    lo <= dlmf_node && dlmf_node <= hi,
                    "order={} i={}: DLMF node {:.17e} vs Golub-Welsch node \
                     {:.17e} — absolute error = {:.3e} (exceeds 1-ulp \
                     tolerance window [{:.17e}, {:.17e}])",
                    order,
                    i,
                    dlmf_node,
                    gw_node,
                    abs_err,
                    lo,
                    hi
                );
            }
        }
    }

    #[test]
    fn test_golub_welsch_weights_within_1ulp_of_dlmf() {
        for &order in &[5usize, 10, 20] {
            let gw = GaussLegendreRule::from_golub_welsch(order);
            let dlmf = GaussLegendreRule::from_dlmf(order);

            for i in 0..order {
                let gw_weight = gw.weights[i];
                let dlmf_weight = dlmf.weights[i];

                let lo = gw_weight.next_down();
                let hi = gw_weight.next_up();

                let abs_err = (gw_weight - dlmf_weight).abs();

                assert!(
                    lo <= dlmf_weight && dlmf_weight <= hi,
                    "order={} i={}: DLMF weight {:.17e} vs Golub-Welsch \
                     weight {:.17e} — absolute error = {:.3e} (exceeds \
                     1-ulp tolerance window [{:.17e}, {:.17e}])",
                    order,
                    i,
                    dlmf_weight,
                    gw_weight,
                    abs_err,
                    lo,
                    hi
                );
            }
        }
    }
    #[test]
    fn test_golub_welsch_encloses_exact_value() {
        for &order in &[5usize, 10, 20] {
            let rule = GaussLegendreRule::from_golub_welsch(order);

            let f = |x: Interval| -> Interval { x.powi(5) };
            let f_2n = |_x: Interval| -> Interval { ZERO };

            let result = gauss_legendre_certified(&rule, f, f_2n, 0.0, 1.0, 1).unwrap();

            let exact = 1.0 / 6.0;

            println!(
                "order = {:2} | result = [{:.17e}, {:.17e}] | encloses exact = {}, distance of the intervals  = {:.17e} ",
                order,
                result.inf(),
                result.sup(),
                result.contains(exact),
                hausdorff_nested(result, interval!(exact, exact).unwrap())
            );

            assert!(
                result.contains(exact),
                "order={}: result {} does not enclose exact value {}",
                order,
                result,
                exact
            );
        }
    }
}
