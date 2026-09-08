//! Adaptive-refinement certified quadrature: recursively subdivides the
//! integration interval, prioritizing (via a `BinaryHeap`) the cell with
//! the largest error term ([`LocalCell::error_bound`]), until the total
//! enclosure width drops below `tolerance`. [`LocalQuadrature`]
//! generalizes the local rule used (midpoint, trapezoidal, Simpson,
//! Gauss-Legendre — see [`integration`](crate::integration) for the
//! non-adaptive variants).

use crate::{
    factorial_interval,
    integration::{GaussLegendreRule, GaussianRule},
};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use inari::{Interval, IntervalError, const_interval, interval};

const ZERO: Interval = const_interval!(0.0, 0.0);
const TWO: Interval = const_interval!(2.0, 2.0);
const FOUR: Interval = const_interval!(4.0, 4.0);
const SIX: Interval = const_interval!(6.0, 6.0);
const TWELVE: Interval = const_interval!(12.0, 12.0);
const TWENTY_FOUR: Interval = const_interval!(24.0, 24.0);
const TWO_THOUSAND_EIGHT_HUNDRED_EIGHTY: Interval = const_interval!(2880.0, 2880.0);

/// Computes the sum of a slice of intervals using a pairwise (binary-tree) algorithm.
/// This reduces the worst-case rounding error compared to a simple left fold.
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

/// Enclosure of ∫ₐᵇ f over a sub-cell `[a, b]`, with the upper bound
/// `error_bound` of the error term (used to order the priority queue in
/// [`adaptive_integration`]).
#[derive(Clone, Copy)]
pub struct LocalCell {
    /// Left endpoint of the subinterval.
    pub a: f64,
    /// Right endpoint of the subinterval.
    pub b: f64,
    /// Enclosure of the integral over `[a, b]`.
    pub enclosure: Interval,
    /// Absolute error bound (used as priority key for refinement).
    pub error_bound: f64,
}

/// Newtype wrapper giving `LocalCell` a total order based on `error_bound`,
/// so cells can live in a `BinaryHeap` (which is a max-heap by default).
///
/// The largest `error_bound` will be at the top, so we refine the most
/// inaccurate cell first.
struct HeapCell(LocalCell);

impl PartialEq for HeapCell {
    fn eq(&self, other: &Self) -> bool {
        self.0.error_bound == other.0.error_bound
    }
}

impl Eq for HeapCell {}

impl PartialOrd for HeapCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCell {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order so that larger error_bound comes first.
        self.0
            .error_bound
            .partial_cmp(&other.0.error_bound)
            .unwrap_or(Ordering::Equal)
    }
}

/// Local quadrature rule: encloses ∫ₐᵇ f over a single cell `[a, b]` and
/// provides the associated error term. Implemented by [`Midpoint`],
/// [`Trapezoidal`], [`Simpson`] and [`GaussLegendre`].

pub trait LocalQuadrature<F> {
    /// Integrates `f` on `[a, b]` and returns a `LocalCell`.
    fn integrate_cell(&self, f: &F, a: f64, b: f64) -> Result<LocalCell, IntervalError>;
}

/// Midpoint rule with rigorous error bound using the second derivative.
pub struct Midpoint<FPP> {
    /// The second derivative of the integrand as an interval extension.
    pub f_pp: FPP,
}

impl<F, FPP> LocalQuadrature<F> for Midpoint<FPP>
where
    F: Fn(Interval) -> Interval,
    FPP: Fn(Interval) -> Interval,
{
    fn integrate_cell(&self, f: &F, a: f64, b: f64) -> Result<LocalCell, IntervalError> {
        let ia = interval!(a, a)?;
        let ib = interval!(b, b)?;

        let h = ib - ia;
        let mid = (ia + ib) / TWO;

        // Midpoint approximation: h * f(mid)
        let quadrature_value = f(mid) * h;

        // Error bound: (h^3 / 24) * f''([a, b])
        let sub = interval!(a, b)?;
        let error_term = (self.f_pp)(sub) * h.powi(3) / TWENTY_FOUR;

        let enclosure = quadrature_value + error_term;
        let error_bound = error_term.inf().abs().max(error_term.sup().abs());

        Ok(LocalCell {
            a,
            b,
            enclosure,
            error_bound,
        })
    }
}

/// Trapezoidal rule with rigorous error bound using the second derivative.
pub struct Trapezoidal<FPP> {
    /// The second derivative of the integrand as an interval extension.
    pub f_pp: FPP,
}

impl<F, FPP> LocalQuadrature<F> for Trapezoidal<FPP>
where
    F: Fn(Interval) -> Interval,
    FPP: Fn(Interval) -> Interval,
{
    fn integrate_cell(&self, f: &F, a: f64, b: f64) -> Result<LocalCell, IntervalError> {
        let ia = interval!(a, a)?;
        let ib = interval!(b, b)?;

        let h = ib - ia;

        // Trapezoidal rule: (h/2) * (f(a) + f(b))
        let quadrature_value = (h / TWO) * (f(ia) + f(ib));

        // Error bound: -(h^3 / 12) * f''([a, b])
        let sub = interval!(a, b)?;
        let error_term = (self.f_pp)(sub) * h.powi(3) / TWELVE;

        let enclosure = quadrature_value - error_term;
        let error_bound = error_term.inf().abs().max(error_term.sup().abs());

        Ok(LocalCell {
            a,
            b,
            enclosure,
            error_bound,
        })
    }
}

/// Simpson's rule with rigorous error bound using the fourth derivative.
pub struct Simpson<F4> {
    /// The fourth derivative of the integrand as an interval extension.
    pub f_4: F4,
}

impl<F, F4> LocalQuadrature<F> for Simpson<F4>
where
    F: Fn(Interval) -> Interval,
    F4: Fn(Interval) -> Interval,
{
    fn integrate_cell(&self, f: &F, a: f64, b: f64) -> Result<LocalCell, IntervalError> {
        let ia = interval!(a, a)?;
        let ib = interval!(b, b)?;

        let h = ib - ia;
        let mid = (ia + ib) / TWO;

        // Simpson's rule: (h/6) * (f(a) + 4f(mid) + f(b))
        let quadrature_value = (h / SIX) * (f(ia) + FOUR * f(mid) + f(ib));

        // Error bound: -(h^5 / 2880) * f''''([a, b])
        let sub = interval!(a, b)?;
        let error_term = (self.f_4)(sub) * h.powi(5) / TWO_THOUSAND_EIGHT_HUNDRED_EIGHTY;

        let enclosure = quadrature_value - error_term;
        let error_bound = error_term.inf().abs().max(error_term.sup().abs());

        Ok(LocalCell {
            a,
            b,
            enclosure,
            error_bound,
        })
    }
}

/// Gauss-Legendre quadrature as a local rule, using a precomputed rule of fixed order.
///
/// The error bound uses the `2*order`-th derivative.
pub struct GaussLegendre<'a, F2N> {
    /// The Gauss-Legendre rule to use (order determines error bound).
    pub rule: &'a GaussLegendreRule,
    /// The `2*order`-th derivative of the integrand as an interval extension.
    pub f_2n: F2N,
}

impl<F, F2N> LocalQuadrature<F> for GaussLegendre<'_, F2N>
where
    F: Fn(Interval) -> Interval,
    F2N: Fn(Interval) -> Interval,
{
    fn integrate_cell(&self, f: &F, a: f64, b: f64) -> Result<LocalCell, IntervalError> {
        let ia = interval!(a, a)?;
        let ib = interval!(b, b)?;

        // Transport the rule to [ia, ib]
        let (nodes, weights) = self.rule.transported(ia, ib)?;

        // Compute the quadrature sum: Σ w_j * f(x_j)
        let point_terms: Vec<Interval> = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * f(x))
            .collect();

        let half = (ib - ia) / TWO;
        let point_value = pairwise_sum(&point_terms) * half;

        // Error bound: (C_n / (2n)!) * f^(2n)([a, b])
        let sub = interval!(a, b)?;
        let n = self.rule.order();
        let fact_2n = factorial_interval(2 * n)?;
        let c_n = self.rule.c_n(ia, ib);
        let coeff = c_n / fact_2n;

        let error_term = (self.f_2n)(sub) * coeff;
        let enclosure = point_value + error_term;

        let error_bound = error_term.inf().abs().max(error_term.sup().abs());

        Ok(LocalCell {
            a,
            b,
            enclosure,
            error_bound,
        })
    }
}

/// Encloses ∫ₐᵇ f by adaptive refinement: at each iteration, splits the
/// cell with the largest `error_bound` (priority queue) in two, until
/// the total enclosure width drops below `tolerance` or `max_cells`
/// cells have been created.
///
/// # Panics
/// Panics if `a > b`, `tolerance <= 0.0` or `max_cells == 0`.

pub fn adaptive_integration<F, M>(
    f: F,
    method: M,
    a: f64,
    b: f64,
    tolerance: f64,
    max_cells: usize,
) -> Result<Interval, IntervalError>
where
    F: Fn(Interval) -> Interval,
    M: LocalQuadrature<F>,
{
    assert!(a <= b, "a must be less than or equal to b");
    assert!(tolerance > 0.0, "tolerance must be positive");
    assert!(max_cells >= 1, "max_cells must be at least 1");

    // Initial cell covering the whole interval
    let first = method.integrate_cell(&f, a, b)?;

    // Priority queue (max-heap) ordered by error_bound
    let mut heap: BinaryHeap<HeapCell> = BinaryHeap::new();
    heap.push(HeapCell(first));

    let mut count = 1usize;
    let mut total = first.enclosure;

    while count < max_cells && total.wid() > tolerance {
        // Pop the cell with the largest error
        let HeapCell(cell) = heap.pop().expect("heap must not be empty");

        // Split the cell in half
        let middle = cell.a + (cell.b - cell.a) / 2.0;

        let left = method.integrate_cell(&f, cell.a, middle)?;
        let right = method.integrate_cell(&f, middle, cell.b)?;

        heap.push(HeapCell(left));
        heap.push(HeapCell(right));
        count += 1;

        // Recompute total from all cells in the heap
        let enclosures: Vec<Interval> = heap.iter().map(|hc| hc.0.enclosure).collect();
        total = pairwise_sum(&enclosures);
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midpoint_adaptive_quadratic() {
        let result = adaptive_integration(
            |x: Interval| x.sqr(),
            Midpoint {
                f_pp: |_| interval!(2.0, 2.0).unwrap(),
            },
            0.0,
            1.0,
            1e-16,
            10_000,
        )
        .unwrap();

        let exact = 1.0 / 3.0;

        println!("Result: {}", result);
        println!("Exact value: {}", exact);
        println!("Width: {:.17e}", result.wid());

        assert!(result.contains(exact));
        assert!(result.wid() <= 1e-12);
    }

    #[test]
    fn test_midpoint_adaptive_exp() {
        let result = adaptive_integration(
            |x: Interval| x.exp(),
            Midpoint {
                f_pp: |x: Interval| x.exp(),
            },
            0.0,
            10.0,
            1e-10,
            1_000,
        )
        .unwrap();

        let exact = 10.0_f64.exp() - 1.0;

        println!("Result: {}", result);
        println!("Exact value: {}", exact);
        println!("Width: {:.17e}", result.wid());

        assert!(result.contains(exact));
    }

    #[test]
    fn test_trapezoidal_adaptive_exp() {
        let result = adaptive_integration(
            |x: Interval| x.exp(),
            Trapezoidal {
                f_pp: |x: Interval| x.exp(),
            },
            0.0,
            10.0,
            1e-10,
            1_000,
        )
        .unwrap();

        let exact = 10.0_f64.exp() - 1.0;

        println!("Result: {}", result);
        println!("Exact value: {}", exact);
        println!("Width: {:.17e}", result.wid());

        assert!(result.contains(exact));
    }

    #[test]
    fn test_simpson_adaptive_exp() {
        let result = adaptive_integration(
            |x: Interval| x.exp(),
            Simpson {
                f_4: |x: Interval| x.exp(),
            },
            0.0,
            10.0,
            1e-10,
            1_000,
        )
        .unwrap();

        let exact = 10.0_f64.exp() - 1.0;

        println!("Result: {}", result);
        println!("Exact value: {}", exact);
        println!("Width: {:.17e}", result.wid());

        assert!(result.contains(exact));
    }

    #[test]
    fn test_gauss_legendre_adaptive_exp() {
        let rule = GaussLegendreRule::new(5);

        let result = adaptive_integration(
            |x: Interval| x.exp(),
            GaussLegendre {
                rule: &rule,
                f_2n: |x: Interval| x.exp(), // f^(10)(x) = exp(x)
            },
            0.0,
            10.0,
            1e-10,
            1_000,
        )
        .unwrap();

        let exact = 10.0_f64.exp() - 1.0;

        println!("Result: {}", result);
        println!("Exact value: {}", exact);
        println!("Width: {:.17e}", result.wid());

        assert!(result.contains(exact));
    }
}
