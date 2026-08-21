use inari::{const_interval, interval, Interval, IntervalError};

const ZERO: Interval = const_interval!(0.0, 0.0);
const TWO: Interval = const_interval!(2.0, 2.0);
const FOUR: Interval = const_interval!(4.0, 4.0);
const SIX: Interval = const_interval!(6.0, 6.0);
const TWELVE: Interval = const_interval!(12.0, 12.0);
const TWENTY_FOUR: Interval = const_interval!(24.0, 24.0);
const TWO_THOUSAND_EIGHT_HUNDRED_EIGHTY: Interval =
    const_interval!(2880.0, 2880.0);

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

#[derive(Clone, Copy)]
pub struct LocalCell {
    a: f64,
    b: f64,
    enclosure: Interval,
    error_bound: f64,
}

pub trait LocalQuadrature<F> {
    fn integrate_cell(
        &self,
        f: &F,
        a: f64,
        b: f64,
    ) -> Result<LocalCell, IntervalError>;
}

pub struct Midpoint<FPP> {
    pub f_pp: FPP,
}

impl<F, FPP> LocalQuadrature<F> for Midpoint<FPP>
where
    F: Fn(Interval) -> Interval,
    FPP: Fn(Interval) -> Interval,
{
    fn integrate_cell(
        &self,
        f: &F,
        a: f64,
        b: f64,
    ) -> Result<LocalCell, IntervalError> {
        let ia = interval!(a, a)?;
        let ib = interval!(b, b)?;

        let h = ib - ia;
        let mid = (ia + ib) / TWO;

        let quadrature_value = f(mid) * h;

        // Interval on which f'' is evaluated.
        let sub = interval!(a, b)?;

        // ∫f ∈ h f(mid) + h³/24 f''([a,b])
        let error_term = (self.f_pp)(sub) * h.powi(3) / TWENTY_FOUR;

        let enclosure = quadrature_value + error_term;

        let error_bound = error_term
            .inf()
            .abs()
            .max(error_term.sup().abs());

        Ok(LocalCell {
            a,
            b,
            enclosure,
            error_bound,
        })
    }
}

pub struct Trapezoidal<FPP> {
    pub f_pp: FPP,
}

impl<F, FPP> LocalQuadrature<F> for Trapezoidal<FPP>
where
    F: Fn(Interval) -> Interval,
    FPP: Fn(Interval) -> Interval,
{
    fn integrate_cell(
        &self,
        f: &F,
        a: f64,
        b: f64,
    ) -> Result<LocalCell, IntervalError> {
        let ia = interval!(a, a)?;
        let ib = interval!(b, b)?;

        let h = ib - ia;
        let sub = interval!(a, b)?;

        let quadrature_value = (h / TWO) * (f(ia) + f(ib));

        // ∫f ∈ T - h³/12 f''([a,b])
        let error_term = (self.f_pp)(sub) * h.powi(3) / TWELVE;

        let enclosure = quadrature_value - error_term;

        let error_bound = error_term
            .inf()
            .abs()
            .max(error_term.sup().abs());

        Ok(LocalCell {
            a,
            b,
            enclosure,
            error_bound,
        })
    }
}

pub struct Simpson<F4> {
    pub f_4: F4,
}

impl<F, F4> LocalQuadrature<F> for Simpson<F4>
where
    F: Fn(Interval) -> Interval,
    F4: Fn(Interval) -> Interval,
{
    fn integrate_cell(
        &self,
        f: &F,
        a: f64,
        b: f64,
    ) -> Result<LocalCell, IntervalError> {
        let ia = interval!(a, a)?;
        let ib = interval!(b, b)?;

        let h = ib - ia;
        let mid = (ia + ib) / TWO;
        let sub = interval!(a, b)?;

        let quadrature_value =
            (h / SIX) * (f(ia) + FOUR * f(mid) + f(ib));

        // ∫f ∈ S - h⁵/2880 f⁽⁴⁾([a,b])
        let error_term =
            (self.f_4)(sub) * h.powi(5) / TWO_THOUSAND_EIGHT_HUNDRED_EIGHTY;

        let enclosure = quadrature_value - error_term;

        let error_bound = error_term
            .inf()
            .abs()
            .max(error_term.sup().abs());

        Ok(LocalCell {
            a,
            b,
            enclosure,
            error_bound,
        })
    }
}

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

    let mut cells = vec![method.integrate_cell(&f, a, b)?];

    loop {
        let total = pairwise_sum(
            &cells
                .iter()
                .map(|cell| cell.enclosure)
                .collect::<Vec<_>>(),
        );

        if total.wid() <= tolerance {
            return Ok(total);
        }

        if cells.len() >= max_cells {
            return Ok(total);
        }

        let index = cells
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.error_bound
                    .partial_cmp(&right.error_bound)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
            .expect("cells must not be empty");

        let cell = cells.swap_remove(index);

        let middle = cell.a + (cell.b - cell.a) / 2.0;

        let left = method.integrate_cell(&f, cell.a, middle)?;
        let right = method.integrate_cell(&f, middle, cell.b)?;

        cells.push(left);
        cells.push(right);
    }
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
            1e-12,
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
            10_000,
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
            10_000,
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
            10_000,
        )
        .unwrap();

        let exact = 10.0_f64.exp() - 1.0;

        println!("Result: {}", result);
        println!("Exact value: {}", exact);
        println!("Width: {:.17e}", result.wid());

        assert!(result.contains(exact));
    }
}

