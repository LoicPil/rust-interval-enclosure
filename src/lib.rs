use core::f64;
use inari::{Interval, IntervalError, const_interval, interval};

pub mod adaptative_integration;
pub mod integration;
pub mod ivp;
pub mod matrix;
pub mod ode;
pub mod taylor;
pub mod timing;
pub mod wilkinson;

pub fn f(x: Interval) -> Interval {
    x.sin() - x * x.cos()
}

pub fn fprime(x: Interval) -> Interval {
    x * x.sin()
}

pub fn range_enclosure<F>(f: F, x: Interval, eps: f64, max_depth: u32) -> Interval
where
    F: Fn(Interval) -> Interval,
{
    fn recurse<F>(f: &F, y: Interval, eps: f64, depth: u32) -> Interval
    where
        F: Fn(Interval) -> Interval,
    {
        let fy = f(y);
        if y.wid() <= eps || depth == 0 {
            return fy;
        }
        let m = y.mid();
        let left = interval!(y.inf(), m).unwrap();
        let right = interval!(m, y.sup()).unwrap();
        recurse(f, left, eps, depth - 1).convex_hull(recurse(f, right, eps, depth - 1))
    }
    recurse(&f, x, eps, max_depth)
}

pub fn range_enclosure_order1<F, G>(
    f: F,
    fprime: G,
    x: Interval,
    eps: f64,
    max_depth: u32,
) -> Interval
where
    F: Fn(Interval) -> Interval,
    G: Fn(Interval) -> Interval,
{
    fn recurse<F, G>(f: &F, fprime: &G, y: Interval, eps: f64, depth: u32) -> Interval
    where
        F: Fn(Interval) -> Interval,
        G: Fn(Interval) -> Interval,
    {
        let m = y.mid();
        let c = interval!(m, m).unwrap();
        let centered = f(c) + fprime(y) * (y - c);

        if y.wid() <= eps || depth == 0 {
            return centered;
        }

        let left = interval!(y.inf(), m).unwrap();
        let right = interval!(m, y.sup()).unwrap();

        recurse(f, fprime, left, eps, depth - 1).convex_hull(recurse(
            f,
            fprime,
            right,
            eps,
            depth - 1,
        ))
    }

    recurse(&f, &fprime, x, eps, max_depth)
}

pub fn hausdorff_nested(inner: Interval, outer: Interval) -> f64 {
    let d_lo = (inner.inf() - outer.inf()).abs();
    let d_hi = (outer.sup() - inner.sup()).abs();
    d_lo.max(d_hi)
}

const ONE: Interval = const_interval!(1.0, 1.0);

pub fn factorial_interval(n: usize) -> Result<Interval, IntervalError> {
    let mut fact = ONE;
    for k in 1..=n {
        fact = fact * interval!(k as f64, k as f64)?;
    }
    Ok(fact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn compare_midpoint_methods() {
        let adaptive = crate::adaptative_integration::adaptive_integration(
            |x: Interval| x.sin(),
            crate::adaptative_integration::Midpoint {
                f_pp: |x: Interval| -x.sin(),
            },
            0.0,
            PI,
            1e-10,
            10_000,
        )
        .unwrap();

        let uniform = crate::integration::midpoint_certified(
            |x: Interval| x.sin(),
            |x: Interval| -x.sin(),
            0.0,
            PI,
            1_000,
        )
        .unwrap();

        let adaptive_width = adaptive.wid();
        let uniform_width = uniform.wid();

        println!("Enclosure adaptative : {}", adaptive);
        println!("Largeur adaptative   : {:.20e}", adaptive_width);

        println!("Enclosure uniforme   : {}", uniform);
        println!("Largeur uniforme     : {:.20e}", uniform_width);

        println!(
            "Différence de largeur : {:.20e}",
            uniform_width - adaptive_width
        );

        assert!(
            adaptive_width < uniform_width,
            "La méthode adaptative n'est pas meilleure : \
             largeur adaptative = {:.20e}, \
             largeur uniforme = {:.20e}",
            adaptive_width,
            uniform_width
        );
    }
}
