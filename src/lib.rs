use core::f64;
use inari::{Interval, interval};

pub mod integration;
pub mod matrix;
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
