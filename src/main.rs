use inari::{Interval, interval};
use matplotlib as plt;
use plt::pyplot::subplots;

fn f(x: Interval) -> Interval {
    x.sin() - x * x.cos()
}

fn fprime(x: Interval) -> Interval {
    x * x.sin()
}

fn range_enclosure<F>(f: F, x: Interval, eps: f64, max_depth: u32) -> Interval
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

fn range_enclosure_order1<F, G>(f: F, fprime: G, x: Interval, eps: f64, max_depth: u32) -> Interval
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

fn hausdorff_nested(inner: Interval, outer: Interval) -> f64 {
    let d_lo = inner.inf() - outer.inf();
    let d_hi = outer.sup() - inner.sup();
    d_lo.max(d_hi)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = interval!(0., 4.)?;
    let reference = interval!(0., std::f64::consts::PI)?; // exact range
    println!("reference = {}", reference);

    let eps_values: Vec<f64> = vec![1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5];
    let mut hs0 = Vec::new();
    let mut d_hs0 = Vec::new();
    let mut hs1 = Vec::new();
    let mut d_hs1 = Vec::new();

    for &eps in &eps_values {
        let enc0 = range_enclosure(f, x, eps, 50);
        let d0 = hausdorff_nested(reference, enc0);
        println!("[order 0] h={:>12.6e}  d_H={:>12.6e}", eps, d0);
        if d0 > 0.0 {
            hs0.push(eps);
            d_hs0.push(d0);
        }

        let enc1 = range_enclosure_order1(f, fprime, x, eps, 60);
        let d1 = hausdorff_nested(reference, enc1);
        println!("[order 1] h={:>12.6e}  d_H={:>12.6e}", eps, d1);
        if d1 > 0.0 {
            hs1.push(eps);
            d_hs1.push(d1);
        }
    }

    let (fig, [[mut ax]]) = subplots()?;
    ax.xy(&hs0, &d_hs0)
        .fmt("o-")
        .markersize(8.0)
        .color([1.0, 0.0, 0.0])
        .plot();
    ax.xy(&hs1, &d_hs1)
        .fmt("o-")
        .markersize(8.0)
        .color([0.0, 0.0, 1.0])
        .plot();
    ax.set_xlabel("$h$");
    ax.set_ylabel("$d_H$");
    ax.set_yscale("log");
    ax.set_xlim(hs0[0], hs0[hs0.len() - 1]);
    ax.grid();
    ax.minorticks_on();
    fig.save().to_file("convergence.pdf")?;
    Ok(())
}
