use inari::{Interval, interval};

use interval_enclosure::integration::{
    GaussLegendreRule, gauss_legendre_certified, midpoint_certified, simpson_certified,
    trapezoidal_certified,
};

use interval_enclosure::{f, fprime, hausdorff_nested, range_enclosure, range_enclosure_order1};

use matplotlib::pyplot::subplots;

/// Tests the convergence of the order-0 and order-1 range enclosures.
fn run_convergence_graph() -> Result<(), Box<dyn std::error::Error>> {
    let x: Interval = interval!(0., 4.)?;
    let reference = interval!(0., std::f64::consts::PI)?;

    println!("reference = {}", reference);

    let eps_values: Vec<f64> = vec![1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5];

    let mut hs0 = Vec::new();
    let mut d_hs0 = Vec::new();

    let mut hs1 = Vec::new();
    let mut d_hs1 = Vec::new();

    for &eps in &eps_values {
        let depth = (4.0 / eps).log2().ceil() as u32;
        let h = 4.0 / 2.0_f64.powi(depth as i32);

        let enc0 = range_enclosure(f, x, eps, 50);
        let d0 = hausdorff_nested(reference, enc0);

        println!(
            "[order 0] eps={:.1e}, h={:.6e}, E={}, d_H={:.6e}",
            eps, h, enc0, d0
        );

        hs0.push(h);
        d_hs0.push(d0);

        let enc1 = range_enclosure_order1(f, fprime, x, eps, 60);
        let d1 = hausdorff_nested(reference, enc1);

        println!(
            "[order 1] eps={:.1e}, h={:.6e}, E={}, d_H={:.6e}",
            eps, h, enc1, d1
        );

        hs1.push(h);
        d_hs1.push(d1);
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
    ax.set_ylim(1e-11, 1e-1);
    ax.set_xscale("log");

    ax.grid();
    ax.minorticks_on();

    fig.save().to_file("convergence.pdf")?;

    Ok(())
}

/// Tests the certified midpoint rule on a simple integral.
fn run_midpoint_demo() {
    let f_pp = |_x: Interval| -> Interval { interval!(-1.0, 1.0).unwrap() };

    let integral = midpoint_certified(
        |x: f64| x.sin(),
        f_pp,
        0.0,
        std::f64::consts::FRAC_PI_2,
        100,
    );

    println!();
    println!("===== Midpoint demo =====");

    println!(
        "midpoint integral = [{}, {}]",
        integral.inf(),
        integral.sup()
    );
}

/// Tests one Gauss-Legendre rule on
///
///     ∫_0^1 sin(πx) dx = 2/π.
///
/// The certified interval is checked against the exact value.
fn run_gauss_demo() {
    let pi = std::f64::consts::PI;
    let exact = 2.0 / pi;

    let order = 4usize;
    let subdivisions = 1;

    let rule = GaussLegendreRule::new(order);

    let pi_2m = pi.powi((2 * order) as i32);

    let f_2m = move |_x: Interval| -> Interval {
        if order % 2 == 0 {
            interval!(0.0, pi_2m).unwrap()
        } else {
            interval!(-pi_2m, 0.0).unwrap()
        }
    };

    let result =
        gauss_legendre_certified(&rule, |x: f64| (pi * x).sin(), f_2m, 0.0, 1.0, subdivisions);

    let width = result.sup() - result.inf();

    let contains_exact = result.inf() <= exact && exact <= result.sup();

    println!();
    println!("===== Gauss-Legendre demo =====");

    println!("order m        = {}", order);
    println!("subdivisions   = {}", subdivisions);

    println!(
        "result         = [{:.15e}, {:.15e}]",
        result.inf(),
        result.sup()
    );

    println!("width          = {:.6e}", width);
    println!("exact          = {:.15e}", exact);
    println!("contains exact = {}", contains_exact);
}

/// Compares the certified convergence of midpoint, trapezoidal,
/// Simpson, and a fixed-order Gauss-Legendre quadrature rule.
///
/// The test uses
///  ax.legend(std::iter::empty());
///     ∫_0^1 sin(πx) dx = 2/π.
///
/// The number of subintervals n is varied while the Gauss-Legendre
/// order is kept fixed.
fn run_quadrature_convergence() -> Result<(), Box<dyn std::error::Error>> {
    let pi = std::f64::consts::PI;
    let exact = 2.0 / pi;

    let pi2 = pi.powi(2);
    let f_pp = move |_x: Interval| -> Interval { interval!(-pi2, pi2).unwrap() };

    let pi4 = pi.powi(4);
    let f_4 = move |_x: Interval| -> Interval { interval!(-pi4, pi4).unwrap() };

    let gauss_order = 3usize;
    let gauss_rule = GaussLegendreRule::new(gauss_order);

    let pi_2m = pi.powi((2 * gauss_order) as i32);

    let f_2m = move |_x: Interval| -> Interval {
        if gauss_order % 2 == 0 {
            interval!(0.0, pi_2m).unwrap()
        } else {
            interval!(-pi_2m, 0.0).unwrap()
        }
    };

    let n_values: Vec<u32> = vec![2, 4, 8, 16, 32, 64, 128, 256];

    let mut ns = Vec::new();

    let mut errors_mid = Vec::new();
    let mut errors_trap = Vec::new();
    let mut errors_simp = Vec::new();
    let mut errors_gauss = Vec::new();

    println!();
    println!("===== Quadrature convergence =====");
    println!("Gauss-Legendre order = {}", gauss_order);
    println!();

    for &n in &n_values {
        let result_mid = midpoint_certified(|x: f64| (pi * x).sin(), f_pp, 0.0, 1.0, n);

        let result_trap = trapezoidal_certified(|x: f64| (pi * x).sin(), f_pp, 0.0, 1.0, n);

        let result_simp = simpson_certified(|x: f64| (pi * x).sin(), f_4, 0.0, 1.0, n);

        let result_gauss =
            gauss_legendre_certified(&gauss_rule, |x: f64| (pi * x).sin(), f_2m, 0.0, 1.0, n);

        let width_mid = result_mid.sup() - result_mid.inf();

        let width_trap = result_trap.sup() - result_trap.inf();

        let width_simp = result_simp.sup() - result_simp.inf();

        let width_gauss = result_gauss.sup() - result_gauss.inf();

        println!(
            "n={:4} | mid={:.3e} | trap={:.3e} | simp={:.3e} | gauss={:.3e}",
            n, width_mid, width_trap, width_simp, width_gauss,
        );

        ns.push(n as f64);

        errors_mid.push(width_mid);
        errors_trap.push(width_trap);
        errors_simp.push(width_simp);
        errors_gauss.push(width_gauss);
    }

    let (fig, [[mut ax]]) = subplots()?;

    ax.xy(&ns, &errors_mid)
        .fmt("o-")
        .markersize(8.0)
        .color([1.0, 0.0, 0.0])
        .label("Midpoint")
        .plot();

    ax.xy(&ns, &errors_trap)
        .fmt("s-")
        .markersize(8.0)
        .color([0.0, 0.0, 1.0])
        .label("Trapezoidal")
        .plot();

    ax.xy(&ns, &errors_simp)
        .fmt("^-")
        .markersize(8.0)
        .color([0.0, 0.6, 0.0])
        .label("Simpson")
        .plot();

    ax.xy(&ns, &errors_gauss)
        .fmt("D-")
        .markersize(8.0)
        .color([0.6, 0.0, 0.8])
        .label("Gauss-Legendre (m=3)")
        .plot();
    ax.set_xlabel("$n$");
    ax.set_ylabel("interval width");

    ax.set_yscale("log");
    ax.set_xscale("log");

    ax.grid();
    ax.minorticks_on();
    ax.legend(std::iter::empty());

    fig.save().to_file("quadrature_convergence.pdf")?;

    println!();
    println!("exact = {:.15e}", exact);

    Ok(())
}

/// Compares the certified convergence of Gauss-Legendre
/// for several orders.
///
/// The Gauss order m is varied while the number of subintervals n
/// is fixed for each curve.
fn run_gauss_orders() -> Result<(), Box<dyn std::error::Error>> {
    let pi = std::f64::consts::PI;

    let orders: Vec<usize> = (1..=10).collect();

    let subdivisions_values: Vec<u32> = vec![1, 2, 5, 10];

    let mut all_widths: Vec<Vec<f64>> = Vec::new();

    for &subdivisions in &subdivisions_values {
        let mut widths = Vec::new();

        for &order in &orders {
            let rule = GaussLegendreRule::new(order);

            let pi_2m = pi.powi((2 * order) as i32);

            let f_2m = move |_x: Interval| -> Interval {
                if order % 2 == 0 {
                    interval!(0.0, pi_2m).unwrap()
                } else {
                    interval!(-pi_2m, 0.0).unwrap()
                }
            };

            let result = gauss_legendre_certified(
                &rule,
                |x: f64| (pi * x).sin(),
                f_2m,
                0.0,
                1.0,
                subdivisions,
            );

            let width = result.sup() - result.inf();

            widths.push(width);
        }

        all_widths.push(widths);
    }

    println!();
    println!("===== Gauss-Legendre order convergence =====");
    println!();

    for (i, &subdivisions) in subdivisions_values.iter().enumerate() {
        println!("n = {}", subdivisions);

        for (j, &order) in orders.iter().enumerate() {
            println!("  m = {:2} | width = {:.6e}", order, all_widths[i][j]);
        }
    }

    let x: Vec<f64> = orders.iter().map(|&m| m as f64).collect();

    let (fig, [[mut ax]]) = subplots()?;

    let formats = ["o-", "s-", "^-", "D-"];

    for (i, &subdivisions) in subdivisions_values.iter().enumerate() {
        ax.xy(&x, &all_widths[i])
            .fmt(formats[i])
            .markersize(8.0)
            .label(&format!("n = {}", subdivisions))
            .plot();
    }

    ax.set_xlabel("Gauss-Legendre order $m$");
    ax.set_ylabel("interval width");

    ax.set_xscale("linear");
    ax.set_yscale("log");

    ax.grid();
    ax.minorticks_on();
    ax.legend(std::iter::empty());

    fig.save().to_file("gauss_order_convergence.pdf")?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Range enclosure convergence
    // run_convergence_graph()?;

    // Midpoint test
    // run_midpoint_demo();

    // Single Gauss-Legendre test
    // run_gauss_demo();

    // Comparison of midpoint, trapezoidal,
    // Simpson, and Gauss-Legendre
    run_quadrature_convergence()?;

    // Gauss-Legendre convergence for several orders
    run_gauss_orders()?;

    Ok(())
}
