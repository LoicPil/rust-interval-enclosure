#![allow(dead_code)]

use inari::{Interval, const_interval, interval};

use interval_enclosure::integration::{
    GaussLegendreRule, gaussian_certified, midpoint_certified, simpson_certified,
    trapezoidal_certified,
};

use interval_enclosure::{f, fprime, hausdorff_nested, range_enclosure, range_enclosure_order1};

use matplotlib::pyplot::subplots;

const PI: Interval = const_interval!(std::f64::consts::PI, std::f64::consts::PI);

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

fn run_midpoint_demo() -> Result<(), Box<dyn std::error::Error>> {
    let f = |x: Interval| -> Interval { (PI * x).sin() };

    let f_pp = |x: Interval| -> Interval { -PI.powi(2) * (PI * x).sin() };

    let integral = midpoint_certified(f, f_pp, 0.0, 1.0, 100)?;

    println!();
    println!("===== Midpoint demo =====");

    println!(
        "midpoint integral = [{}, {}]",
        integral.inf(),
        integral.sup()
    );

    Ok(())
}

fn run_gauss_demo() -> Result<(), Box<dyn std::error::Error>> {
    let pi = std::f64::consts::PI;

    let exact = 2.0 / pi;

    let order = 4usize;

    let subdivisions = 1;

    let rule = GaussLegendreRule::new(order);

    let f = |x: Interval| -> Interval { (PI * x).sin() };

    let f_2m = move |x: Interval| -> Interval {
        let coeff = interval!(pi.powi((2 * order) as i32), pi.powi((2 * order) as i32)).unwrap();

        if order % 2 == 0 {
            coeff * (PI * x).sin()
        } else {
            -coeff * (PI * x).sin()
        }
    };

    let result = gaussian_certified(&rule, f, f_2m, 0.0, 1.0, subdivisions)?;

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

    Ok(())
}

fn run_quadrature_convergence() -> Result<(), Box<dyn std::error::Error>> {
    let pi = std::f64::consts::PI;

    let exact = 2.0 / pi;

    let f = |x: Interval| -> Interval { (PI * x).sin() };

    let f_pp = |x: Interval| -> Interval { -PI.powi(2) * (PI * x).sin() };

    let f_4 = |x: Interval| -> Interval { PI.powi(4) * (PI * x).sin() };

    let gauss_order = 3usize;

    let gauss_rule = GaussLegendreRule::new(gauss_order);

    let f_2m = move |x: Interval| -> Interval {
        let coeff = PI.powi((2 * gauss_order) as i32);

        if gauss_order % 2 == 0 {
            coeff * (PI * x).sin()
        } else {
            -coeff * (PI * x).sin()
        }
    };

    let n_values: Vec<u32> = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

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
        let result_mid = midpoint_certified(f, f_pp, 0.0, 1.0, n)?;

        let result_trap = trapezoidal_certified(f, f_pp, 0.0, 1.0, n)?;

        let result_simp = simpson_certified(f, f_4, 0.0, 1.0, n)?;

        let result_gauss = gaussian_certified(&gauss_rule, f, f_2m, 0.0, 1.0, n)?;

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

fn run_gauss_orders() -> Result<(), Box<dyn std::error::Error>> {
    let pi = std::f64::consts::PI;

    let orders: Vec<usize> = (1..=10).collect();

    let subdivisions_values: Vec<u32> = vec![1, 2, 5, 10];

    let mut all_widths: Vec<Vec<f64>> = Vec::new();

    let f = |x: Interval| -> Interval { (PI * x).sin() };

    for &subdivisions in &subdivisions_values {
        let mut widths = Vec::new();

        for &order in &orders {
            let rule = GaussLegendreRule::new(order);

            let f_2m = move |x: Interval| -> Interval {
                let coeff =
                    interval!(pi.powi((2 * order) as i32), pi.powi((2 * order) as i32)).unwrap();

                if order % 2 == 0 {
                    coeff * (PI * x).sin()
                } else {
                    -coeff * (PI * x).sin()
                }
            };

            let result = gaussian_certified(&rule, f, f_2m, 0.0, 1.0, subdivisions)?;

            widths.push(result.sup() - result.inf());
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
// Assuming W(h) ≈ C h^p, we have ln(W) ≈ ln(C) + p ln(h).
// The convergence rate p is therefore estimated as the slope
// of the least-squares linear regression of ln(W) against ln(h).
fn convergence_rate(hs: &[f64], widths: &[f64]) -> f64 {
    let n = hs.len() as f64;

    let xs: Vec<f64> = hs.iter().map(|&h| h.ln()).collect();
    let ys: Vec<f64> = widths.iter().map(|&w| w.ln()).collect();

    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();

    let sum_x2: f64 = xs.iter().map(|x| x * x).sum();

    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();

    (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
}
fn run_gauss_convergence_rates() -> Result<(), Box<dyn std::error::Error>> {
    let orders: Vec<usize> = (1..=7).collect();

    let n_values: Vec<u32> = vec![2, 4, 8, 16, 32];

    let f = |x: Interval| -> Interval { (PI * x).sin() };

    println!();
    println!("===== Gauss-Legendre convergence rates =====");
    println!();

    for &order in &orders {
        let rule = GaussLegendreRule::new(order);

        let mut hs = Vec::new();
        let mut widths = Vec::new();

        for &n in &n_values {
            let f_2m = move |x: Interval| -> Interval {
                let coeff = PI.powi((2 * order) as i32);

                if order % 2 == 0 {
                    coeff * (PI * x).sin()
                } else {
                    -coeff * (PI * x).sin()
                }
            };

            let result = gaussian_certified(&rule, f, f_2m, 0.0, 1.0, n)?;

            hs.push(1.0 / n as f64);
            widths.push(result.sup() - result.inf());
        }

        let rate = convergence_rate(&hs, &widths);

        println!("m = {:2} | experimental rate = {:.6}", order, rate);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // run_convergence_graph()?;
    //
    // run_midpoint_demo()?;

    run_gauss_demo()?;

    run_quadrature_convergence()?;
    run_gauss_orders()?;
    run_gauss_convergence_rates()?;
    Ok(())
}
