#![allow(dead_code)]

use inari::{Interval, const_interval, interval};

use interval_enclosure::integration::{
    GaussLegendreRule, GaussianRule, gaussian_certified, midpoint_certified, simpson_certified,
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

        let width_mid = result_mid.wid();
        let width_trap = result_trap.wid();
        let width_simp = result_simp.wid();
        let width_gauss = result_gauss.wid();

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

            widths.push(result.wid());
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

/// Assuming W(h) ≈ C h^p, we have ln(W) ≈ ln(C) + p ln(h).
/// The convergence rate p is therefore estimated as the slope
/// of the least-squares linear regression of ln(W) against ln(h).
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

    let n_values: Vec<u32> = vec![2, 5, 10];

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
            widths.push(result.wid());
        }

        let rate = convergence_rate(&hs, &widths);

        println!("m = {:2} | experimental rate = {:.6}", order, rate);
    }

    Ok(())
}

fn print_gauss_legendre(order: usize) {
    let rule = GaussLegendreRule::new(order);

    println!();
    println!("===== Gauss-Legendre rule m = {} =====", order);

    for i in 0..order {
        println!(
            "i = {:2} | node = {:.16e} | weight = {:.16e}",
            i,
            rule.nodes()[i],
            rule.weights()[i]
        );
    }
}

fn check_gauss_legendre(order: usize) {
    let rule = GaussLegendreRule::new(order);

    let (reference_nodes, reference_weights): (&[f64], &[f64]) = match order {
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
                -0.3737060887154195,
                -0.2277858511416451,
                -0.0765265211334973,
                0.0765265211334973,
                0.2277858511416451,
                0.3737060887154195,
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
                0.0406014298003869,
                0.0626720483341091,
                0.0832767415767047,
                0.1019301198172404,
                0.1181945319615184,
                0.1316886384491766,
                0.1420961091852,
                0.1491729864726037,
                0.1527533871307259,
                0.1527533871307259,
                0.1491729864726037,
                0.1420961091852,
                0.1316886384491766,
                0.1181945319615184,
                0.1019301198172404,
                0.0832767415767047,
                0.0626720483341091,
                0.0406014297993869,
                0.0176140071391521,
            ],
        ),

        _ => {
            println!("Reference values are available only for m = 5, 10, 20.");
            return;
        }
    };

    println!();
    println!("===== Gauss-Legendre validation m = {} =====", order);

    let mut max_node_error: f64 = 0.0;
    let mut max_weight_error: f64 = 0.0;

    for i in 0..order {
        let node_error = (rule.nodes()[i] - reference_nodes[i]).abs();

        let weight_error = (rule.weights()[i] - reference_weights[i]).abs();

        max_node_error = max_node_error.max(node_error);
        max_weight_error = max_weight_error.max(weight_error);

        println!(
            "i = {:2} | node error = {:.3e} | weight error = {:.3e}",
            i, node_error, weight_error
        );
    }

    println!();
    println!("maximum node error   = {:.3e}", max_node_error);

    println!("maximum weight error = {:.3e}", max_weight_error);
}

fn run_gauss_width_vs_h() -> Result<(), Box<dyn std::error::Error>> {
    let orders: Vec<usize> = (1..=10).collect();

    let n_values: Vec<u32> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256];

    let f = |x: Interval| -> Interval { (PI * x).sin() };

    println!();
    println!("===== Gauss-Legendre interval width vs h =====");
    println!();

    let mut all_hs: Vec<Vec<f64>> = Vec::new();
    let mut all_widths: Vec<Vec<f64>> = Vec::new();

    for &order in &orders {
        let rule = GaussLegendreRule::new(order);

        let mut hs = Vec::new();
        let mut widths = Vec::new();

        for &n in &n_values {
            let f_2m = |x: Interval| -> Interval {
                let coeff = PI.powi((2 * order) as i32);

                if order % 2 == 0 {
                    coeff * (PI * x).sin()
                } else {
                    -coeff * (PI * x).sin()
                }
            };

            let result = gaussian_certified(&rule, f, f_2m, 0.0, 1.0, n)?;

            let h = 1.0 / n as f64;
            let width = result.wid();

            hs.push(h);
            widths.push(width);
        }

        all_hs.push(hs);
        all_widths.push(widths);
    }

    let (fig, [[mut ax]]) = subplots()?;

    let formats = ["o-", "s-", "^-", "D-", "v-", "<-", ">-", "p-", "h-", "*-"];

    for (i, &order) in orders.iter().enumerate() {
        ax.xy(&all_hs[i], &all_widths[i])
            .fmt(formats[i])
            .markersize(7.0)
            .label(&format!("m = {}", order))
            .plot();
    }

    ax.set_xlabel("$h$");
    ax.set_ylabel("interval width");

    ax.set_xscale("log");
    ax.set_yscale("log");

    ax.grid();
    ax.minorticks_on();
    ax.legend(std::iter::empty());

    fig.save().to_file("gauss_width_vs_h.pdf")?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // run_convergence_graph()?;
    //
    // run_midpoint_demo()?;
    //
    // run_gauss_demo()?;
    //
    // run_quadrature_convergence()?;
    // run_gauss_orders()?;

    run_gauss_convergence_rates()?;

    print_gauss_legendre(5);
    print_gauss_legendre(10);

    check_gauss_legendre(5);
    check_gauss_legendre(10);

    run_gauss_width_vs_h()?;

    Ok(())
}
