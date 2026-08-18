//! Wilkinson-style demo: product form vs expanded (Horner) form,
//! in f64 and in interval arithmetic.

use inari::{Interval, interval};

pub fn coeffs_from_roots(n: u32) -> Vec<f64> {
    let mut coeffs: Vec<f64> = vec![1.0];
    for i in 0..=n {
        let mut next = vec![0.0; coeffs.len() + 1];
        for (k, &c) in coeffs.iter().enumerate() {
            next[k + 1] += c; // contribution of x * g(x)
            next[k] += c * (-(i as f64)); // contribution of -i * g(x)
        }
        coeffs = next;
    }
    coeffs
}

pub fn eval_product_f64(x: f64, n: u32) -> f64 {
    (0..=n).map(|i| x - i as f64).product()
}

pub fn eval_horner_f64(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

pub fn eval_product_interval(x: Interval, n: u32) -> Interval {
    let mut acc = interval!(1.0, 1.0).unwrap();
    for i in 0..=n {
        let ci = interval!(i as f64, i as f64).unwrap();
        acc = acc * (x - ci);
    }
    acc
}

pub fn eval_horner_interval(coeffs: &[f64], x: Interval) -> Interval {
    let mut acc = interval!(0.0, 0.0).unwrap();
    for &c in coeffs.iter().rev() {
        let ci = interval!(c, c).unwrap();
        acc = acc * x + ci;
    }
    acc
}

/// Format an Interval's bounds. If `scientific` is true, use exponential
/// notation for each bound; otherwise fall back to Interval's own Display.
fn format_interval(iv: Interval, scientific: bool) -> String {
    if scientific {
        format!("[{:.6e}, {:.6e}]", iv.inf(), iv.sup())
    } else {
        format!("{}", iv)
    }
}

pub fn run_demo(x0: f64, x_iv: Interval, n_max: u32, scientific: bool) {
    println!(
        "{:>4} {:>16} {:>16} {:>12}   {:>26} {:>26}",
        "N", "product f64", "Horner f64", "rel.err f64", "product interval", "Horner interval"
    );
    for n in 2..=n_max {
        let coeffs = coeffs_from_roots(n);

        let p_direct = eval_product_f64(x0, n);
        let p_expanded = eval_horner_f64(&coeffs, x0);
        let rel_err = if p_direct != 0.0 {
            (p_direct - p_expanded).abs() / p_direct.abs()
        } else {
            (p_direct - p_expanded).abs()
        };

        let iv_direct = eval_product_interval(x_iv, n);
        let iv_expanded = eval_horner_interval(&coeffs, x_iv);

        let iv_direct_str = format_interval(iv_direct, scientific);
        let iv_expanded_str = format_interval(iv_expanded, scientific);

        println!(
            "{:>4} {:>16.6e} {:>16.6e} {:>12.3e}   {:>26} {:>26}",
            n, p_direct, p_expanded, rel_err, iv_direct_str, iv_expanded_str
        );
    }
}
