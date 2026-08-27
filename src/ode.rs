use crate::taylor::TaylorModel;
use inari::{Interval, interval};
use matplotlib::pyplot::subplots;

pub type OdeFunction = fn(Vec<TaylorModel>) -> Vec<TaylorModel>;

/// K(y) = y0 + ∫_{t0}^{t} f(y) dt, evaluated in Taylor model arithmetic.
pub fn picard_step(
    initial: &[TaylorModel],
    current: &[TaylorModel],
    f: OdeFunction,
    t0: f64,
) -> Vec<TaylorModel> {
    assert_eq!(initial.len(), current.len());

    let derivatives = f(current.to_vec());
    assert_eq!(derivatives.len(), current.len());

    derivatives
        .into_iter()
        .zip(initial.iter())
        .map(|(fi, yi)| yi.clone() + fi.integrate_time(t0))
        .collect()
}

/// p^(0) = q,  p^(i+1) = K(p^(i)); after `iterations` steps the polynomial
/// part is the order-`iterations` Taylor polynomial of the flow.
pub fn picard_iteration(
    initial: Vec<TaylorModel>,
    f: OdeFunction,
    t0: f64,
    iterations: usize,
) -> Vec<TaylorModel> {
    let mut current = initial.clone();
    for _ in 0..iterations {
        current = picard_step(&initial, &current, f, t0);
    }
    current
}

/// E_(i+1) = [1-eps, 1+eps] * E'_i + [-delta, delta].
pub fn inflate_remainder(remainder: Interval, epsilon: f64, delta: f64) -> Interval {
    let factor = interval!(1.0 - epsilon, 1.0 + epsilon).expect("invalid inflation factor");
    let delta_interval = interval!(-delta, delta).expect("invalid delta");
    factor * remainder + delta_interval
}

pub fn with_remainder(tm: &TaylorModel, remainder: Interval) -> TaylorModel {
    TaylorModel {
        polynomial: tm.polynomial.clone(),
        remainder,
        domain: tm.domain.clone(),
        order: tm.order,
    }
}

pub fn interval_subset(a: Interval, b: Interval) -> bool {
    b.inf() <= a.inf() && a.sup() <= b.sup()
}

/// p+E ⊆ p+F for identical polynomial parts iff E ⊆ F.
pub fn remainder_subset(a: &TaylorModel, b: &TaylorModel) -> bool {
    interval_subset(a.remainder, b.remainder)
}

pub fn vector_subset(a: &[TaylorModel], b: &[TaylorModel]) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .all(|(ai, bi)| remainder_subset(ai, bi))
}

/// Find a remainder E with K(p+E) ⊆ p+E, given the fixed polynomial `p`.
pub fn verify_remainder(
    polynomial: &[TaylorModel],
    initial: &[TaylorModel],
    f: OdeFunction,
    t0: f64,
    epsilon: f64,
    delta: f64,
    max_iterations: usize,
) -> Option<Vec<TaylorModel>> {
    let dimension = polynomial.len();
    let mut remainder = vec![interval!(0.0, 0.0).unwrap(); dimension];

    for _ in 0..max_iterations {
        let enclosure: Vec<TaylorModel> = polynomial
            .iter()
            .zip(remainder.iter())
            .map(|(p, &e)| with_remainder(p, e))
            .collect();

        let image = picard_step(initial, &enclosure, f, t0);

        if vector_subset(&image, &enclosure) {
            return Some(enclosure);
        }

        for i in 0..dimension {
            remainder[i] = inflate_remainder(image[i].remainder, epsilon, delta);
        }
    }

    None
}

/// One full integration step: Picard iteration for p, then ε-inflation for E.
pub fn bunger_step(
    initial: Vec<TaylorModel>,
    f: OdeFunction,
    t0: f64,
    picard_iterations: usize,
    epsilon: f64,
    delta: f64,
    max_inflation_iterations: usize,
) -> Option<Vec<TaylorModel>> {
    crate::taylor::clear_caches();
    let polynomial = picard_iteration(initial.clone(), f, t0, picard_iterations);

    verify_remainder(
        &polynomial,
        &initial,
        f,
        t0,
        epsilon,
        delta,
        max_inflation_iterations,
    )
}

pub fn ranges(state: &[TaylorModel]) -> Vec<Interval> {
    state.iter().map(|tm| tm.range()).collect()
}

/// q(x) := p(x, h),  J := E — the initial set for the next integration step.
/// Dimension drops by one (time is fixed); re-lifted to n+1 dims by the caller.
pub fn endpoint(state: &[TaylorModel], h: f64) -> Vec<TaylorModel> {
    state.iter().map(|tm| tm.substitute_time(h)).collect()
}

pub fn solve_bunger(
    initial: Vec<TaylorModel>,
    f: OdeFunction,
    t0: f64,
    h: f64,
    n_steps: usize,
    picard_iterations: usize,
    epsilon: f64,
    delta: f64,
    max_inflation_iterations: usize,
) -> Result<Vec<(f64, Vec<Interval>)>, String> {
    let mut current = initial;
    let mut result = Vec::with_capacity(n_steps + 1);
    result.push((t0, ranges(&current)));

    let mut t = t0;

    for step in 0..n_steps {
        let enclosure = bunger_step(
            current,
            f,
            t,
            picard_iterations,
            epsilon,
            delta,
            max_inflation_iterations,
        )
        .ok_or_else(|| format!("verification failed at step {} (t = {})", step, t))?;

        t += h;
        result.push((t, ranges(&enclosure)));

        current = endpoint(&enclosure, h)
            .into_iter()
            .map(|tm| tm.extend_with_time(h))
            .collect();
    }

    Ok(result)
}
/// Thin wrapper around solve_bunger with sensible defaults, and doing the
/// initial dimension-lift (Bünger Step 1 -> Step 2): the caller builds a
/// time-independent q+J in dimension n, this lifts it to n+1 with a fresh
/// time domain [0, h] before handing it to the main solver.
pub fn solve_ode(
    initial: Vec<TaylorModel>,
    f: OdeFunction,
    t0: f64,
    h: f64,
    n_steps: usize,
) -> Result<Vec<(f64, Vec<Interval>)>, String> {
    let order = initial[0].order;
    let lifted: Vec<TaylorModel> = initial
        .into_iter()
        .map(|tm| tm.extend_with_time(h))
        .collect();

    solve_bunger(lifted, f, t0, h, n_steps, order, 0.01, 1e-12, 50)
}

pub fn plot_component(
    result: &[(f64, Vec<Interval>)],
    component: usize,
    title: &str,
    ylabel: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(!result.is_empty());
    assert!(component < result[0].1.len());

    let ts: Vec<f64> = result.iter().map(|(t, _)| *t).collect();
    let lower: Vec<f64> = result.iter().map(|(_, v)| v[component].inf()).collect();
    let upper: Vec<f64> = result.iter().map(|(_, v)| v[component].sup()).collect();

    let (fig, [[mut ax]]) = subplots()?;
    ax.xy(&ts, &lower).fmt("-").label("lower").plot();
    ax.xy(&ts, &upper).fmt("-").label("upper").plot();
    ax.set_title(title);
    ax.set_xlabel("t");
    ax.set_ylabel(ylabel);
    ax.grid();
    ax.legend(std::iter::empty());
    fig.save().to_file(filename)?;
    Ok(())
}

pub fn plot_solution(
    result: &[(f64, Vec<Interval>)],
    title: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(!result.is_empty());
    let dimension = result[0].1.len();
    let ts: Vec<f64> = result.iter().map(|(t, _)| *t).collect();

    let (fig, [[mut ax]]) = subplots()?;
    for component in 0..dimension {
        let lower: Vec<f64> = result.iter().map(|(_, v)| v[component].inf()).collect();
        let upper: Vec<f64> = result.iter().map(|(_, v)| v[component].sup()).collect();
        ax.xy(&ts, &lower)
            .fmt("-")
            .label(&format!("y{} lower", component + 1))
            .plot();
        ax.xy(&ts, &upper)
            .fmt("-")
            .label(&format!("y{} upper", component + 1))
            .plot();
    }
    ax.set_title(title);
    ax.set_xlabel("t");
    ax.set_ylabel("y");
    ax.grid();
    ax.legend(std::iter::empty());
    fig.save().to_file(filename)?;
    Ok(())
}
