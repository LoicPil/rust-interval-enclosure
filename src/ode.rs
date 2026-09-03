use crate::taylor::TaylorModel;
use inari::{Interval, interval};
use matplotlib::pyplot::subplots;

pub type OdeFunction = fn(Vec<TaylorModel>) -> Vec<TaylorModel>;

#[derive(Clone, Debug)]
pub struct BungerOptions {
    pub order: usize,
    pub h: f64,
    pub epsilon: f64,
    pub delta: f64,
    pub max_inflation_iterations: usize,
    pub blunt_tau: Option<f64>,
    pub preconditioning: bool,
}

impl Default for BungerOptions {
    fn default() -> Self {
        Self {
            order: 10,
            h: 0.01,
            epsilon: 0.01,
            delta: 1e-12,
            max_inflation_iterations: 50,
            blunt_tau: None,
            preconditioning: false,
        }
    }
}

/// K(y) = y0 + ∫_{t0}^{t} f(y) dt, evaluated in Taylor model arithmetic.
pub fn picard_step(
    initial: &[TaylorModel],
    current: &[TaylorModel],
    f: OdeFunction,
) -> Vec<TaylorModel> {
    assert_eq!(initial.len(), current.len());

    let derivatives = f(current.to_vec());
    assert_eq!(derivatives.len(), current.len());

    derivatives
        .into_iter()
        .zip(initial.iter())
        .map(|(fi, yi)| yi.clone() + fi.integrate_time())
        .collect()
}

/// p^(0) = q,  p^(i+1) = K(p^(i)); after `iterations` steps the polynomial
/// part is the order-`iterations` Taylor polynomial of the flow.
pub fn picard_iteration(initial: Vec<TaylorModel>, f: OdeFunction) -> Vec<TaylorModel> {
    let iterations = initial[0].order;
    let mut current = initial.clone();
    for _ in 0..iterations {
        current = picard_step(&initial, &current, f);
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
pub fn remainder_subset(image: &TaylorModel, enclosure: &TaylorModel) -> bool {
    let drift = (image.polynomial.clone() - enclosure.polynomial.clone()).evaluate(&image.domain);
    let effective = image.remainder + drift;
    interval_subset(effective, enclosure.remainder)
}

pub fn vector_subset(image: &[TaylorModel], enclosure: &[TaylorModel]) -> bool {
    assert_eq!(image.len(), enclosure.len());
    image
        .iter()
        .zip(enclosure.iter())
        .all(|(im, en)| remainder_subset(im, en))
}

/// Find a remainder E with K(p+E) ⊆ p+E, given the fixed polynomial `p`.
pub fn verify_remainder(
    polynomial: &[TaylorModel],
    initial: &[TaylorModel],
    f: OdeFunction,
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

        let image = picard_step(initial, &enclosure, f);

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
    options: &BungerOptions,
) -> Option<Vec<TaylorModel>> {
    crate::taylor::clear_caches();

    // Désactiver la sparsification pendant la Picard iteration (seuil infini)
    let old_threshold = crate::taylor::get_sparsity_threshold();
    crate::taylor::set_sparsity_threshold(f64::INFINITY);

    let polynomial = picard_iteration(initial.clone(), f);

    // Restaurer le seuil pour l'inflation
    crate::taylor::set_sparsity_threshold(old_threshold);

    verify_remainder(
        &polynomial,
        &initial,
        f,
        options.epsilon,
        options.delta,
        options.max_inflation_iterations,
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
    n_steps: usize,
    options: &BungerOptions,
) -> Result<Vec<(f64, Vec<Interval>)>, String> {
    let mut current = initial;
    let mut result = Vec::with_capacity(n_steps + 1);
    result.push((t0, ranges(&current)));

    let mut t = t0;

    for step in 0..n_steps {
        let enclosure = bunger_step(current, f, options)
            .ok_or_else(|| format!("verification failed at step {} (t = {})", step, t))?;

        t += options.h;
        result.push((t, ranges(&enclosure)));

        current = endpoint(&enclosure, options.h)
            .into_iter()
            .map(|tm| tm.extend_with_time(options.h))
            .collect();
    }

    Ok(result)
}

/// Thin wrapper around solve_bunger doing the initial dimension-lift
/// (Bünger Step 1 -> Step 2): the caller builds a time-independent q+J
/// in dimension n, this lifts it to n+1 with a fresh time domain [0, h]
/// before handing it to the main solver.
pub fn solve_ode(
    initial: Vec<TaylorModel>,
    f: OdeFunction,
    t0: f64,
    n_steps: usize,
    options: &BungerOptions,
) -> Result<Vec<(f64, Vec<Interval>)>, String> {
    let lifted: Vec<TaylorModel> = initial
        .into_iter()
        .map(|tm| tm.extend_with_time(options.h))
        .collect();

    solve_bunger(lifted, f, t0, n_steps, options)
}

pub mod linalg {
    pub fn invert(a: &[Vec<f64>], floor: f64) -> Vec<Vec<f64>> {
        let n = a.len();
        let mut m: Vec<Vec<f64>> = a.to_vec();
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            inv[i][i] = 1.0;
        }

        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())
                .unwrap();
            m.swap(col, pivot);
            inv.swap(col, pivot);

            let d = if m[col][col].abs() < floor {
                floor.copysign(if m[col][col] == 0.0 { 1.0 } else { m[col][col] })
            } else {
                m[col][col]
            };

            for k in 0..n {
                m[col][k] /= d;
                inv[col][k] /= d;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = m[row][col];
                for k in 0..n {
                    m[row][k] -= factor * m[col][k];
                    inv[row][k] -= factor * inv[col][k];
                }
            }
        }
        inv
    }

    /// Smallest pivot seen during elimination — cheap proxy for near-singularity.
    pub fn smallest_pivot_magnitude(a: &[Vec<f64>]) -> f64 {
        let n = a.len();
        let mut m: Vec<Vec<f64>> = a.to_vec();
        let mut min_pivot = f64::INFINITY;
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())
                .unwrap();
            m.swap(col, pivot);
            let d = m[col][col];
            min_pivot = min_pivot.min(d.abs());
            if d.abs() < 1e-300 {
                break;
            }
            for row in (col + 1)..n {
                let factor = m[row][col] / d;
                for k in col..n {
                    m[row][k] -= factor * m[col][k];
                }
            }
        }
        min_pivot
    }

    /// Permuted QR factorization A·P = Q·R̃, R := R̃·Pᵀ (Lohner's method, §3.3).
    pub fn qr_pivoted(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = a.len();
        let cols: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| a[i][j]).collect()).collect();
        let norms: Vec<f64> = cols
            .iter()
            .map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();

        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&i, &j| norms[j].partial_cmp(&norms[i]).unwrap());

        let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut r_tilde = vec![vec![0.0; n]; n];

        for k in 0..n {
            let mut v = cols[perm[k]].clone();
            for j in 0..k {
                let r_jk: f64 = (0..n).map(|i| q_cols[j][i] * v[i]).sum();
                r_tilde[j][k] = r_jk;
                for i in 0..n {
                    v[i] -= r_jk * q_cols[j][i];
                }
            }
            let norm_v = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            r_tilde[k][k] = norm_v;
            if norm_v > 1e-14 {
                q_cols.push(v.iter().map(|x| x / norm_v).collect());
            } else {
                let mut e = vec![0.0; n];
                e[k] = 1.0;
                for j in 0..k {
                    let d: f64 = (0..n).map(|i| q_cols[j][i] * e[i]).sum();
                    for i in 0..n {
                        e[i] -= d * q_cols[j][i];
                    }
                }
                let ne = e.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
                q_cols.push(e.iter().map(|x| x / ne).collect());
            }
        }

        let mut q = vec![vec![0.0; n]; n];
        for i in 0..n {
            for k in 0..n {
                q[i][k] = q_cols[k][i];
            }
        }

        let mut r = vec![vec![0.0; n]; n];
        for k in 0..n {
            let orig_col = perm[k];
            for row in 0..n {
                r[row][orig_col] = r_tilde[row][k];
            }
        }

        (q, r)
    }
}

/// Splits space-only TMs p(x) into (A, b(x), c): linear coefficients,
/// nonlinear remainder polynomial, and constant term.
fn linear_decompose(p: &[TaylorModel]) -> (Vec<Vec<f64>>, Vec<TaylorModel>, Vec<f64>) {
    let n = p.len();
    let dim = p[0].polynomial.dimension;
    let order = p[0].order;
    let domain = p[0].domain.clone();

    let mut a = vec![vec![0.0; n]; n];
    let mut c = vec![0.0; n];
    let mut b = Vec::with_capacity(n);

    for i in 0..n {
        let mut b_poly = crate::taylor::Polynomial::new(dim, order);
        for (exponents, coeff) in p[i].polynomial.terms() {
            let deg: usize = exponents.iter().sum();
            if deg == 0 {
                c[i] = coeff;
            } else if deg == 1 {
                let j = exponents.iter().position(|&e| e == 1).unwrap();
                a[i][j] = coeff;
            } else {
                b_poly.set(&exponents, coeff);
            }
        }
        b.push(TaylorModel {
            polynomial: b_poly,
            remainder: interval!(0.0, 0.0).unwrap(),
            domain: domain.clone(),
            order,
        });
    }
    (a, b, c)
}

pub struct Preconditioned {
    pub q_l: Vec<TaylorModel>,
    pub q_r: Vec<TaylorModel>,
}

/// Parallelepiped preconditioning (Q:=A, R:=I) with blunting guarding the inverse.
pub fn precondition(
    p_star_l: &[TaylorModel],
    q_r: &[TaylorModel],
    blunt_tau: Option<f64>,
) -> Preconditioned {
    let n = p_star_l.len();
    let dim = p_star_l[0].polynomial.dimension;
    let order = p_star_l[0].order;
    let domain = p_star_l[0].domain.clone();

    let (a, b, c) = linear_decompose(p_star_l);

    let a_used = match blunt_tau {
        Some(tau) => {
            let min_pivot = linalg::smallest_pivot_magnitude(&a);
            if min_pivot < tau {
                let mut a2 = a.clone();
                for i in 0..n {
                    a2[i][i] += tau;
                }
                a2
            } else {
                a
            }
        }
        None => a,
    };

    let (q_mat, r_mat) = linalg::qr_pivoted(&a_used);
    let q_t: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| q_mat[j][i]).collect())
        .collect();

    let i_l_star: Vec<Interval> = p_star_l.iter().map(|tm| tm.remainder).collect();
    let mut composed: Vec<TaylorModel> = b.iter().map(|bi| bi.polynomial.compose_tm(q_r)).collect();
    for (ci, &il) in composed.iter_mut().zip(i_l_star.iter()) {
        ci.remainder = ci.remainder + il;
    }

    let u_plus_f: Vec<TaylorModel> = (0..n)
        .map(|i| {
            let mut acc = TaylorModel::constant(0.0, dim, order, domain.clone());

            for j in 0..n {
                if r_mat[i][j] != 0.0 {
                    let mut poly = q_r[j].polynomial.clone();
                    for (exponents, coeff) in q_r[j].polynomial.terms() {
                        poly.set(&exponents, coeff * r_mat[i][j]);
                    }
                    let scaled = TaylorModel {
                        polynomial: poly,
                        remainder: q_r[j].remainder * scalar(r_mat[i][j]),
                        domain: domain.clone(),
                        order,
                    };
                    acc = acc + scaled;
                }
            }

            for j in 0..n {
                if q_t[i][j] != 0.0 {
                    let mut poly = composed[j].polynomial.clone();
                    for (exponents, coeff) in composed[j].polynomial.terms() {
                        poly.set(&exponents, coeff * q_t[i][j]);
                    }
                    let scaled = TaylorModel {
                        polynomial: poly,
                        remainder: composed[j].remainder * scalar(q_t[i][j]),
                        domain: domain.clone(),
                        order,
                    };
                    acc = acc + scaled;
                }
            }
            acc
        })
        .collect();

    eprintln!(
        "m = {:?}",
        u_plus_f
            .iter()
            .map(|tm| {
                let r = tm.range();
                r.inf().abs().max(r.sup().abs())
            })
            .collect::<Vec<_>>()
    );
    let s: Vec<f64> = u_plus_f
        .iter()
        .map(|tm| {
            let r = tm.range();
            let m = r.inf().abs().max(r.sup().abs());
            if m > 0.0 { 1.0 / m } else { 1.0 }
        })
        .collect();
    eprintln!("s = {:?}", s);
    let q_r_new: Vec<TaylorModel> = u_plus_f
        .iter()
        .zip(s.iter())
        .map(|(tm, &si)| {
            let mut poly = tm.polynomial.clone();
            for (exponents, coeff) in tm.polynomial.terms() {
                poly.set(&exponents, coeff * si);
            }
            TaylorModel {
                polynomial: poly,
                remainder: tm.remainder * scalar(si),
                domain: domain.clone(),
                order,
            }
        })
        .collect();

    let q_l_new: Vec<TaylorModel> = (0..n)
        .map(|i| {
            let mut poly = crate::taylor::Polynomial::constant(dim, order, c[i]);
            for j in 0..n {
                let coeff = q_mat[i][j] / s[j];
                if coeff != 0.0 {
                    poly = poly + crate::taylor::Polynomial::variable(dim, order, j, coeff);
                }
            }
            TaylorModel {
                polynomial: poly,
                remainder: interval!(0.0, 0.0).unwrap(),
                domain: domain.clone(),
                order,
            }
        })
        .collect();

    Preconditioned {
        q_l: q_l_new,
        q_r: q_r_new,
    }
}

/// Preconditioned solver.
pub fn solve_bunger_preconditioned(
    initial: Vec<TaylorModel>,
    f: OdeFunction,
    t0: f64,
    n_steps: usize,
    options: &BungerOptions,
) -> Result<Vec<(f64, Vec<Interval>)>, String> {
    let blunt_tau = options.blunt_tau;
    let n = initial.len();
    let dim = initial[0].polynomial.dimension;
    let order = initial[0].order;
    let domain = initial[0].domain.clone();

    let mut q_l = initial;
    let mut q_r: Vec<TaylorModel> = (0..n)
        .map(|i| TaylorModel::variable(i, 1.0, dim, order, domain.clone()))
        .collect();

    let mut result = Vec::with_capacity(n_steps + 1);
    result.push((t0, ranges(&crate::taylor::compose(&q_l, &q_r))));

    let mut t = t0;
    for step in 0..n_steps {
        let lifted: Vec<TaylorModel> = q_l
            .into_iter()
            .map(|tm| tm.extend_with_time(options.h))
            .collect();

        let enclosure = bunger_step(lifted, f, options)
            .ok_or_else(|| format!("verification failed at step {} (t = {})", step, t))?;

        t += options.h;
        if step % 100 == 0 {
            eprintln!("step {} / {}, t = {:.3}", step, n_steps, t);
        }
        let p_star_l = endpoint(&enclosure, options.h);

        let Preconditioned {
            q_l: new_q_l,
            q_r: new_q_r,
        } = precondition(&p_star_l, &q_r, blunt_tau);

        let after = crate::taylor::compose(&new_q_l, &new_q_r);
        result.push((t, ranges(&after)));

        q_l = new_q_l;
        q_r = new_q_r;
    }

    Ok(result)
}

/// Single entry point.
pub fn solve(
    initial: Vec<TaylorModel>,
    f: OdeFunction,
    t0: f64,
    tf: f64,
    options: &BungerOptions,
) -> Result<Vec<(f64, Vec<Interval>)>, String> {
    assert_eq!(
        initial[0].order, options.order,
        "TaylorModel order ({}) must match BungerOptions.order ({})",
        initial[0].order, options.order
    );

    let n_steps = ((tf - t0) / options.h).round() as usize;

    if options.preconditioning {
        solve_bunger_preconditioned(initial, f, t0, n_steps, options)
    } else {
        solve_ode(initial, f, t0, n_steps, options)
    }
}

// ------------------------------------------------------------
// Plotting functions (unchanged)
// ------------------------------------------------------------
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

fn scalar(x: f64) -> Interval {
    interval!(x, x).unwrap()
}
