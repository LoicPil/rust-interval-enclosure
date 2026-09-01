//! Naive interval Euler method for scalar IVPs: u' = f(t,u), u(t0) = u0.
//! U_{n+1} = U_n + h * f(t_n, U_n), all in interval arithmetic.
//! No a priori enclosure, no Taylor remainder -- for visualizing
//! how the interval width grows/behaves along the integration.

use inari::{Interval, interval};

pub fn euler_step<F>(field: &F, t_n: f64, h: f64, u_n: Interval) -> Interval
where
    F: Fn(Interval, Interval) -> Interval,
{
    let t_point = interval!(t_n, t_n).unwrap();
    let step = interval!(h, h).unwrap();
    u_n + step * field(t_point, u_n)
}

pub fn euler_integrate<F>(
    field: &F,
    t0: f64,
    u0: Interval,
    h: f64,
    n_steps: usize,
) -> Vec<(f64, Interval)>
where
    F: Fn(Interval, Interval) -> Interval,
{
    let mut t = t0;
    let mut u = u0;
    let mut trace = vec![(t, u)];

    for _ in 0..n_steps {
        u = euler_step(field, t, h, u);
        t += h;
        trace.push((t, u));
    }
    trace
}

pub struct RKTableau<const S: usize> {
    pub c: [f64; S],
    pub a: [[f64; S]; S], // strictly lower triangular for explicit RK
    pub b: [f64; S],
}

/// Classical RK4 (Iserles-style tableau), the most common choice.
pub const RK4: RKTableau<4> = RKTableau {
    c: [0.0, 0.5, 0.5, 1.0],
    a: [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    b: [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
};

/// Heun's method (order 2, 2 stages) -- the 2-stage tableau from Iserles (3.7)
/// with c2=1, a21=1, b1=b2=1/2.
pub const HEUN2: RKTableau<2> = RKTableau {
    c: [0.0, 1.0],
    a: [[0.0, 0.0], [1.0, 0.0]],
    b: [0.5, 0.5],
};

/// Midpoint method (order 2, 2 stages) -- c2=1/2, a21=1/2, b1=0, b2=1.
pub const MIDPOINT2: RKTableau<2> = RKTableau {
    c: [0.0, 0.5],
    a: [[0.0, 0.0], [0.5, 0.0]],
    b: [0.0, 1.0],
};

fn rk_step<F, const S: usize>(
    field: &F,
    tableau: &RKTableau<S>,
    t_n: f64,
    h: f64,
    u_n: Interval,
) -> Interval
where
    F: Fn(Interval, Interval) -> Interval,
{
    let mut k = [interval!(0.0, 0.0).unwrap(); S];
    let h_iv = interval!(h, h).unwrap();

    for j in 0..S {
        // ξ_i = y_n + h Σ(i) a_ji f(t_n +c_i*h, ξ_i)
        let mut weighted_sum = interval!(0.0, 0.0).unwrap();
        for i in 0..j {
            let a_ji = tableau.a[j][i];
            if a_ji != 0.0 {
                let a_ji_iv = interval!(a_ji, a_ji).unwrap();
                weighted_sum = weighted_sum + a_ji_iv * k[i];
            }
        }
        let stage_u = u_n + h_iv * weighted_sum;

        let t_stage = t_n + tableau.c[j] * h;
        let t_stage_iv = interval!(t_stage, t_stage).unwrap();
        k[j] = field(t_stage_iv, stage_u);
    }

    let mut weighted_sum = interval!(0.0, 0.0).unwrap();
    for j in 0..S {
        // u_n+1 = u_n+h Σ(j) b_j f(t_n+c_j*h, ξ_j)
        let b_j = tableau.b[j];
        if b_j != 0.0 {
            let b_j_iv = interval!(b_j, b_j).unwrap();
            weighted_sum = weighted_sum + b_j_iv * k[j];
        }
    }
    u_n + h_iv * weighted_sum
}

pub fn rk_integrate<F, const S: usize>(
    field: &F,
    tableau: &RKTableau<S>,
    t0: f64,
    u0: Interval,
    h: f64,
    n_steps: usize,
) -> Vec<(f64, Interval)>
where
    F: Fn(Interval, Interval) -> Interval,
{
    let mut t = t0;
    let mut u = u0;
    let mut trace = vec![(t, u)];

    for _ in 0..n_steps {
        u = rk_step(field, tableau, t, h, u);
        t += h;
        trace.push((t, u));
    }
    trace
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_grows_geometrically_for_linear_field() {
        let field = |_t: Interval, u: Interval| u;
        let u0 = interval!(0.9, 1.1).unwrap();
        let trace = euler_integrate(&field, 0.0, u0, 0.1, 2);

        let w0 = trace[0].1.sup() - trace[0].1.inf();
        let w2 = trace[2].1.sup() - trace[2].1.inf();

        // U_{n+1} = U_n * (1+h) here, so width follows w_n = w0 * (1+h)^n.
        let expected_w2 = w0 * 1.1_f64.powi(2);
        assert!((w2 - expected_w2).abs() < 1e-10);
    }
}
