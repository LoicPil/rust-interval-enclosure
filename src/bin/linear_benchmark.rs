use std::time::Instant;

use inari::interval;

use interval_enclosure::ode::{BungerOptions, OdeFunction, solve};
use interval_enclosure::taylor::TaylorModel;

const ORDER: usize = 12; // matches paper's m := 12
const T0: f64 = 0.0;
const TF: f64 = 100.0;
const H: f64 = 0.25; // matches paper's hmin := 0.25

const Y0_MID: f64 = 1.0;
const Y0_RAD: f64 = 0.001; // y0 := (1,1,1)^T ± 0.001

fn apply_matrix(b: &[[f64; 3]; 3], y: Vec<TaylorModel>) -> Vec<TaylorModel> {
    assert_eq!(y.len(), 3);
    let dim = y[0].polynomial.dimension;
    let order = y[0].order;
    let domain = y[0].domain.clone();

    (0..3)
        .map(|i| {
            let mut acc = TaylorModel::constant(0.0, dim, order, domain.clone());
            for j in 0..3 {
                if b[i][j] != 0.0 {
                    let coeff = TaylorModel::constant(b[i][j], dim, order, domain.clone());
                    acc = acc + coeff * y[j].clone();
                }
            }
            acc
        })
        .collect()
}

/// Case (1): contraction, eigenvalues -1/2, -3/4, 0.
fn linear1(y: Vec<TaylorModel>) -> Vec<TaylorModel> {
    apply_matrix(
        &[
            [-0.4375, 0.0625, -0.2651650429],
            [0.0625, -0.4375, -0.2651650429],
            [-0.2651650429, -0.2651650429, -0.375],
        ],
        y,
    )
}

/// Case (2): rotation of a hyperplane, eigenvalues ±i, 0.
fn linear2(y: Vec<TaylorModel>) -> Vec<TaylorModel> {
    apply_matrix(
        &[
            [0.0, -0.7071067810, -0.5],
            [0.7071067810, 0.0, 0.5],
            [0.5, -0.5, 0.0],
        ],
        y,
    )
}

/// Case (3): mixture of rotation and contraction, eigenvalues ±i, -1/2.
fn linear3(y: Vec<TaylorModel>) -> Vec<TaylorModel> {
    apply_matrix(
        &[
            [-0.125, -0.8321067810, -0.3232233048],
            [0.5821067810, -0.125, 0.6767766952],
            [0.6767766952, -0.3232233048, -0.25],
        ],
        y,
    )
}

fn initial_state() -> Vec<TaylorModel> {
    let domain = vec![
        interval!(-1.0, 1.0).unwrap(),
        interval!(-1.0, 1.0).unwrap(),
        interval!(-1.0, 1.0).unwrap(),
    ];

    (0..3)
        .map(|i| {
            TaylorModel::constant(Y0_MID, 3, ORDER, domain.clone())
                + TaylorModel::variable(i, Y0_RAD, 3, ORDER, domain.clone())
        })
        .collect()
}

fn run_naive(f: OdeFunction) {
    let initial = initial_state();
    let options = BungerOptions {
        order: ORDER,
        h: H,
        preconditioning: false,
        ..Default::default()
    };
    let start = Instant::now();

    match solve(initial, f, T0, TF, &options) {
        Ok(result) => {
            let elapsed = start.elapsed();
            let (_, final_values) = result.last().unwrap();
            let y1 = final_values[0];
            println!(
                "  naive         : ok,   {:>8.3?}, y1(100) width = {:.3e}, y1 = [{:.6e}, {:.6e}]",
                elapsed,
                y1.wid(),
                y1.inf(),
                y1.sup()
            );
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!("  naive         : FAILED after {:.3?}: {}", elapsed, e);
        }
    }
}

fn run_preconditioned(f: OdeFunction) {
    let initial = initial_state();
    let options = BungerOptions {
        order: ORDER,
        h: H,
        blunt_tau: Some(1e-3),
        preconditioning: true,
        ..Default::default()
    };
    let start = Instant::now();

    match solve(initial, f, T0, TF, &options) {
        Ok(result) => {
            let elapsed = start.elapsed();
            let (_, final_values) = result.last().unwrap();
            let y1 = final_values[0];
            println!(
                "  preconditioned: ok,   {:>8.3?}, y1(100) width = {:.3e}, y1 = [{:.6e}, {:.6e}]",
                elapsed,
                y1.wid(),
                y1.inf(),
                y1.sup()
            );
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!("  preconditioned: FAILED after {:.3?}: {}", elapsed, e);
        }
    }
}

fn main() {
    println!("==========================================");
    println!("Bünger linear ODE benchmark (paper, sec. 3.4)");
    println!("==========================================");
    println!("Taylor order : {}", ORDER);
    println!("Time interval: [{}, {}]", T0, TF);
    println!("Step size    : {}", H);
    println!("Initial y0   : (1,1,1) ± 0.001\n");

    println!("Case (1) — contraction, eig. -1/2,-3/4,0");
    println!("  reference (COSY/verifyode QR): y1(100) ~ [0.147300161861, 0.145593055090]");
    println!("  reference (shrink wrap)      : [-2.282e+112, 2.282e+112]  <- expected to fail");
    run_naive(linear1);
    run_preconditioned(linear1);

    println!("\nCase (2) — rotation, eig. ±i,0");
    println!("  reference (COSY/verifyode QR): y1(100) ~ 1.495212933089...");
    run_naive(linear2);
    run_preconditioned(linear2);

    println!("\nCase (3) — rotation+contraction, eig. ±i,-1/2");
    println!("  reference (COSY/verifyode QR): y1(100) ~ 1.348619867744...");
    println!("  reference (shrink wrap)      : [-3.566e+106, 3.566e+106]  <- expected to fail");
    run_naive(linear3);
    run_preconditioned(linear3);
}
