use inari::interval;

use interval_enclosure::ode::{plot_component, solve_bunger};
use interval_enclosure::taylor::TaylorModel;

// ============================================================
// Configuration
// ============================================================

const ORDER: usize = 10;

const T0: f64 = 0.0;
const TF: f64 = 3.0;
const H: f64 = 0.01;

const N_STEPS: usize = ((TF - T0) / H) as usize;

// Initial uncertainty
const U_LO: f64 = 0.499;
const U_HI: f64 = 0.501;

// ============================================================
// Bünger parameters
// ============================================================

const PICARD_ITERATIONS: usize = ORDER;

const EPSILON: f64 = 1e-2;
const DELTA: f64 = 1e-12;

const MAX_INFLATION_ITERATIONS: usize = 100;

// ============================================================
// ODE
//
//                 u' = u² - u
//
// IMPORTANT:
//
// This function receives a Taylor model in the
// (x,t) domain after the initial state has been
// lifted with extend_with_time().
//
// ============================================================

fn ode_1d(y: Vec<TaylorModel>) -> Vec<TaylorModel> {
    assert_eq!(y.len(), 1);

    let u = &y[0];

    // u² - u
    //
    // We clone only where ownership requires it.
    let u_squared = u.clone() * u.clone();

    vec![u_squared - u.clone()]
}

// ============================================================
// Initial condition
//
// x ∈ [-1,1]
//
// u(0,x)
//     = u_mid + u_rad x
//
// Therefore
//
// u(0,x) ∈ [0.499,0.501].
// ============================================================

fn initial_state() -> Vec<TaylorModel> {
    let domain = vec![interval!(-1.0, 1.0).unwrap()];

    let u_mid = 0.5 * (U_LO + U_HI);
    let u_rad = 0.5 * (U_HI - U_LO);

    let constant = TaylorModel::constant(u_mid, 1, ORDER, domain.clone());

    let uncertainty = TaylorModel::variable(0, u_rad, 1, ORDER, domain);

    vec![constant + uncertainty]
}

// ============================================================
// Main
// ============================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("==========================================");
    println!("Bünger Taylor Model - 1D ODE");
    println!("==========================================");

    println!("ODE          : u' = u² - u");
    println!("Taylor order : {}", ORDER);
    println!("Time interval: [{}, {}]", T0, TF);
    println!("Step size    : {}", H);
    println!("Steps        : {}", N_STEPS);

    println!();
    println!("Initial u    : [{}, {}]", U_LO, U_HI);

    println!();
    println!("Picard iterations       : {}", PICARD_ITERATIONS);
    println!("Epsilon                 : {}", EPSILON);
    println!("Delta                   : {:.3e}", DELTA);
    println!("Max inflation iterations: {}", MAX_INFLATION_ITERATIONS);

    // --------------------------------------------------------
    // Initial model
    // --------------------------------------------------------

    let initial = initial_state();

    // --------------------------------------------------------
    // IMPORTANT:
    //
    // solve_bunger() expects the Taylor models to already
    // contain the time variable.
    //
    // Initially:
    //
    //     dimension = 1
    //
    // After lifting:
    //
    //     dimension = 2
    //
    //     (x,t)
    //
    let initial = initial
        .into_iter()
        .map(|tm| tm.extend_with_time(H))
        .collect();

    // --------------------------------------------------------
    // Solve
    // --------------------------------------------------------

    let result = solve_bunger(
        initial,
        ode_1d,
        T0,
        H,
        N_STEPS,
        PICARD_ITERATIONS,
        EPSILON,
        DELTA,
        MAX_INFLATION_ITERATIONS,
    )?;

    // --------------------------------------------------------
    // Final enclosure
    // --------------------------------------------------------

    println!();
    println!("==========================================");
    println!("Final enclosure at t = {}", TF);

    let (_, final_values) = result.last().unwrap();

    let u = final_values[0];

    println!("u = [{:.15e}, {:.15e}]", u.inf(), u.sup());

    println!("width(u) = {:.6e}", u.wid());

    println!("==========================================");

    // --------------------------------------------------------
    // Plot
    // --------------------------------------------------------

    plot_component(
        &result,
        0,
        "Bünger Taylor Model: u' = u² - u",
        "u",
        "ode_1d_taylor_model.pdf",
    )?;

    println!();
    println!("PDF written to: ode_1d_taylor_model.pdf");

    Ok(())
}
