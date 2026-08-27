use inari::interval;

use interval_enclosure::ode::{plot_solution, solve_ode};
use interval_enclosure::taylor::TaylorModel;

// ============================================================
// Configuration
// ============================================================

const ORDER: usize = 10;

const T0: f64 = 0.0;
const TF: f64 = 0.15;

const H: f64 = 0.01;

const N_STEPS: usize = ((TF - T0) / H) as usize;

// ============================================================
// Lorenz parameters
// ============================================================

const SIGMA: f64 = 10.0;
const RHO: f64 = 28.0;
const BETA: f64 = 8.0 / 3.0;

// ============================================================
// Initial intervals
// ============================================================

const Y1_LO: f64 = -8.001;
const Y1_HI: f64 = -7.998;

const Y2_LO: f64 = 7.998;
const Y2_HI: f64 = 8.001;

const Y3_LO: f64 = 26.998;
const Y3_HI: f64 = 27.001;

// ============================================================
// Lorenz RHS
//
// y1' = sigma (y2 - y1)
//
// y2' = (rho - y3)y1 - y2
//
// y3' = y1 y2 - beta y3
// ============================================================

fn lorenz(y: Vec<TaylorModel>) -> Vec<TaylorModel> {
    assert_eq!(y.len(), 3);

    let y1 = y[0].clone();
    let y2 = y[1].clone();
    let y3 = y[2].clone();

    let dimension = y1.polynomial.dimension;
    let order = y1.order;
    let domain = y1.domain.clone();

    // Constants as Taylor models.

    let sigma = TaylorModel::constant(SIGMA, dimension, order, domain.clone());

    let rho = TaylorModel::constant(RHO, dimension, order, domain.clone());

    let beta = TaylorModel::constant(BETA, dimension, order, domain);

    // --------------------------------------------------------
    // Lorenz equations
    // --------------------------------------------------------

    let f1 = sigma * (y2.clone() - y1.clone());

    let f2 = (rho - y3.clone()) * y1.clone() - y2.clone();

    let f3 = y1 * y2 - beta * y3;

    vec![f1, f2, f3]
}

// ============================================================
// Initial Taylor model
//
// x1,x2,x3 ∈ [-1,1]
//
// y1 = midpoint(y1) + radius(y1) x1
// y2 = midpoint(y2) + radius(y2) x2
// y3 = midpoint(y3) + radius(y3) x3
//
// The time variable is added by solve_ode().
// ============================================================

fn initial_state() -> Vec<TaylorModel> {
    let domain = vec![
        interval!(-1.0, 1.0).unwrap(),
        interval!(-1.0, 1.0).unwrap(),
        interval!(-1.0, 1.0).unwrap(),
    ];

    // --------------------------------------------------------
    // y1
    // --------------------------------------------------------

    let y1_mid = 0.5 * (Y1_LO + Y1_HI);
    let y1_rad = 0.5 * (Y1_HI - Y1_LO);

    let y1 = TaylorModel::constant(y1_mid, 3, ORDER, domain.clone())
        + TaylorModel::variable(0, y1_rad, 3, ORDER, domain.clone());

    // --------------------------------------------------------
    // y2
    // --------------------------------------------------------

    let y2_mid = 0.5 * (Y2_LO + Y2_HI);
    let y2_rad = 0.5 * (Y2_HI - Y2_LO);

    let y2 = TaylorModel::constant(y2_mid, 3, ORDER, domain.clone())
        + TaylorModel::variable(1, y2_rad, 3, ORDER, domain.clone());

    // --------------------------------------------------------
    // y3
    // --------------------------------------------------------

    let y3_mid = 0.5 * (Y3_LO + Y3_HI);
    let y3_rad = 0.5 * (Y3_HI - Y3_LO);

    let y3 = TaylorModel::constant(y3_mid, 3, ORDER, domain)
        + TaylorModel::variable(
            2,
            y3_rad,
            3,
            ORDER,
            vec![
                interval!(-1.0, 1.0).unwrap(),
                interval!(-1.0, 1.0).unwrap(),
                interval!(-1.0, 1.0).unwrap(),
            ],
        );

    vec![y1, y2, y3]
}

// ============================================================
// Main
// ============================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("==========================================");
    println!("Bünger Taylor Model - Lorenz System");
    println!("==========================================");

    println!("Taylor order : {}", ORDER);
    println!("Time interval: [{}, {}]", T0, TF);
    println!("Step size    : {}", H);
    println!("Steps        : {}", N_STEPS);

    println!();

    println!("Initial y1 = [{}, {}]", Y1_LO, Y1_HI);
    println!("Initial y2 = [{}, {}]", Y2_LO, Y2_HI);
    println!("Initial y3 = [{}, {}]", Y3_LO, Y3_HI);

    println!();

    println!("Bünger Taylor-model arithmetic");
    println!("Record Feature: DISABLED");
    println!("Lorenz RHS: TaylorModel");
    println!("Polynomial multiplication: standard");
    println!("No QR preconditioning");
    println!("No shrink wrapping");

    println!();

    // --------------------------------------------------------
    // Initial condition
    // --------------------------------------------------------

    let initial = initial_state();

    // --------------------------------------------------------
    // Solve
    //
    // solve_ode() performs:
    //
    // 1. time lifting
    // 2. Picard iteration
    // 3. remainder verification
    // 4. epsilon inflation
    // 5. endpoint substitution
    // --------------------------------------------------------

    let result = solve_ode(initial, lorenz, T0, H, N_STEPS)?;

    // --------------------------------------------------------
    // Final enclosure
    // --------------------------------------------------------

    println!();
    println!("==========================================");
    println!("Final enclosure at t = {}", TF);

    let (_, final_values) = result.last().expect("empty result");

    for (i, value) in final_values.iter().enumerate() {
        println!("y{} = [{:.15e}, {:.15e}]", i + 1, value.inf(), value.sup());

        println!("width(y{}) = {:.6e}", i + 1, value.wid());
    }

    println!("==========================================");

    // --------------------------------------------------------
    // Plot
    // --------------------------------------------------------

    plot_solution(
        &result,
        "Bünger Taylor Model - Lorenz system",
        "lorenz_taylor_model.pdf",
    )?;

    println!();
    println!("PDF written to: lorenz_taylor_model.pdf");

    Ok(())
}
