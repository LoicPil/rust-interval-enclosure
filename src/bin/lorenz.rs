use inari::Interval;
use matplotlib::pyplot::subplots;

use interval_enclosure::ode::{BungerOptions, solve};
use interval_enclosure::taylor::TaylorModel;

// ============================================================
// Configuration
// ============================================================

const ORDER: usize = 10;

const T0: f64 = 0.0;
const TF: f64 = 100.0;

const H: f64 = 0.01;

// ============================================================
// Lorenz parameters
// ============================================================

const SIGMA: f64 = 10.0;
const RHO: f64 = 28.0;
const BETA: f64 = 8.0 / 3.0;

// ============================================================
// Initial intervals
// ============================================================
//
// x(0) ∈ [-8.001, -7.998]
// y(0) ∈ [ 7.998,  8.001]
// z(0) ∈ [26.998, 27.001]
//
// ============================================================

const Y1_LO: f64 = -8.001;
const Y1_HI: f64 = -7.998;

const Y2_LO: f64 = 7.998;
const Y2_HI: f64 = 8.001;

const Y3_LO: f64 = 26.998;
const Y3_HI: f64 = 27.001;

// ============================================================
// Lorenz RHS
// ============================================================
//
//     y1' = sigma (y2 - y1)
//
//     y2' = (rho - y3)y1 - y2
//
//     y3' = y1 y2 - beta y3
//
// ============================================================

fn lorenz(y: Vec<TaylorModel>) -> Vec<TaylorModel> {
    assert_eq!(y.len(), 3);

    let y1 = y[0].clone();
    let y2 = y[1].clone();
    let y3 = y[2].clone();

    let dimension = y1.polynomial.dimension;
    let order = y1.order;
    let domain = y1.domain.clone();

    // --------------------------------------------------------
    // Constants
    // --------------------------------------------------------

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
// Initial state
// ============================================================
//
// The initial parameters are:
//
//     x1, x2, x3 ∈ [-1,1]
//
// with:
//
//     y1 = midpoint(y1) + radius(y1) x1
//     y2 = midpoint(y2) + radius(y2) x2
//     y3 = midpoint(y3) + radius(y3) x3
//
// ============================================================

fn initial_state() -> Vec<TaylorModel> {
    TaylorModel::parameterized_initial_set(
        &[
            0.5 * (Y1_LO + Y1_HI),
            0.5 * (Y2_LO + Y2_HI),
            0.5 * (Y3_LO + Y3_HI),
        ],
        &[
            0.5 * (Y1_HI - Y1_LO),
            0.5 * (Y2_HI - Y2_LO),
            0.5 * (Y3_HI - Y3_LO),
        ],
        ORDER,
    )
}

// ============================================================
// Helper: interval midpoint
// ============================================================

fn interval_midpoint(x: Interval) -> f64 {
    0.5 * (x.inf() + x.sup())
}

// ============================================================
// Lorenz attractor: x-z projection
// ============================================================

fn plot_lorenz_attractor(
    result: &[(f64, Vec<Interval>)],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(!result.is_empty());

    let xs: Vec<f64> = result
        .iter()
        .map(|(_, values)| {
            assert!(values.len() >= 3);
            interval_midpoint(values[0])
        })
        .collect();

    let zs: Vec<f64> = result
        .iter()
        .map(|(_, values)| {
            assert!(values.len() >= 3);
            interval_midpoint(values[2])
        })
        .collect();

    let (fig, [[mut ax]]) = subplots()?;

    ax.xy(&xs, &zs).fmt("-").label("trajectory").plot();

    ax.set_title("Lorenz Attractor");
    ax.set_xlabel("x");
    ax.set_ylabel("z");
    ax.grid();
    ax.legend(std::iter::empty());

    fig.save().to_file(filename)?;

    Ok(())
}

// ============================================================
// Lorenz attractor: x-y projection
// ============================================================

fn plot_lorenz_xy(
    result: &[(f64, Vec<Interval>)],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(!result.is_empty());

    let xs: Vec<f64> = result
        .iter()
        .map(|(_, values)| {
            assert!(values.len() >= 3);
            interval_midpoint(values[0])
        })
        .collect();

    let ys: Vec<f64> = result
        .iter()
        .map(|(_, values)| {
            assert!(values.len() >= 3);
            interval_midpoint(values[1])
        })
        .collect();

    let (fig, [[mut ax]]) = subplots()?;

    ax.xy(&xs, &ys).fmt("-").label("trajectory").plot();

    ax.set_title("Lorenz Attractor - x/y projection");
    ax.set_xlabel("x");
    ax.set_ylabel("y");
    ax.grid();
    ax.legend(std::iter::empty());

    fig.save().to_file(filename)?;

    Ok(())
}

// ============================================================
// Lorenz attractor: y-z projection
// ============================================================

fn plot_lorenz_yz(
    result: &[(f64, Vec<Interval>)],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(!result.is_empty());

    let ys: Vec<f64> = result
        .iter()
        .map(|(_, values)| {
            assert!(values.len() >= 3);
            interval_midpoint(values[1])
        })
        .collect();

    let zs: Vec<f64> = result
        .iter()
        .map(|(_, values)| {
            assert!(values.len() >= 3);
            interval_midpoint(values[2])
        })
        .collect();

    let (fig, [[mut ax]]) = subplots()?;

    ax.xy(&ys, &zs).fmt("-").label("trajectory").plot();

    ax.set_title("Lorenz Attractor - y/z projection");
    ax.set_xlabel("y");
    ax.set_ylabel("z");
    ax.grid();
    ax.legend(std::iter::empty());

    fig.save().to_file(filename)?;

    Ok(())
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
    println!("Steps        : {}", ((TF - T0) / H).round() as usize);

    println!();

    println!("Initial y1 = [{}, {}]", Y1_LO, Y1_HI);
    println!("Initial y2 = [{}, {}]", Y2_LO, Y2_HI);
    println!("Initial y3 = [{}, {}]", Y3_LO, Y3_HI);

    println!();

    println!("Bünger Taylor-model arithmetic");
    println!("Record Feature: ENABLED");
    println!("Lorenz RHS: TaylorModel");
    println!("Polynomial multiplication: recorded");
    println!("QR preconditioning: ENABLED");
    println!("Shrink wrapping: DISABLED");

    println!();

    // ========================================================
    // Initial condition
    // ========================================================

    let initial = initial_state();

    // ========================================================
    // Bünger options
    // ========================================================

    let options = BungerOptions {
        order: ORDER,
        h: H,
        epsilon: 0.01,
        delta: 1e-12,
        max_inflation_iterations: 50,

        // Kept available for future blunting experiments.
        // The current QR preconditioner does not use it.
        blunt_tau: Some(1e-8),

        // This is the important part:
        // use Bünger's QR preconditioning.
        preconditioning: true,
    };

    // ========================================================
    // Verified integration
    // ========================================================

    let result = solve(initial, lorenz, T0, TF, &options)
        .map_err(|e| format!("Bünger solver failed: {}", e))?;

    // ========================================================
    // Final enclosure
    // ========================================================

    println!();

    println!("==========================================");
    println!("Final enclosure at t = {}", TF);

    let (_, final_values) = result.last().expect("empty result");

    for (i, value) in final_values.iter().enumerate() {
        println!("y{} = [{:.15e}, {:.15e}]", i + 1, value.inf(), value.sup());

        println!("width(y{}) = {:.6e}", i + 1, value.wid());
    }

    println!("==========================================");

    // ========================================================
    // Plot 1: components versus time
    // ========================================================

    interval_enclosure::ode::plot_solution(
        &result,
        "Bünger Taylor Model - Lorenz system",
        "lorenz_taylor_model.pdf",
    )?;

    println!();
    println!("PDF written to: lorenz_taylor_model.pdf");

    // ========================================================
    // Plot 2: Lorenz attractor x-z
    // ========================================================

    plot_lorenz_attractor(&result, "lorenz_attractor.pdf")?;

    println!("PDF written to: lorenz_attractor.pdf");

    // ========================================================
    // Plot 3: x-y projection
    // ========================================================

    plot_lorenz_xy(&result, "lorenz_attractor_xy.pdf")?;

    println!("PDF written to: lorenz_attractor_xy.pdf");

    // ========================================================
    // Plot 4: y-z projection
    // ========================================================

    plot_lorenz_yz(&result, "lorenz_attractor_yz.pdf")?;

    println!("PDF written to: lorenz_attractor_yz.pdf");

    println!();

    Ok(())
}
