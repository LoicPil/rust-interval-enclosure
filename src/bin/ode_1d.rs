use inari::interval;
use interval_enclosure::ode::{BungerOptions, plot_solution, solve};
use interval_enclosure::taylor::TaylorModel;
use interval_enclosure::timed;

const ORDER: usize = 10;
const T0: f64 = 0.0;
const TF: f64 = 2.5;
const H: f64 = 0.001; // pas plus petit pour plus de stabilité

const U_LO: f64 = 0.499;
const U_HI: f64 = 0.501;

/// u' = u² - u
fn ode_1d(y: Vec<TaylorModel>) -> Vec<TaylorModel> {
    assert_eq!(y.len(), 1);
    let u = &y[0];
    let u_squared = u.clone() * u.clone();
    vec![u_squared - u.clone()]
}

/// u(0,x) = u_mid + u_rad * x,  x ∈ [-1,1]  ⇒  u(0) ∈ [0.499, 0.501]
fn initial_state() -> Vec<TaylorModel> {
    let domain = vec![interval!(-1.0, 1.0).unwrap()];
    let u_mid = 0.5 * (U_LO + U_HI);
    let u_rad = 0.5 * (U_HI - U_LO);

    // Avec la version f64, constant et variable prennent des f64
    let constant = TaylorModel::constant(u_mid, 1, ORDER, domain.clone());
    let uncertainty = TaylorModel::variable(0, u_rad, 1, ORDER, domain);

    vec![constant + uncertainty]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("==========================================");
    println!("Bünger Taylor Model - 1D ODE");
    println!("==========================================");
    println!("ODE          : u' = u² - u");

    let options = BungerOptions {
        order: ORDER,
        h: H,
        epsilon: 1e-2,
        delta: 1e-12,
        max_inflation_iterations: 50,
        blunt_tau: Some(1e-8),
        preconditioning: false, // ESSENTIEL : désactiver le préconditionneur factice
    };

    println!("Taylor order : {}", options.order);
    println!("Time interval: [{}, {}]", T0, TF);
    println!("Step size    : {}", options.h);
    println!();
    println!("Initial u    : [{}, {}]", U_LO, U_HI);
    println!();
    println!("Epsilon                 : {}", options.epsilon);
    println!("Delta                   : {:.3e}", options.delta);
    println!(
        "Max inflation iterations: {}",
        options.max_inflation_iterations
    );

    let initial = initial_state();

    let result = timed!("Solve :", solve(initial, ode_1d, T0, TF, &options))?;

    println!();
    println!("==========================================");
    println!("Final enclosure at t = {}", TF);

    let (_, final_values) = result.last().unwrap();
    let u = final_values[0];

    println!("u = [{:.15e}, {:.15e}]", u.inf(), u.sup());
    println!("width(u) = {:.6e}", u.wid());
    println!("==========================================");

    // Utiliser plot_solution (exportée) ou plot_component selon votre préférence
    plot_solution(
        &result,
        "Bünger Taylor Model: u' = u² - u",
        "ode_1d_taylor_model.pdf",
    )?;

    println!();
    println!("PDF written to: ode_1d_taylor_model.pdf");

    Ok(())
}
