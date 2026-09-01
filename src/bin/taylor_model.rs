// ============================================================
// src/bin/taylor_model.rs
//
// Reproduction of Bünger's Figure 4
//
// f(x,y) = ( x - y*(0.125 + 2y),
//            y + 6x^3 )
//
// B = [-0.1, 0.1]^2
//
// Taylor-model order = 18
// Four iterations
//
// ============================================================

use inari::interval;
use interval_enclosure::taylor::TaylorModel;
use matplotlib::pyplot::subplots;

// ============================================================
// Parameters
// ============================================================

const ORDER: usize = 10;
const RADIUS: f64 = 0.1;

const BOUNDARY_POINTS: usize = 1001;
const GRID: usize = 201;

const ITERATIONS: usize = 4;

// ============================================================
// Bünger map
// ============================================================

fn f(x: TaylorModel, y: TaylorModel) -> (TaylorModel, TaylorModel) {
    let domain = x.domain.clone();

    // 0.125
    let c = TaylorModel::constant(0.125, 2, ORDER, domain.clone());

    // 2
    let two = TaylorModel::constant(2.0, 2, ORDER, domain.clone());

    // 6
    let six = TaylorModel::constant(6.0, 2, ORDER, domain);

    // --------------------------------------------------------
    // x' = x - y(0.125 + 2y)
    // --------------------------------------------------------

    let new_x = x.clone() - y.clone() * (c + two * y.clone());

    // --------------------------------------------------------
    // y' = y + 6x^3
    // --------------------------------------------------------

    let new_y = y + six * x.powi(3);

    (new_x, new_y)
}

// ============================================================
// Initial boundary of B = [-0.1,0.1]^2
// ============================================================

fn initial_boundary() -> Vec<[f64; 2]> {
    let n = BOUNDARY_POINTS;

    let mut boundary = Vec::with_capacity(4 * n);

    for i in 0..n {
        let a = -1.0 + 2.0 * i as f64 / (n - 1) as f64;

        // Bottom
        boundary.push([RADIUS * a, -RADIUS]);

        // Right
        boundary.push([RADIUS, RADIUS * a]);

        // Top
        boundary.push([RADIUS * a, RADIUS]);

        // Left
        boundary.push([-RADIUS, RADIUS * a]);
    }

    boundary
}

// ============================================================
// Floating-point version of Bünger's map
// ============================================================

fn f_point(p: [f64; 2]) -> [f64; 2] {
    let x = p[0];
    let y = p[1];

    [x - y * (0.125 + 2.0 * y), y + 6.0 * x.powi(3)]
}

// ============================================================
// Apply f to the true boundary
// ============================================================

fn map_boundary(boundary: &mut [[f64; 2]]) {
    for p in boundary.iter_mut() {
        *p = f_point(*p);
    }
}

// ============================================================
// Sample polynomial image
//
// This evaluates only p(u,v), not p(u,v) + E.
// ============================================================

fn sample_polynomial_image(x_tm: &TaylorModel, y_tm: &TaylorModel) -> Vec<[f64; 2]> {
    let mut points = Vec::with_capacity(GRID * GRID);

    for i in 0..GRID {
        let u = -1.0 + 2.0 * i as f64 / (GRID - 1) as f64;

        for j in 0..GRID {
            let v = -1.0 + 2.0 * j as f64 / (GRID - 1) as f64;

            let x = x_tm.sample(&[u, v]);

            let y = y_tm.sample(&[u, v]);

            points.push([x, y]);
        }
    }

    points
}

// ============================================================
// Sample Taylor-model enclosure
//
// A Taylor model is:
//
//     p(D) + E
//
// We sample p(D), then add the four corners of E.
//
// This is only a visualization of the enclosure.
// The rigorous enclosure itself is x_tm.range(), y_tm.range().
// ============================================================

fn sample_taylor_model_range(x_tm: &TaylorModel, y_tm: &TaylorModel) -> Vec<[f64; 2]> {
    let polynomial_points = sample_polynomial_image(x_tm, y_tm);

    let ex = x_tm.remainder;
    let ey = y_tm.remainder;

    let ex_inf = ex.inf();
    let ex_sup = ex.sup();

    let ey_inf = ey.inf();
    let ey_sup = ey.sup();

    let shifts = [
        [ex_inf, ey_inf],
        [ex_inf, ey_sup],
        [ex_sup, ey_inf],
        [ex_sup, ey_sup],
    ];

    let mut points = Vec::with_capacity(polynomial_points.len() * 4);

    for p in polynomial_points {
        for shift in shifts {
            points.push([p[0] + shift[0], p[1] + shift[1]]);
        }
    }

    points
}

// ============================================================
// Main
// ============================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=========================================");
    println!("Bünger Figure 4");
    println!("Taylor Model Arithmetic");
    println!("=========================================");

    println!("Taylor order : {}", ORDER);
    println!("Initial box  : [{}, {}]^2", -RADIUS, RADIUS);
    println!("Iterations   : {}", ITERATIONS);
    println!();

    // ========================================================
    // Taylor-model domain
    //
    // D = [-1,1]^2
    //
    // Initial set:
    //
    // x = 0.1 u
    // y = 0.1 v
    //
    // (u,v) in [-1,1]^2
    // ========================================================

    let domain = vec![interval!(-1.0, 1.0).unwrap(), interval!(-1.0, 1.0).unwrap()];

    // ========================================================
    // Initial Taylor models
    // ========================================================

    let mut x = TaylorModel::variable(0, RADIUS, 2, ORDER, domain.clone());

    let mut y = TaylorModel::variable(1, RADIUS, 2, ORDER, domain.clone());

    // ========================================================
    // True boundary
    // ========================================================

    let mut true_boundary = initial_boundary();

    // ========================================================
    // Store all iterations
    // ========================================================

    let mut all_tm_points: Vec<Vec<[f64; 2]>> = Vec::with_capacity(ITERATIONS);

    let mut all_true_boundaries: Vec<Vec<[f64; 2]>> = Vec::with_capacity(ITERATIONS);

    // ========================================================
    // Iterate
    // ========================================================

    for iteration in 1..=ITERATIONS {
        // ----------------------------------------------------
        // Taylor-model image
        // ----------------------------------------------------

        let (new_x, new_y) = f(x, y);

        x = new_x;
        y = new_y;

        // ----------------------------------------------------
        // True image
        // ----------------------------------------------------

        map_boundary(&mut true_boundary);

        // ----------------------------------------------------
        // Store
        // ----------------------------------------------------

        all_tm_points.push(sample_taylor_model_range(&x, &y));

        all_true_boundaries.push(true_boundary.clone());

        // -------- DEBUG OUTPUT --------
        println!();
        println!("Iteration {} — Diagnostic", iteration);
        println!("  x poly range:  {}", x.polynomial_range());
        println!("  x remainder:   {}", x.remainder);
        println!("  x total range: {}", x.range());
        println!("  y poly range:  {}", y.polynomial_range());
        println!("  y remainder:   {}", y.remainder);
        println!("  y total range: {}", y.range());
    }

    println!();

    // ========================================================
    // Plot
    // ========================================================

    let (fig, [[mut ax1, mut ax2], [mut ax3, mut ax4]]) = subplots()?;

    let mut axes = [&mut ax1, &mut ax2, &mut ax3, &mut ax4];

    for i in 0..ITERATIONS {
        let ax = &mut axes[i];

        // ----------------------------------------------------
        // Taylor-model enclosure
        // ----------------------------------------------------

        let tm_points = &all_tm_points[i];

        let xs_tm: Vec<f64> = tm_points.iter().map(|p| p[0]).collect();

        let ys_tm: Vec<f64> = tm_points.iter().map(|p| p[1]).collect();

        ax.xy(&xs_tm, &ys_tm)
            .fmt(".")
            .markersize(0.4)
            .color([0.0, 0.0, 0.0])
            .plot();

        // ----------------------------------------------------
        // True boundary
        // ----------------------------------------------------

        let true_points = &all_true_boundaries[i];

        let xs_true: Vec<f64> = true_points.iter().map(|p| p[0]).collect();

        let ys_true: Vec<f64> = true_points.iter().map(|p| p[1]).collect();

        ax.xy(&xs_true, &ys_true)
            .fmt("-")
            .color([1.0, 0.0, 0.0])
            .linewidth(1.0)
            .plot();

        // ----------------------------------------------------
        // Formatting
        // ----------------------------------------------------

        ax.set_title(&format!("Iteration {}", i + 1));

        ax.set_xlabel("x");
        ax.set_ylabel("y");

        ax.grid();
    }

    // ========================================================
    // Save
    // ========================================================

    fig.save().to_file("taylor_model_bunger_figure4.pdf")?;

    println!("Figure saved to taylor_model_bunger_figure4.pdf");

    Ok(())
}
