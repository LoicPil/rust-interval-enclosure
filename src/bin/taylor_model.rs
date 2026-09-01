// ============================================================
// taylor_model_test.rs – Test de l'arithmétique Taylor
// (figure 4 de l'article de Bünger)
// ============================================================

use interval_enclosure::taylor::TaylorModel;
use inari::{Interval, interval};
use matplotlib::pyplot::subplots;

const ORDER: usize = 18;
const RADIUS: f64 = 0.1;
const GRID: usize = 401;
const ITERATIONS: usize = 4;

// Carte non-linéaire : (x,y) -> ( x - y*(0.125 + 2y),  y + 6x^3 )
fn f(x: TaylorModel, y: TaylorModel) -> (TaylorModel, TaylorModel) {
    let domain = x.domain.clone();
    let c   = TaylorModel::constant(interval!(0.125, 0.125).unwrap(), 2, ORDER, domain.clone());
    let two = TaylorModel::constant(interval!(2.0, 2.0).unwrap(),     2, ORDER, domain.clone());
    let six = TaylorModel::constant(interval!(6.0, 6.0).unwrap(),     2, ORDER, domain);
    let new_x = x.clone() - y.clone() * (c + two * y.clone());
    let new_y = y + six * x.powi(3);
    (new_x, new_y)
}

// Échantillonne le modèle sur la grille [-1,1]^2 (renvoie les milieux)
fn sample_tm(tm: &TaylorModel) -> Vec<Vec<f64>> {
    let mut pts = Vec::with_capacity(GRID*GRID);
    for i in 0..GRID {
        let u = -1.0 + 2.0 * i as f64 / (GRID-1) as f64;
        for j in 0..GRID {
            let v = -1.0 + 2.0 * j as f64 / (GRID-1) as f64;
            let val = tm.sample(&[u, v]); // renvoie un Interval
            pts.push(val.mid());
        }
    }
    pts
}

// Frontière vraie (application itérée de la carte sur le bord du carré)
fn true_boundary() -> Vec<[f64; 2]> {
    let mut pts = Vec::new();
    let n = GRID;
    for i in 0..n {
        let a = -RADIUS + 2.0*RADIUS * i as f64 / (n-1) as f64;
        pts.push([a, -RADIUS]);
        pts.push([RADIUS, a]);
        pts.push([a, RADIUS]);
        pts.push([-RADIUS, a]);
    }
    for _ in 0..ITERATIONS {
        for p in &mut pts {
            let x = p[0];
            let y = p[1];
            p[0] = x - y*(0.125 + 2.0*y);
            p[1] = y + 6.0*x.powi(3);
        }
    }
    pts
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = vec![interval!(-1.0, 1.0).unwrap(), interval!(-1.0, 1.0).unwrap()];
    let mut x = TaylorModel::variable(0, interval!(RADIUS, RADIUS).unwrap(), 2, ORDER, domain.clone());
    let mut y = TaylorModel::variable(1, interval!(RADIUS, RADIUS).unwrap(), 2, ORDER, domain);

    let mut all_points = Vec::new();
    for _ in 0..ITERATIONS {
        (x, y) = f(x, y);
        let pts = sample_tm(&x);
        all_points.push(pts);
    }

    let true_b = true_boundary();
    let xs_true: Vec<f64> = true_b.iter().map(|p| p[0]).collect();
    let ys_true: Vec<f64> = true_b.iter().map(|p| p[1]).collect();

    // Tracer les 4 itérations
    let (fig, [[mut ax1, mut ax2], [mut ax3, mut ax4]]) = subplots()?;
    let axes = [&mut ax1, &mut ax2, &mut ax3, &mut ax4];

    for (i, ax) in axes.into_iter().enumerate() {
        let pts = &all_points[i];
        // On prend un point sur GRID^2, on peut sous-échantillonner pour le tracé
        let xs: Vec<f64> = pts.iter().step_by(1).map(|v| v[0]).collect();
        let ys: Vec<f64> = pts.iter().step_by(1).map(|v| v[1]).collect();
        ax.xy(&xs, &ys).fmt(".").markersize(0.5).color([0.0, 0.0, 1.0]).plot();
        ax.xy(&xs_true, &ys_true).fmt("-").color([1.0, 0.0, 0.0]).plot();
        ax.set_title(&format!("Itération {}", i+1));
        ax.set_xlabel("x");
        ax.set_ylabel("y");
        ax.grid();
    }

    fig.save().to_file("taylor_model_test.pdf")?;
    println!("Graphique sauvegardé dans taylor_model_test.pdf");
    Ok(())
}
