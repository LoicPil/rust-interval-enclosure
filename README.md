# Interval-enclosure

Rigorous interval enclosure for ODE solutions using Taylor models and certified integration.

## Overview

This crate implements verified numerical methods for two main problems:

1. **Certified ODE integration** using the Taylor model approach inspired by the work of Florian Bünger [1].
2. **Certified quadrature** (midpoint, trapezoidal, Simpson, and Gauss-Legendre) with rigorous error bounds via interval arithmetic.

Taylor models combine multivariate polynomials with interval remainders to represent and propagate solution sets with guaranteed error bounds.

**Current limitations:**

- ODE solver only supports first-order systems (higher-order ODEs must be reduced to first-order form)
- Taylor model standard functions (exp, sin, cos, etc.) are not yet fully implemented

## Features

**ODE solving:**

- Taylor model arithmetic with rigorous rounding error tracking
- Picard iteration for polynomial part computation
- ε-δ inflation for remainder verification
- Preconditioning (parallelepiped and QR) to reduce the wrapping effect
- Blunting for near-singular matrices
- Verified enclosures for interval initial value problems

**Certified integration:**

- Midpoint rule with rigorous error bounds
- Trapezoidal rule with rigorous error bounds
- Simpson rule with rigorous error bounds  
- Gauss-Legendre quadrature (Golub-Welsch construction)
- Adaptive integration with error control
- Certified error bounds using interval arithmetic for derivatives

## Theoretical Background

### Interval Arithmetic

Interval arithmetic provides the foundation for rigorous computation, where real numbers are represented by intervals to bound errors. Key concepts include:

- Interval evaluation of expressions yields guaranteed bounds
- Outward rounding ensures results contain the true mathematical value
- Verified computation provides mathematically rigorous error bounds

See the Wikipedia article on interval arithmetic [3] and the work of Günter Mayer [2] for a comprehensive treatment.

This crate uses the [inari](https://github.com/steffahn/inari) library with MPFR/GMP backend for rigorous interval operations.

### Certified Quadrature

The quadrature methods in this crate follow the approach described in Gautschi [5]:

1. **Gauss-Legendre quadrature** is constructed via the Golub-Welsch algorithm:
   - The Jacobi matrix is formed from the recurrence coefficients of Legendre polynomials
   - Eigenvalues give the nodes, eigenvectors give the weights
   - DLMF values [4] are used as reference for orders 5, 10, and 20

2. **Certified error bounds** are obtained using:
   - The classical error formula for each quadrature rule
   - Interval arithmetic to bound the derivative term
   - The result is an interval guaranteed to contain the exact integral

3. **Adaptive integration** uses a priority queue to refine subintervals with the largest error estimates.

### ODE Integration

The ODE verification method follows **Bünger [1]**: Picard iteration computes the Taylor polynomial, then an ε-δ inflation loop verifies the enclosure condition `K(p+E) ⊆ p+E` using Schauder's fixed-point theorem.

## Dependencies

- [rust-matplotlib](https://github.com/Chris00/rust-matplotlib) for plotting (requires Python + matplotlib)
- [inari](https://github.com/steffahn/inari) for interval arithmetic (requires MPFR + GMP)
- [nalgebra](https://github.com/dimforge/nalgebra) for linear algebra (Golub-Welsch)

## References

1. **[Bünger, F. (2020)](https://doi.org/10.1016/j.cam.2019.112511)**  
   *A Taylor model toolbox for solving ODEs implemented in MATLAB/INTLAB*  
   Journal of Computational and Applied Mathematics, 368, 112511.

2. **[Mayer, G. (2017)](https://doi.org/10.1515/9783110499469)**  
   *Interval Analysis: And Automatic Result Verification*  
   De Gruyter. ISBN: 978-3-110-49380-1

3. **[Interval Arithmetic (Wikipedia)](https://en.wikipedia.org/wiki/Interval_arithmetic)**

4. **[NIST Digital Library of Mathematical Functions (2024)](https://dlmf.nist.gov/)**  
   Chapter 18: Orthogonal Polynomials. Table 18.3.1 for leading coefficients and norms.

5. **[Gautschi, W. (2004)](https://doi.org/10.1093/acprof:oso/9780198506720.001.0001)**  
   *Orthogonal Polynomials: Computation and Approximation*  
   Oxford University Press.

## Example: ODE Integration

```rust
use interval_enclosure::{BungerOptions, solve, TaylorModel};

fn harmonic_oscillator(y: Vec<TaylorModel>) -> Vec<TaylorModel> {
    vec![y[1].clone(), -y[0].clone()]
}

let initial = TaylorModel::parameterized_initial_set(&[1.0, 0.0], &[1e-6, 1e-6], 10);
let options = BungerOptions { order: 10, h: 0.01, epsilon: 0.01, delta: 1e-12, 
                              max_inflation_iterations: 50, blunt_tau: Some(1e-10), 
                              preconditioning: true };

let result = solve(initial, harmonic_oscillator, 0.0, 10.0, &options).unwrap();
interval_enclosure::plot_solution(&result, "Harmonic Oscillator", "osc.png").unwrap();
```

## Example: Certified Integration

```rust
use interval_enclosure::integration::{GaussLegendreRule, gauss_legendre_certified};

let rule = GaussLegendreRule::new(5);

// Integral of exp(x) from 0 to 1, with certified error bound
// f^(10)(x) = exp(x) for order 5 rule (2n = 10)
let result = gauss_legendre_certified(
    &rule,
    |x| x.exp(),
    |x| x.exp(),
    0.0, 1.0, 1
).unwrap();

assert!(result.contains(std::f64::consts::E - 1.0));
```
