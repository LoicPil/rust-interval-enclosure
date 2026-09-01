use inari::{Interval, interval};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul, Sub};

/// Below this magnitude, a coefficient is dropped from the polynomial and
/// its worst-case contribution over the domain is folded into the owning
/// TaylorModel's remainder instead (see `TaylorModel::sparsify`). Tune this
/// like Bünger's `sparsity_tol` (he uses 1e-20 for Lorenz, 1e-12/1e-16 for
/// the double pendulum) — too loose inflates the remainder faster than
/// necessary, too tight leaves term counts (and runtime) exploding.
pub const SPARSITY_THRESHOLD: f64 = 1e-16;

#[derive(Clone, Debug)]
pub struct Polynomial {
    coeffs: HashMap<Vec<usize>, f64>,
    degree: usize,
    pub dimension: usize,
}

impl Polynomial {
    pub fn new(dimension: usize, degree: usize) -> Self {
        Self {
            coeffs: HashMap::new(),
            degree,
            dimension,
        }
    }

    pub fn get(&self, exponents: &[usize]) -> f64 {
        *self.coeffs.get(exponents).unwrap_or(&0.0)
    }

    /// Sets a coefficient. Only drops it on an EXACT zero — this is a plain
    /// bookkeeping accessor, not a sparsity policy. Sparsification (which is
    /// allowed to drop small-but-nonzero coefficients) must go through
    /// `sparsify()` below, which folds the dropped range into the remainder
    /// so the enclosure stays sound.
    pub fn set(&mut self, exponents: &[usize], value: f64) {
        assert_eq!(exponents.len(), self.dimension);
        assert!(
            value.is_finite(),
            "non-finite coefficient {} at {:?}",
            value,
            exponents
        );
        if value == 0.0 {
            self.coeffs.remove(exponents);
        } else {
            self.coeffs.insert(exponents.to_vec(), value);
        }
    }

    pub fn constant(dimension: usize, degree: usize, value: f64) -> Self {
        let mut p = Self::new(dimension, degree);
        p.set(&vec![0; dimension], value);
        p
    }

    pub fn variable(dimension: usize, degree: usize, variable: usize, coefficient: f64) -> Self {
        assert!(variable < dimension);
        let mut p = Self::new(dimension, degree);
        let mut exponents = vec![0; dimension];
        exponents[variable] = 1;
        p.set(&exponents, coefficient);
        p
    }

    pub fn total_degree(exponents: &[usize]) -> usize {
        exponents.iter().sum()
    }

    pub fn evaluate(&self, domain: &[Interval]) -> Interval {
        assert_eq!(domain.len(), self.dimension);
        let mut result = interval!(0.0, 0.0).unwrap();
        for (exponents, coefficient) in &self.coeffs {
            let mut term = interval!(*coefficient, *coefficient).unwrap();
            for (i, exponent) in exponents.iter().enumerate() {
                if *exponent > 0 {
                    term *= domain[i].powi(*exponent as i32);
                }
            }
            result += term;
        }
        result
    }

    pub fn split(&self, order: usize) -> (Polynomial, Polynomial) {
        let mut low = Polynomial::new(self.dimension, order);
        let mut high = Polynomial::new(self.dimension, self.degree);
        for (exponents, coefficient) in &self.coeffs {
            if Self::total_degree(exponents) <= order {
                low.set(exponents, *coefficient);
            } else {
                high.set(exponents, *coefficient);
            }
        }
        (low, high)
    }

    pub fn sample(&self, point: &[f64]) -> f64 {
        assert_eq!(point.len(), self.dimension);
        let mut result = 0.0;
        for (exponents, coefficient) in &self.coeffs {
            let mut term = *coefficient;
            for (x, exponent) in point.iter().zip(exponents.iter()) {
                term *= x.powi(*exponent as i32);
            }
            result += term;
        }
        result
    }

    pub fn terms(&self) -> Vec<(&Vec<usize>, &f64)> {
        let mut terms: Vec<_> = self.coeffs.iter().collect();
        terms.sort_by_key(|(e, _)| Self::total_degree(e));
        terms
    }

    /// Drops coefficients with |c| < threshold; returns the total range those
    /// dropped terms could have contributed over `domain`. The caller (see
    /// `TaylorModel::sparsify`) must fold this into the remainder so the
    /// enclosure remains a rigorous superset even after the drop.
    pub fn sparsify(&mut self, domain: &[Interval], threshold: f64) -> Interval {
        assert_eq!(domain.len(), self.dimension);

        let to_remove: Vec<Vec<usize>> = self
            .coeffs
            .iter()
            .filter(|&(_, &c)| c.abs() < threshold)
            .map(|(e, _)| e.clone())
            .collect();

        let mut dropped = interval!(0.0, 0.0).unwrap();
        for exponents in to_remove {
            let coeff = self.coeffs.remove(&exponents).unwrap();
            let mut term = interval!(coeff, coeff).unwrap();
            for (i, &exponent) in exponents.iter().enumerate() {
                if exponent > 0 {
                    term *= domain[i].powi(exponent as i32);
                }
            }
            dropped += term;
        }
        dropped
    }
}

impl Add for Polynomial {
    type Output = Polynomial;
    fn add(mut self, other: Polynomial) -> Polynomial {
        assert_eq!(self.dimension, other.dimension);
        assert_eq!(self.degree, other.degree);
        for (exponents, coefficient) in other.coeffs {
            let value = self.get(&exponents) + coefficient;
            self.set(&exponents, value);
        }
        self
    }
}

impl Sub for Polynomial {
    type Output = Polynomial;
    fn sub(mut self, other: Polynomial) -> Polynomial {
        assert_eq!(self.dimension, other.dimension);
        assert_eq!(self.degree, other.degree);
        for (exponents, coefficient) in other.coeffs {
            let value = self.get(&exponents) - coefficient;
            self.set(&exponents, value);
        }
        self
    }
}

impl Mul for Polynomial {
    type Output = Polynomial;
    fn mul(self, other: Polynomial) -> Polynomial {
        assert_eq!(self.dimension, other.dimension);
        let degree = self.degree + other.degree;
        let mut result = Polynomial::new(self.dimension, degree);
        for (e1, c1) in &self.coeffs {
            for (e2, c2) in &other.coeffs {
                let exponents: Vec<usize> = e1.iter().zip(e2.iter()).map(|(a, b)| a + b).collect();
                let value = result.get(&exponents) + c1 * c2;
                result.set(&exponents, value);
            }
        }
        result
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PolyKey(Vec<(Vec<usize>, u64)>);

impl PolyKey {
    fn from(p: &Polynomial) -> Self {
        let mut v: Vec<(Vec<usize>, u64)> = p
            .terms()
            .into_iter()
            .map(|(e, c)| (e.clone(), c.to_bits()))
            .collect();
        v.sort();
        PolyKey(v)
    }
}

thread_local! {
    static MUL_CACHE: RefCell<HashMap<(PolyKey, PolyKey), (Polynomial, Interval, Interval, Interval)>> =
        RefCell::new(HashMap::new());
    static INTEGRATE_CACHE: RefCell<HashMap<PolyKey, (Polynomial, Interval)>> =
        RefCell::new(HashMap::new());
}

/// Drops all cached polynomial-only results. Call once per integration step
/// (Bünger's record feature is only valid while p stays frozen; a new step
/// means a new p, so stale entries must not survive across steps).
pub fn clear_caches() {
    MUL_CACHE.with(|c| c.borrow_mut().clear());
    INTEGRATE_CACHE.with(|c| c.borrow_mut().clear());
}

#[derive(Clone, Debug)]
pub struct TaylorModel {
    pub polynomial: Polynomial,
    pub remainder: Interval,
    pub domain: Vec<Interval>,
    pub order: usize,
}

impl TaylorModel {
    pub fn constant(value: f64, dimension: usize, order: usize, domain: Vec<Interval>) -> Self {
        Self {
            polynomial: Polynomial::constant(dimension, order, value),
            remainder: interval!(0.0, 0.0).unwrap(),
            domain,
            order,
        }
    }

    pub fn variable(
        variable: usize,
        coefficient: f64,
        dimension: usize,
        order: usize,
        domain: Vec<Interval>,
    ) -> Self {
        Self {
            polynomial: Polynomial::variable(dimension, order, variable, coefficient),
            remainder: interval!(0.0, 0.0).unwrap(),
            domain,
            order,
        }
    }

    pub fn from_interval(
        value: Interval,
        dimension: usize,
        order: usize,
        domain: Vec<Interval>,
    ) -> Self {
        Self {
            polynomial: Polynomial::constant(dimension, order, 0.0),
            remainder: value,
            domain,
            order,
        }
    }

    pub fn range(&self) -> Interval {
        self.polynomial.evaluate(&self.domain) + self.remainder
    }

    pub fn polynomial_range(&self) -> Interval {
        self.polynomial.evaluate(&self.domain)
    }

    /// Drops small coefficients (see `SPARSITY_THRESHOLD`), folding their
    /// worst-case range into `remainder` so the enclosure stays sound.
    /// Called automatically at the end of `+`, `-`, `*`, and `integrate_time`
    /// — the operations that actually grow term count.
    pub fn sparsify(mut self, threshold: f64) -> TaylorModel {
        let dropped = self.polynomial.sparsify(&self.domain, threshold);
        self.remainder = self.remainder + dropped;
        self
    }

    pub fn powi(&self, exponent: usize) -> TaylorModel {
        if exponent == 0 {
            return TaylorModel::constant(
                1.0,
                self.polynomial.dimension,
                self.order,
                self.domain.clone(),
            );
        }
        let mut result = self.clone();
        for _ in 1..exponent {
            result = result * self.clone();
        }
        result
    }

    pub fn sample(&self, point: &[f64]) -> f64 {
        self.polynomial.sample(point)
    }

    pub fn integrate_time(&self) -> TaylorModel {
        let time_var = self.polynomial.dimension - 1;
        let key = PolyKey::from(&self.polynomial);

        let (r, s_range) = INTEGRATE_CACHE.with(|cache| {
            if let Some(cached) = cache.borrow().get(&key) {
                return cached.clone();
            }
            let mut antideriv = Polynomial::new(self.polynomial.dimension, self.order + 1);
            for (exponents, coeff) in self.polynomial.terms() {
                let mut new_exp = exponents.clone();
                new_exp[time_var] += 1;
                let k = new_exp[time_var] as f64;
                let value = antideriv.get(&new_exp) + *coeff / k;
                antideriv.set(&new_exp, value);
            }
            let (r, s) = antideriv.split(self.order);
            let s_range = s.evaluate(&self.domain);
            let entry = (r, s_range);
            cache.borrow_mut().insert(key, entry.clone());
            entry
        });

        let time_range = self.domain[time_var];
        let new_remainder = s_range + self.remainder * time_range;

        let result = TaylorModel {
            polynomial: r,
            remainder: new_remainder,
            domain: self.domain.clone(),
            order: self.order,
        };
        result.sparsify(SPARSITY_THRESHOLD)
    }

    pub fn substitute_time(&self, t: f64) -> TaylorModel {
        let time_var = self.polynomial.dimension - 1;
        let new_dim = time_var;

        let mut new_poly = Polynomial::new(new_dim, self.order);
        for (exponents, coeff) in self.polynomial.terms() {
            let factor = t.powi(exponents[time_var] as i32);
            let new_exp = exponents[..new_dim].to_vec();
            let value = new_poly.get(&new_exp) + *coeff * factor;
            new_poly.set(&new_exp, value);
        }

        TaylorModel {
            polynomial: new_poly,
            remainder: self.remainder,
            domain: self.domain[..new_dim].to_vec(),
            order: self.order,
        }
    }

    pub fn extend_with_time(&self, h: f64) -> TaylorModel {
        let new_dim = self.polynomial.dimension + 1;
        let mut new_poly = Polynomial::new(new_dim, self.order);
        for (exponents, coeff) in self.polynomial.terms() {
            let mut new_exp = exponents.clone();
            new_exp.push(0);
            new_poly.set(&new_exp, *coeff);
        }

        let mut new_domain = self.domain.clone();
        new_domain.push(interval!(0.0, h).unwrap());

        TaylorModel {
            polynomial: new_poly,
            remainder: self.remainder,
            domain: new_domain,
            order: self.order,
        }
    }
}

impl Add for TaylorModel {
    type Output = TaylorModel;
    fn add(self, other: TaylorModel) -> TaylorModel {
        assert_eq!(self.domain, other.domain);
        assert_eq!(self.order, other.order);
        let result = TaylorModel {
            polynomial: self.polynomial + other.polynomial,
            remainder: self.remainder + other.remainder,
            domain: self.domain,
            order: self.order,
        };
        result.sparsify(SPARSITY_THRESHOLD)
    }
}

impl Sub for TaylorModel {
    type Output = TaylorModel;
    fn sub(self, other: TaylorModel) -> TaylorModel {
        assert_eq!(self.domain, other.domain);
        assert_eq!(self.order, other.order);
        let result = TaylorModel {
            polynomial: self.polynomial - other.polynomial,
            remainder: self.remainder - other.remainder,
            domain: self.domain,
            order: self.order,
        };
        result.sparsify(SPARSITY_THRESHOLD)
    }
}

impl Mul for TaylorModel {
    type Output = TaylorModel;
    fn mul(self, other: TaylorModel) -> TaylorModel {
        assert_eq!(self.domain, other.domain);
        assert_eq!(self.order, other.order);
        let order = self.order;

        let key = (
            PolyKey::from(&self.polynomial),
            PolyKey::from(&other.polynomial),
        );

        let (low, high_range, p_range, q_range) = MUL_CACHE.with(|cache| {
            if let Some(cached) = cache.borrow().get(&key) {
                return cached.clone();
            }
            let product = self.polynomial.clone() * other.polynomial.clone();
            let (low, high) = product.split(order);
            let high_range = high.evaluate(&self.domain);
            let p_range = self.polynomial.evaluate(&self.domain);
            let q_range = other.polynomial.evaluate(&self.domain);
            let entry = (low, high_range, p_range, q_range);
            cache.borrow_mut().insert(key, entry.clone());
            entry
        });

        // G1 = s(D-x0) + p(D-x0)*F + E*(q(D-x0)+F)
        let g1 =
            high_range + p_range * other.remainder + self.remainder * (q_range + other.remainder);
        // G2 = s(D-x0) + q(D-x0)*E + F*(p(D-x0)+E)
        let g2 =
            high_range + q_range * self.remainder + other.remainder * (p_range + self.remainder);

        // G := G1 ∩ G2 — both individually valid, so the intersection is
        // still sound and strictly tighter.
        let inter = g1.intersection(g2);
        let error = if inter.is_empty() {
            g1.convex_hull(g2) // <- la méthode existe bien
        } else {
            inter
        };
        let result = TaylorModel {
            polynomial: low,
            remainder: error,
            domain: self.domain,
            order,
        };
        result.sparsify(SPARSITY_THRESHOLD)
    }
}

impl fmt::Display for TaylorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = ["x", "y", "z", "t"];
        let terms = self.polynomial.terms();

        if terms.is_empty() {
            write!(f, "0")?;
        } else {
            let mut first = true;
            for (exponents, coefficient) in terms {
                if !first {
                    if *coefficient >= 0.0 {
                        write!(f, " + {:.6}", coefficient)?;
                    } else {
                        write!(f, " - {:.6}", coefficient.abs())?;
                    }
                } else {
                    write!(f, "{:.6}", coefficient)?;
                    first = false;
                }

                for (i, exponent) in exponents.iter().enumerate() {
                    if *exponent == 0 {
                        continue;
                    }
                    write!(f, "*{}", names[i])?;
                    if *exponent > 1 {
                        write!(f, "^{}", exponent)?;
                    }
                }
            }
        }

        write!(f, " + E, E = {}", self.remainder)
    }
}

impl Polynomial {
    /// Substitutes a Taylor-model vector into self's variables (TM arithmetic).
    pub fn compose_tm(&self, args: &[TaylorModel]) -> TaylorModel {
        assert_eq!(args.len(), self.dimension);
        let dim = args[0].polynomial.dimension;
        let order = args[0].order;
        let domain = args[0].domain.clone();

        let mut result = TaylorModel::constant(0.0, dim, order, domain.clone());
        for (exponents, coeff) in self.terms() {
            let mut term = TaylorModel::constant(*coeff, dim, order, domain.clone());
            for (j, &e) in exponents.iter().enumerate() {
                if e > 0 {
                    term = term * args[j].powi(e);
                }
            }
            result = result + term;
        }
        result
    }
}

/// (left) ∘ (right) := left.polynomial(right) + left.remainder, per component.
pub fn compose(left: &[TaylorModel], right: &[TaylorModel]) -> Vec<TaylorModel> {
    left.iter()
        .map(|li| {
            let composed = li.polynomial.compose_tm(right);
            TaylorModel {
                remainder: composed.remainder + li.remainder,
                ..composed
            }
        })
        .collect()
}

impl TaylorModel {
    pub fn standard_domain(dim: usize) -> Vec<Interval> {
        (0..dim).map(|_| interval!(-1.0, 1.0).unwrap()).collect()
    }

    pub fn parameterized_initial_set(
        centers: &[f64],
        radii: &[f64],
        order: usize,
    ) -> Vec<TaylorModel> {
        assert_eq!(centers.len(), radii.len());

        let n = centers.len();
        let domain = TaylorModel::standard_domain(n);

        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let constant = TaylorModel::constant(centers[i], n, order, domain.clone());

            let variable = TaylorModel::variable(i, radii[i], n, order, domain.clone());

            result.push(constant + variable);
        }

        result
    }
}
