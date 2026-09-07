use inari::{Interval, interval};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul, Sub};

/// Default threshold for coefficient sparsification.
pub const SPARSITY_THRESHOLD: f64 = 1e-16;

/// Thread-local sparsity threshold.
thread_local! {
    static CURRENT_SPARSITY_THRESHOLD: RefCell<f64> = RefCell::new(SPARSITY_THRESHOLD);
}

/// Sets the global sparsity threshold.
pub fn set_sparsity_threshold(threshold: f64) {
    CURRENT_SPARSITY_THRESHOLD.with(|cell| *cell.borrow_mut() = threshold);
}

/// Gets the current sparsity threshold.
pub fn get_sparsity_threshold() -> f64 {
    CURRENT_SPARSITY_THRESHOLD.with(|cell| *cell.borrow())
}

/// Returns the zero interval [0, 0].
fn zero_iv() -> Interval {
    interval!(0.0, 0.0).unwrap()
}

/// Converts a scalar to a point interval [x, x].
fn pt(x: f64) -> Interval {
    interval!(x, x).unwrap()
}

/// Rounds an exact interval to a single f64 point and returns the residual.
///
/// # Invariant
/// `exact ⊆ pt(point) + residual`
#[inline]
fn round_to_point(exact: Interval) -> (f64, Interval) {
    let point = exact.mid();
    (point, exact - pt(point))
}

/// A multivariate polynomial with f64 coefficients.
///
/// Coefficients are stored sparsely in a hash map. Operations are performed
/// exactly using interval arithmetic, then rounded to f64 with residuals
/// tracked explicitly.
#[derive(Clone, Debug)]
pub struct Polynomial {
    coeffs: HashMap<usize, f64>,
    degree: usize,
    pub dimension: usize,
}

impl Polynomial {
    /// Creates a new zero polynomial with given dimension and degree.
    pub fn new(dimension: usize, degree: usize) -> Self {
        Self {
            coeffs: HashMap::new(),
            degree,
            dimension,
        }
    }

    /// Encodes exponents into a unique index.
    #[inline]
    fn index(&self, exponents: &[usize]) -> usize {
        let base = self.degree + 1;
        let mut idx = 0;
        let mut pow = 1;
        for i in 0..self.dimension {
            idx += exponents[i] * pow;
            pow *= base;
        }
        idx
    }

    /// Decodes an index back to exponents.
    fn decode(&self, mut idx: usize) -> Vec<usize> {
        let base = self.degree + 1;
        let mut exponents = vec![0; self.dimension];
        for i in 0..self.dimension {
            exponents[i] = idx % base;
            idx /= base;
        }
        exponents
    }

    /// Gets the coefficient for the given exponents.
    #[inline]
    pub fn get(&self, exponents: &[usize]) -> f64 {
        *self.coeffs.get(&self.index(exponents)).unwrap_or(&0.0)
    }

    /// Sets the coefficient for the given exponents.
    #[inline]
    pub fn set(&mut self, exponents: &[usize], value: f64) {
        assert_eq!(exponents.len(), self.dimension);
        assert!(
            value.is_finite(),
            "non-finite coefficient {} at {:?}",
            value,
            exponents
        );
        let idx = self.index(exponents);
        if value == 0.0 {
            self.coeffs.remove(&idx);
        } else {
            self.coeffs.insert(idx, value);
        }
    }

    /// Creates a constant polynomial.
    pub fn constant(dimension: usize, degree: usize, value: f64) -> Self {
        let mut p = Self::new(dimension, degree);
        p.set(&vec![0; dimension], value);
        p
    }

    /// Creates a linear monomial.
    pub fn variable(dimension: usize, degree: usize, variable: usize, coefficient: f64) -> Self {
        assert!(variable < dimension);
        let mut p = Self::new(dimension, degree);
        let mut exponents = vec![0; dimension];
        exponents[variable] = 1;
        p.set(&exponents, coefficient);
        p
    }

    /// Computes the total degree of exponents.
    pub fn total_degree(exponents: &[usize]) -> usize {
        exponents.iter().sum()
    }

    /// Evaluates the polynomial over an interval domain.
    pub fn evaluate(&self, domain: &[Interval]) -> Interval {
        assert_eq!(domain.len(), self.dimension);
        let mut result = zero_iv();
        for (&idx, &coefficient) in &self.coeffs {
            let exponents = self.decode(idx);
            let mut term = pt(coefficient);
            for (i, exponent) in exponents.iter().enumerate() {
                if *exponent > 0 {
                    term *= domain[i].powi(*exponent as i32);
                }
            }
            result += term;
        }
        result
    }

    /// Splits the polynomial into low and high degree parts.
    pub fn split(&self, order: usize) -> (Polynomial, Polynomial) {
        let mut low = Polynomial::new(self.dimension, order);
        let mut high = Polynomial::new(self.dimension, self.degree);
        for (&idx, &coefficient) in &self.coeffs {
            let exponents = self.decode(idx);
            if Self::total_degree(&exponents) <= order {
                low.set(&exponents, coefficient);
            } else {
                high.set(&exponents, coefficient);
            }
        }
        (low, high)
    }

    /// Evaluates the polynomial at a point (f64).
    pub fn sample(&self, point: &[f64]) -> f64 {
        assert_eq!(point.len(), self.dimension);
        let mut result = 0.0;
        for (&idx, &coefficient) in &self.coeffs {
            let exponents = self.decode(idx);
            let mut term = coefficient;
            for (x, exponent) in point.iter().zip(exponents.iter()) {
                term *= x.powi(*exponent as i32);
            }
            result += term;
        }
        result
    }

    /// Returns all terms as a vector of (exponents, coefficient).
    pub fn terms(&self) -> Vec<(Vec<usize>, f64)> {
        let mut terms: Vec<_> = self
            .coeffs
            .iter()
            .map(|(&idx, &c)| (self.decode(idx), c))
            .collect();
        terms.sort_by_key(|(e, _)| Self::total_degree(e));
        terms
    }

    /// Removes coefficients below the threshold and returns their contribution.
    pub fn sparsify(&mut self, domain: &[Interval], threshold: f64) -> Interval {
        assert_eq!(domain.len(), self.dimension);
        if threshold.is_infinite() || threshold <= 0.0 {
            return zero_iv();
        }

        let to_remove: Vec<usize> = self
            .coeffs
            .iter()
            .filter(|&(_, &c)| c.abs() < threshold)
            .map(|(&idx, _)| idx)
            .collect();

        let mut dropped = zero_iv();
        for idx in to_remove {
            let coeff = self.coeffs.remove(&idx).unwrap();
            let exponents = self.decode(idx);
            let mut term = pt(coeff);
            for (i, &exponent) in exponents.iter().enumerate() {
                if exponent > 0 {
                    term *= domain[i].powi(exponent as i32);
                }
            }
            dropped += term;
        }
        dropped
    }

    /// Addition with exact rounding error tracking.
    pub fn add_checked(mut self, other: Polynomial) -> (Polynomial, Interval) {
        assert_eq!(self.dimension, other.dimension);
        assert_eq!(self.degree, other.degree);
        let mut error = zero_iv();

        for (&idx, &c2) in other.coeffs.iter() {
            let exponents = other.decode(idx);
            let c1 = self.get(&exponents);
            let exact = pt(c1) + pt(c2);
            let (point, residual) = round_to_point(exact);
            error += residual;
            self.set(&exponents, point);
        }
        (self, error)
    }

    /// Subtraction with exact rounding error tracking.
    pub fn sub_checked(mut self, other: Polynomial) -> (Polynomial, Interval) {
        assert_eq!(self.dimension, other.dimension);
        assert_eq!(self.degree, other.degree);
        let mut error = zero_iv();

        for (&idx, &c2) in other.coeffs.iter() {
            let exponents = other.decode(idx);
            let c1 = self.get(&exponents);
            let exact = pt(c1) - pt(c2);
            let (point, residual) = round_to_point(exact);
            error += residual;
            self.set(&exponents, point);
        }
        (self, error)
    }

    /// Multiplication with exact rounding error tracking.
    ///
    /// Accumulates contributions exactly using interval arithmetic,
    /// then rounds each coefficient once at the end.
    pub fn mul_checked(&self, other: &Polynomial) -> (Polynomial, Interval) {
        assert_eq!(self.dimension, other.dimension);
        let degree = self.degree + other.degree;
        let base_in = self.degree + 1;
        let base_other = other.degree + 1;
        let base_out = degree + 1;

        let mut acc: HashMap<usize, Interval> = HashMap::new();

        for (&idx1, &c1) in self.coeffs.iter() {
            for (&idx2, &c2) in other.coeffs.iter() {
                let mut new_idx = 0;
                let mut pow = 1;
                let mut a = idx1;
                let mut b = idx2;
                for _ in 0..self.dimension {
                    let e1 = a % base_in;
                    let e2 = b % base_other;
                    new_idx += (e1 + e2) * pow;
                    pow *= base_out;
                    a /= base_in;
                    b /= base_other;
                }
                let term = pt(c1) * pt(c2);
                let current = acc.get(&new_idx).copied().unwrap_or_else(zero_iv);
                acc.insert(new_idx, current + term);
            }
        }

        let mut result = Polynomial::new(self.dimension, degree);
        let mut error = zero_iv();
        for (idx, exact) in acc {
            let (point, residual) = round_to_point(exact);
            error += residual;
            if point != 0.0 {
                result.coeffs.insert(idx, point);
            }
        }
        (result, error)
    }
}

impl Add for Polynomial {
    type Output = Polynomial;
    fn add(self, other: Polynomial) -> Polynomial {
        self.add_checked(other).0
    }
}

impl Sub for Polynomial {
    type Output = Polynomial;
    fn sub(self, other: Polynomial) -> Polynomial {
        self.sub_checked(other).0
    }
}

impl Mul for Polynomial {
    type Output = Polynomial;
    fn mul(self, other: Polynomial) -> Polynomial {
        self.mul_checked(&other).0
    }
}

/// Key for polynomial caching based on coefficients.
#[derive(Clone, PartialEq, Eq, Hash)]
struct PolyKey(Vec<(usize, u64)>);

impl PolyKey {
    fn from(p: &Polynomial) -> Self {
        let mut v: Vec<(usize, u64)> = p
            .coeffs
            .iter()
            .map(|(&idx, &c)| (idx, c.to_bits()))
            .collect();
        v.sort();
        PolyKey(v)
    }
}

thread_local! {
    /// Cache for multiplication results.
    static MUL_CACHE: RefCell<HashMap<(PolyKey, PolyKey), (Polynomial, Interval, Interval, Interval, Interval)>> =
        RefCell::new(HashMap::new());

    /// Cache for integration results.
    static INTEGRATE_CACHE: RefCell<HashMap<PolyKey, (Polynomial, Interval, Interval)>> =
        RefCell::new(HashMap::new());
}

/// Clears all caches.
pub fn clear_caches() {
    MUL_CACHE.with(|c| c.borrow_mut().clear());
    INTEGRATE_CACHE.with(|c| c.borrow_mut().clear());
}

/// A Taylor model consisting of a polynomial part and a remainder interval.
#[derive(Clone, Debug)]
pub struct TaylorModel {
    pub polynomial: Polynomial,
    pub remainder: Interval,
    pub domain: Vec<Interval>,
    pub order: usize,
}

impl TaylorModel {
    /// Creates a constant Taylor model.
    pub fn constant(value: f64, dimension: usize, order: usize, domain: Vec<Interval>) -> Self {
        Self {
            polynomial: Polynomial::constant(dimension, order, value),
            remainder: zero_iv(),
            domain,
            order,
        }
    }

    /// Creates a linear variable Taylor model.
    pub fn variable(
        variable: usize,
        coefficient: f64,
        dimension: usize,
        order: usize,
        domain: Vec<Interval>,
    ) -> Self {
        Self {
            polynomial: Polynomial::variable(dimension, order, variable, coefficient),
            remainder: zero_iv(),
            domain,
            order,
        }
    }

    /// Creates a Taylor model from an interval (constant polynomial, interval remainder).
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

    /// Computes the range (interval enclosure) of the Taylor model.
    pub fn range(&self) -> Interval {
        self.polynomial.evaluate(&self.domain) + self.remainder
    }

    /// Computes the range of the polynomial part only.
    pub fn polynomial_range(&self) -> Interval {
        self.polynomial.evaluate(&self.domain)
    }

    /// Sparsifies the polynomial part and merges dropped terms into the remainder.
    pub fn sparsify(mut self, threshold: f64) -> TaylorModel {
        if threshold.is_infinite() || threshold <= 0.0 {
            return self;
        }
        let dropped = self.polynomial.sparsify(&self.domain, threshold);
        self.remainder = self.remainder + dropped;
        self
    }

    /// Computes the power of the Taylor model.
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

    /// Samples the Taylor model at a point.
    pub fn sample(&self, point: &[f64]) -> f64 {
        self.polynomial.sample(point)
    }

    /// Integrates the Taylor model with respect to time.
    ///
    /// The antiderivative is computed exactly using interval arithmetic,
    /// then rounded with error tracking.
    pub fn integrate_time(&self) -> TaylorModel {
        let time_var = self.polynomial.dimension - 1;
        let key = PolyKey::from(&self.polynomial);

        let (r, s_range, round_err) = INTEGRATE_CACHE.with(|cache| {
            if let Some(cached) = cache.borrow().get(&key) {
                return cached.clone();
            }

            // Accumulate antiderivative exactly using interval arithmetic
            let mut acc: HashMap<usize, Interval> = HashMap::new();
            for (exponents, coeff) in self.polynomial.terms() {
                let mut new_exp = exponents.clone();
                new_exp[time_var] += 1;
                let k = new_exp[time_var] as f64;
                let key_idx = {
                    let tmp = Polynomial::new(self.polynomial.dimension, self.order + 1);
                    tmp.index_pub(&new_exp)
                };
                let term = pt(coeff) / pt(k);
                let current = acc.get(&key_idx).copied().unwrap_or_else(zero_iv);
                acc.insert(key_idx, current + term);
            }

            let mut antideriv = Polynomial::new(self.polynomial.dimension, self.order + 1);
            let mut round_err = zero_iv();
            for (idx, exact) in acc {
                let (point, residual) = round_to_point(exact);
                round_err += residual;
                if point != 0.0 {
                    antideriv.coeffs.insert(idx, point);
                }
            }

            let (r, s) = antideriv.split(self.order);
            let s_range = s.evaluate(&self.domain);
            let entry = (r, s_range, round_err);
            cache.borrow_mut().insert(key, entry.clone());
            entry
        });

        let time_range = self.domain[time_var];
        let new_remainder = s_range + self.remainder * time_range + round_err;

        let result = TaylorModel {
            polynomial: r,
            remainder: new_remainder,
            domain: self.domain.clone(),
            order: self.order,
        };
        result.sparsify(get_sparsity_threshold())
    }

    /// Substitutes time with a fixed value.
    pub fn substitute_time(&self, t: f64) -> TaylorModel {
        let time_var = self.polynomial.dimension - 1;
        let new_dim = time_var;
        let t_iv = pt(t);

        let mut acc: HashMap<usize, Interval> = HashMap::new();
        for (exponents, coeff) in self.polynomial.terms() {
            let factor = t_iv.powi(exponents[time_var] as i32);
            let new_exp = exponents[..new_dim].to_vec();
            let tmp = Polynomial::new(new_dim, self.order);
            let idx = tmp.index_pub(&new_exp);
            let term = pt(coeff) * factor;
            let current = acc.get(&idx).copied().unwrap_or_else(zero_iv);
            acc.insert(idx, current + term);
        }

        let mut new_poly = Polynomial::new(new_dim, self.order);
        let mut round_err = zero_iv();
        for (idx, exact) in acc {
            let (point, residual) = round_to_point(exact);
            round_err += residual;
            if point != 0.0 {
                new_poly.coeffs.insert(idx, point);
            }
        }

        TaylorModel {
            polynomial: new_poly,
            remainder: self.remainder + round_err,
            domain: self.domain[..new_dim].to_vec(),
            order: self.order,
        }
    }

    /// Extends the Taylor model with an extra time dimension.
    pub fn extend_with_time(&self, h: f64) -> TaylorModel {
        let new_dim = self.polynomial.dimension + 1;
        let mut new_poly = Polynomial::new(new_dim, self.order);
        for (exponents, coeff) in self.polynomial.terms() {
            let mut new_exp = exponents.clone();
            new_exp.push(0);
            new_poly.set(&new_exp, coeff);
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
        let (poly, round_err) = self.polynomial.add_checked(other.polynomial);
        let result = TaylorModel {
            polynomial: poly,
            remainder: self.remainder + other.remainder + round_err,
            domain: self.domain,
            order: self.order,
        };
        result.sparsify(get_sparsity_threshold())
    }
}

impl Sub for TaylorModel {
    type Output = TaylorModel;
    fn sub(self, other: TaylorModel) -> TaylorModel {
        assert_eq!(self.domain, other.domain);
        assert_eq!(self.order, other.order);
        let (poly, round_err) = self.polynomial.sub_checked(other.polynomial);
        let result = TaylorModel {
            polynomial: poly,
            remainder: self.remainder - other.remainder + round_err,
            domain: self.domain,
            order: self.order,
        };
        result.sparsify(get_sparsity_threshold())
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

        let (low, high_range, p_range, q_range, round_err) = MUL_CACHE.with(|cache| {
            if let Some(cached) = cache.borrow().get(&key) {
                return cached.clone();
            }
            let (product, round_err) = self.polynomial.mul_checked(&other.polynomial);
            let (low, high) = product.split(order);
            let high_range = high.evaluate(&self.domain);
            let p_range = self.polynomial.evaluate(&self.domain);
            let q_range = other.polynomial.evaluate(&self.domain);
            let entry = (low, high_range, p_range, q_range, round_err);
            cache.borrow_mut().insert(key, entry.clone());
            entry
        });

        // Compute error bounds using intersection or convex hull
        let g1 =
            high_range + p_range * other.remainder + self.remainder * (q_range + other.remainder);
        let g2 =
            high_range + q_range * self.remainder + other.remainder * (p_range + self.remainder);

        let inter = g1.intersection(g2);
        let base_error = if inter.is_empty() {
            g1.convex_hull(g2)
        } else {
            inter
        };

        let result = TaylorModel {
            polynomial: low,
            remainder: base_error + round_err,
            domain: self.domain,
            order,
        };
        result.sparsify(get_sparsity_threshold())
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
                    if coefficient >= 0.0 {
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
    /// Public version of `index` for use in other modules.
    pub fn index_pub(&self, exponents: &[usize]) -> usize {
        self.index(exponents)
    }

    /// Composes the polynomial with a vector of Taylor models.
    pub fn compose_tm(&self, args: &[TaylorModel]) -> TaylorModel {
        assert_eq!(args.len(), self.dimension);
        let dim = args[0].polynomial.dimension;
        let order = args[0].order;
        let domain = args[0].domain.clone();
        let mut result = TaylorModel::constant(0.0, dim, order, domain.clone());

        for (exponents, coeff) in self.terms() {
            let mut term = TaylorModel::constant(coeff, dim, order, domain.clone());
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

/// Composes a vector of Taylor models with another vector.
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
    /// Creates a standard domain [-1, 1]^dim.
    pub fn standard_domain(dim: usize) -> Vec<Interval> {
        (0..dim).map(|_| interval!(-1.0, 1.0).unwrap()).collect()
    }

    /// Creates a parameterized initial set from centers and radii.
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
