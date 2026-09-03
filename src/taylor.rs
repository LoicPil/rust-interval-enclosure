use inari::{Interval, interval};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul, Sub};

pub const SPARSITY_THRESHOLD: f64 = 1e-16;

thread_local! {
    static CURRENT_SPARSITY_THRESHOLD: RefCell<f64> = RefCell::new(SPARSITY_THRESHOLD);
}

pub fn set_sparsity_threshold(threshold: f64) {
    CURRENT_SPARSITY_THRESHOLD.with(|cell| *cell.borrow_mut() = threshold);
}

pub fn get_sparsity_threshold() -> f64 {
    CURRENT_SPARSITY_THRESHOLD.with(|cell| *cell.borrow())
}

// -----------------------------------------------------------------------------
// Polynomial
// -----------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct Polynomial {
    coeffs: HashMap<usize, f64>,
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

    fn decode(&self, mut idx: usize) -> Vec<usize> {
        let base = self.degree + 1;
        let mut exponents = vec![0; self.dimension];
        for i in 0..self.dimension {
            exponents[i] = idx % base;
            idx /= base;
        }
        exponents
    }

    #[inline]
    pub fn get(&self, exponents: &[usize]) -> f64 {
        *self.coeffs.get(&self.index(exponents)).unwrap_or(&0.0)
    }

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
        for (&idx, &coefficient) in &self.coeffs {
            let exponents = self.decode(idx);
            let mut term = interval!(coefficient, coefficient).unwrap();
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

    pub fn terms(&self) -> Vec<(Vec<usize>, f64)> {
        let mut terms: Vec<_> = self
            .coeffs
            .iter()
            .map(|(&idx, &c)| (self.decode(idx), c))
            .collect();
        terms.sort_by_key(|(e, _)| Self::total_degree(e));
        terms
    }

    pub fn sparsify(&mut self, domain: &[Interval], threshold: f64) -> Interval {
        assert_eq!(domain.len(), self.dimension);
        if threshold.is_infinite() || threshold <= 0.0 {
            return interval!(0.0, 0.0).unwrap();
        }

        let to_remove: Vec<usize> = self
            .coeffs
            .iter()
            .filter(|&(_, &c)| c.abs() < threshold)
            .map(|(&idx, _)| idx)
            .collect();

        let mut dropped = interval!(0.0, 0.0).unwrap();
        for idx in to_remove {
            let coeff = self.coeffs.remove(&idx).unwrap();
            let exponents = self.decode(idx);
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
        for (&idx, &coefficient) in other.coeffs.iter() {
            let exponents = other.decode(idx);
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
        for (&idx, &coefficient) in other.coeffs.iter() {
            let exponents = other.decode(idx);
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

        let base_in = self.degree + 1;
        let base_out = result.degree + 1;

        fn combine_idx(
            idx1: usize,
            idx2: usize,
            base_in: usize,
            base_out: usize,
            dim: usize,
        ) -> usize {
            let mut idx = 0;
            let mut pow = 1;
            let mut a = idx1;
            let mut b = idx2;
            for _ in 0..dim {
                let e1 = a % base_in;
                let e2 = b % base_in;
                idx += (e1 + e2) * pow;
                pow *= base_out;
                a /= base_in;
                b /= base_in;
            }
            idx
        }

        for (&idx1, &c1) in self.coeffs.iter() {
            for (&idx2, &c2) in other.coeffs.iter() {
                let new_idx = combine_idx(idx1, idx2, base_in, base_out, self.dimension);
                let value = result.coeffs.get(&new_idx).unwrap_or(&0.0) + c1 * c2;
                result.coeffs.insert(new_idx, value);
            }
        }
        result
    }
}

// -----------------------------------------------------------------------------
// Caches
// -----------------------------------------------------------------------------
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
    static MUL_CACHE: RefCell<HashMap<(PolyKey, PolyKey), (Polynomial, Interval, Interval, Interval)>> =
        RefCell::new(HashMap::new());
    static INTEGRATE_CACHE: RefCell<HashMap<PolyKey, (Polynomial, Interval)>> =
        RefCell::new(HashMap::new());
}

pub fn clear_caches() {
    MUL_CACHE.with(|c| c.borrow_mut().clear());
    INTEGRATE_CACHE.with(|c| c.borrow_mut().clear());
}

// -----------------------------------------------------------------------------
// TaylorModel
// -----------------------------------------------------------------------------
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

    /// Sparsifie avec le seuil passé ; si seuil infini ou négatif, ne fait rien.
    pub fn sparsify(mut self, threshold: f64) -> TaylorModel {
        if threshold.is_infinite() || threshold <= 0.0 {
            return self;
        }
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
                let value = antideriv.get(&new_exp) + coeff / k;
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
        result.sparsify(get_sparsity_threshold()) // utilise le seuil courant
    }

    pub fn substitute_time(&self, t: f64) -> TaylorModel {
        let time_var = self.polynomial.dimension - 1;
        let new_dim = time_var;
        let mut new_poly = Polynomial::new(new_dim, self.order);
        for (exponents, coeff) in self.polynomial.terms() {
            let factor = t.powi(exponents[time_var] as i32);
            let new_exp = exponents[..new_dim].to_vec();
            let value = new_poly.get(&new_exp) + coeff * factor;
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
        let result = TaylorModel {
            polynomial: self.polynomial + other.polynomial,
            remainder: self.remainder + other.remainder,
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
        let result = TaylorModel {
            polynomial: self.polynomial - other.polynomial,
            remainder: self.remainder - other.remainder,
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

        let g1 =
            high_range + p_range * other.remainder + self.remainder * (q_range + other.remainder);
        let g2 =
            high_range + q_range * self.remainder + other.remainder * (p_range + self.remainder);

        let inter = g1.intersection(g2);
        let error = if inter.is_empty() {
            g1.convex_hull(g2)
        } else {
            inter
        };
        let result = TaylorModel {
            polynomial: low,
            remainder: error,
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
