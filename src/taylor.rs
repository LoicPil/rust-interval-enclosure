use inari::{Interval, interval};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul, Sub};

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

    pub fn set(&mut self, exponents: &[usize], value: f64) {
        assert_eq!(exponents.len(), self.dimension);
        if value.abs() < 1e-14 {
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

    /// Splits self into (low, high): low keeps all terms of total degree ≤ order,
    /// high keeps the rest. Used both for TM multiplication and integration.
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

    /// K's integral term: antiderivative of self wrt the time variable (last
    /// coordinate), from the (shifted) center 0 up to the variable t itself.
    /// Corresponds to the general TM-integration formula, Bünger sec. 3.1.
    pub fn integrate_time(&self, _t0: f64) -> TaylorModel {
        let time_var = self.polynomial.dimension - 1;

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
        let time_range = self.domain[time_var];
        let new_remainder = s_range + self.remainder * time_range;

        TaylorModel {
            polynomial: r,
            remainder: new_remainder,
            domain: self.domain.clone(),
            order: self.order,
        }
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
        TaylorModel {
            polynomial: self.polynomial + other.polynomial,
            remainder: self.remainder + other.remainder,
            domain: self.domain,
            order: self.order,
        }
    }
}

impl Sub for TaylorModel {
    type Output = TaylorModel;
    fn sub(self, other: TaylorModel) -> TaylorModel {
        assert_eq!(self.domain, other.domain);
        assert_eq!(self.order, other.order);
        TaylorModel {
            polynomial: self.polynomial - other.polynomial,
            remainder: self.remainder - other.remainder,
            domain: self.domain,
            order: self.order,
        }
    }
}

impl Mul for TaylorModel {
    type Output = TaylorModel;
    fn mul(self, other: TaylorModel) -> TaylorModel {
        assert_eq!(self.domain, other.domain);
        assert_eq!(self.order, other.order);
        let order = self.order;

        let product = self.polynomial.clone() * other.polynomial.clone();
        let (low, high) = product.split(order);

        let high_range = high.evaluate(&self.domain);
        let p_range = self.polynomial.evaluate(&self.domain);
        let q_range = other.polynomial.evaluate(&self.domain);

        let error = high_range
            + p_range * other.remainder
            + q_range * self.remainder
            + self.remainder * other.remainder;

        TaylorModel {
            polynomial: low,
            remainder: error,
            domain: self.domain,
            order,
        }
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
