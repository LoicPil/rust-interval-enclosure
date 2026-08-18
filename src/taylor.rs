use inari::{interval,Interval};
use std::ops::Mul;
use std::ops::Add;

#[derive(Clone, Debug)]
struc  Taylor{
    coeffs: Vec<Interval>,
}

impl Taylor{
    pub fn constant(value:Interval, order:usize)-> self{
        let zero = interval!(0.0,0.0).unwrap();

        let mut coeffs= vec![zero, order+1];
        coeffs[0]= value;

        self{coeffs}
    }
    pub fn variable(x0: Interval, order:usize) -> self{
        let zero = interval!(0.0, 0.0).unwrap();

        let mut coeffs = vec![zero; order + 1];

        coeffs[0] = x0;
        coeffs[1] = interval!(1.0, 1.0).unwrap();

        Self { coeffs }
    }
}
impl Add for Taylor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.coeffs.len(), rhs.coeffs.len());

        let coeffs = self
            .coeffs
            .into_iter()
            .zip(rhs.coeffs)
            .map(|(a, b)| a + b)
            .collect();

        Self { coeffs }
    }
}

impl Mul for Taylor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        assert_eq!(self.coeffs.len(), rhs.coeffs.len());

        let n = self.coeffs.len();
        let zero = interval!(0.0, 0.0).unwrap();

        let mut coeffs = vec![zero; n];

        for k in 0..n {
            for i in 0..=k {
                coeffs[k] =
                    coeffs[k]
                        + self.coeffs[i] * rhs.coeffs[k - i];
            }
        }

        Self { coeffs }
    }
}
