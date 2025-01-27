use std::fmt::{Display, Formatter, Result};
use std::ops::{Add, Mul};

#[derive(Clone, Copy)]
pub struct Value<T> {
    data: T,
}

impl Value<f64> {
    pub fn new(data: f64) -> Self {
        Self { data }
    }
}

impl Display for Value<f64> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "data={}", self.data)
    }
}

impl Add for Value<f64> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.data + rhs.data)
    }
}

impl Mul for Value<f64> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.data * rhs.data)
    }
}
