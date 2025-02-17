use std::cell::RefCell;
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{Add, Mul};
use std::rc::Rc;

#[derive(Clone)]
pub struct Value {
    pub label: String,
    pub uuid: uuid::Uuid,
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<Rc<RefCell<Value>>>,
    pub op: Option<char>,
}

impl Value {
    pub fn new(
        data: f64,
        label: String,
        grad: f64,
        prev: Vec<Rc<RefCell<Value>>>,
        op: Option<char>,
    ) -> Self {
        Self {
            label,
            uuid: uuid::Uuid::new_v4(),
            data,
            grad,
            prev,
            op,
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::new(0.0, "".to_string(), 0.0, vec![], None)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}:{}", self.label, self.data)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "data={}", self.data)
    }
}

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.data + rhs.data,
            "".to_string(),
            1.0,
            vec![Rc::new(RefCell::new(self)), Rc::new(RefCell::new(rhs))]
                .into_iter()
                .collect(),
            Some('+'),
        )
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.data * rhs.data,
            "".to_string(),
            1.0,
            vec![Rc::new(RefCell::new(self)), Rc::new(RefCell::new(rhs))]
                .into_iter()
                .collect(),
            Some('*'),
        )
    }
}

#[cfg(test)]
mod tests {}
