use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;
use uuid::Uuid;

#[derive(Clone)]
pub struct Value(Rc<RefCell<InternalValue>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Value::new_internal(data, 0.0, vec![], None, None)
    }

    fn new_internal(
        data: f64,
        grad: f64,
        prev: Vec<Value>,
        label: Option<String>,
        op: Option<String>,
    ) -> Self {
        Self(Rc::new(RefCell::new(InternalValue::new(
            data, grad, prev, label, op,
        ))))
    }

    pub fn with_label(self, label: &str) -> Value {
        self.0.borrow_mut().label = Some(label.to_string());
        self
    }

    pub fn tanh(&self) -> Self {
        let data = self.0.borrow().data.tanh();
        let lhs_internal = Rc::clone(&self.0);

        let out = Self::new_internal(
            data,
            0.0,
            vec![Value(lhs_internal)],
            None,
            Some(String::from("tanh")),
        );

        let lhs_internal = Rc::clone(&self.0);
        let out_internal = Rc::clone(&out.0);

        let backward = move || {
            let mut lhs = lhs_internal.borrow_mut();
            let out_grad = out_internal.borrow().grad;
            lhs.grad += (1.0 - data.powf(2.0)) * out_grad;
        };

        let out_internal = Rc::clone(&out.0);
        let mut out_internal_mut = out_internal.borrow_mut();
        out_internal_mut.backward = Some(Rc::new(RefCell::new(backward)));
        out
    }

    pub fn backward(&self) {
        let mut topo = vec![];
        let mut visited = HashSet::new();
        build_topo(self, &mut topo, &mut visited);

        self.0.borrow_mut().grad = 1.0;
        for node in topo.iter().rev() {
            if let Some(backward) = &node.0.borrow().backward {
                backward.borrow_mut()();
            }
        }
    }

    pub fn uuid(&self) -> Uuid {
        self.0.borrow().uuid
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn label(&self) -> String {
        if let Some(label) = &self.0.borrow().label {
            label.clone()
        } else {
            "".to_string()
        }
    }

    pub fn op(&self) -> Option<String> {
        self.0.borrow().op.clone()
    }

    /// Build a set of all nodes and edges in a graph.
    pub fn trace(&self) -> (Vec<RcDataValue>, Vec<(RcDataValue, RcDataValue)>) {
        let mut nodes = vec![];
        let mut edges = vec![];
        let mut visited = HashSet::new();
        fn build(
            v: &Value,
            nodes: &mut Vec<RcDataValue>,
            edges: &mut Vec<(RcDataValue, RcDataValue)>,
            visited: &mut HashSet<Uuid>,
        ) {
            let data_val_ref = Rc::new(DataValue::from(v));
            if !visited.contains(&data_val_ref.uuid) {
                visited.insert(data_val_ref.uuid);
                nodes.push(Rc::clone(&data_val_ref));
                for child in &v.0.borrow().prev {
                    let child_data_val_ref = Rc::new(DataValue::from(child));
                    edges.push((Rc::clone(&child_data_val_ref), Rc::clone(&data_val_ref)));
                    build(child, nodes, edges, visited);
                }
            }
        }
        build(self, &mut nodes, &mut edges, &mut visited);
        (nodes, edges)
    }
}

fn build_topo(v: &Value, topo: &mut Vec<Value>, visited: &mut HashSet<Uuid>) {
    if !visited.contains(&v.uuid()) {
        visited.insert(v.uuid());
        for child in &v.0.borrow().prev {
            build_topo(child, topo, visited);
        }
        topo.push(Value(Rc::clone(&v.0)));
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::new_internal(0.0, 0.0, vec![], None, None)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Display::fmt(self, f)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let int_val = &self.0.borrow();
        if let Some(ref l) = int_val.label {
            write!(f, "label: {}", l)?;
        }
        write!(f, "data: {}, grad: {}", int_val.data, int_val.grad)
    }
}
/// out = self + rfh.
///
/// self.grad = dL/d(self) = dL/d(out) * d(out)/d(self)
/// = out.grad * d(self + rhf)/d(self) = out.grad * (1+0) = out.grad
impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let is_self = Rc::ptr_eq(&self.0, &rhs.0);

        let data = self.0.borrow().data + rhs.0.borrow().data;
        let lhs_internal = Rc::clone(&self.0);
        let rhs_internal = Rc::clone(&rhs.0);

        let mut prev = vec![self];
        if !is_self {
            prev.push(rhs);
        }

        let out = Self::new_internal(data, 0.0, prev, None, Some(String::from("+")));
        let out_internal = Rc::clone(&out.0);

        let backward = move || {
            let mut lhs = lhs_internal.borrow_mut();
            let out_grad = out_internal.borrow().grad;
            lhs.grad += out_grad;

            if is_self {
                lhs.grad *= 2.0;
            } else {
                rhs_internal.borrow_mut().grad += out_grad;
            }
        };

        let out_internal = Rc::clone(&out.0);
        let mut out_internal_mut = out_internal.borrow_mut();
        out_internal_mut.backward = Some(Rc::new(RefCell::new(backward)));
        out
    }
}

/// out = self - rfh.
///
/// self.grad = dL/d(self) = dL/d(out) * d(out)/d(self)
/// = out.grad * d(self - rhf)/d(self) = out.grad * (1+0) = out.grad
///
/// rhs.grad = dL/d(rhs) = dL/d(out) * d(out)/d(rhs)
/// = out.grad * d(self - rhf)/d(rhs) = out.grad * (0-1) = -out.grad
impl Sub for Value {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let is_self = Rc::ptr_eq(&self.0, &rhs.0);

        let data = self.0.borrow().data - rhs.0.borrow().data;
        let lhs_internal = Rc::clone(&self.0);
        let rhs_internal = Rc::clone(&rhs.0);

        let mut prev = vec![self];
        if !is_self {
            prev.push(rhs);
        }

        let out = Self::new_internal(data, 0.0, prev, None, Some(String::from("-")));
        let out_internal = Rc::clone(&out.0);

        let backward = move || {
            let mut lhs = lhs_internal.borrow_mut();
            let out_grad = out_internal.borrow().grad;
            lhs.grad += out_grad;

            if is_self {
                lhs.grad *= 0.0;
            } else {
                rhs_internal.borrow_mut().grad -= out_grad;
            }
        };

        let out_internal = Rc::clone(&out.0);
        let mut out_internal_mut = out_internal.borrow_mut();
        out_internal_mut.backward = Some(Rc::new(RefCell::new(backward)));
        out
    }
}

/// out = self * rfh.
///
/// self.grad = dL/d(self) = dL/d(out) * d(out)/d(self)
/// = out.grad * d(self * rhf)/d(self) = out.grad * rhf
impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let is_self = Rc::ptr_eq(&self.0, &rhs.0);

        let data = self.0.borrow().data * rhs.0.borrow().data;
        let lhs_internal = Rc::clone(&self.0);
        let rhs_internal = Rc::clone(&rhs.0);

        let mut prev = vec![self];
        if !is_self {
            prev.push(rhs);
        }

        let out = Self::new_internal(data, 0.0, prev, None, Some(String::from("*")));
        let out_internal = Rc::clone(&out.0);

        let backward = move || {
            let mut lhs = lhs_internal.borrow_mut();
            let rhs_data = if is_self {
                lhs.data
            } else {
                rhs_internal.borrow_mut().data
            };

            let out_grad = out_internal.borrow().grad;
            lhs.grad += rhs_data * out_grad;

            if is_self {
                lhs.grad *= 2.0;
            } else {
                rhs_internal.borrow_mut().grad += lhs.data * out_grad;
            }
        };

        let out_internal = Rc::clone(&out.0);
        let mut out_internal_mut = out_internal.borrow_mut();
        out_internal_mut.backward = Some(Rc::new(RefCell::new(backward)));
        out
    }
}

type RcDataValue = Rc<DataValue>;

#[derive(Clone)]
pub struct DataValue {
    pub uuid: Uuid,
    pub data: f64,
    pub grad: f64,
    pub label: String,
    pub op: Option<String>,
}

impl DataValue {
    pub fn new(uuid: Uuid, data: f64, grad: f64, label: String, op: Option<String>) -> Self {
        Self {
            uuid,
            data,
            grad,
            label,
            op,
        }
    }
}

impl From<&Value> for DataValue {
    fn from(value: &Value) -> Self {
        DataValue::new(
            value.uuid(),
            value.data(),
            value.grad(),
            value.label(),
            value.op(),
        )
    }
}

#[derive(Clone)]
struct InternalValue {
    uuid: Uuid,
    data: f64,
    grad: f64,
    prev: Vec<Value>,
    label: Option<String>,
    op: Option<String>,
    backward: Option<Rc<RefCell<dyn FnMut()>>>,
}

impl InternalValue {
    pub fn new(
        data: f64,
        grad: f64,
        prev: Vec<Value>,
        label: Option<String>,
        op: Option<String>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            data,
            grad,
            prev,
            label,
            op,
            backward: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::value::Value;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn final_grad() {
        let a = Value::new(3.0);
        assert_eq!(a.data(), 3.0);
        assert_eq!(a.grad(), 0.0);

        a.backward();
        assert_eq!(a.data(), 3.0);
        assert_eq!(a.grad(), 1.0);
    }

    #[test]
    fn add() {
        let a = Value::new(3.0);
        let b = Value::new(4.0);
        let c = a.clone() + b.clone();

        c.backward();

        assert_eq!(a.data(), 3.0);
        assert_eq!(a.grad(), 1.0);

        assert_eq!(b.data(), 4.0);
        assert_eq!(b.grad(), 1.0);

        assert_eq!(c.data(), 7.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn add_self() {
        let a = Value::new(3.0);
        let c = a.clone() + a.clone();

        c.backward();

        assert_eq!(a.data(), 3.0);
        assert_eq!(a.grad(), 2.0);

        assert_eq!(c.data(), 6.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn sub() {
        let a = Value::new(3.0);
        let b = Value::new(4.0);
        let c = a.clone() - b.clone();

        c.backward();

        assert_eq!(a.data(), 3.0);
        assert_eq!(a.grad(), 1.0);

        assert_eq!(b.data(), 4.0);
        assert_eq!(b.grad(), -1.0);

        assert_eq!(c.data(), -1.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn sub_self() {
        let a = Value::new(3.0);
        let c = a.clone() - a.clone();

        c.backward();

        assert_eq!(a.data(), 3.0);
        assert_eq!(a.grad(), 0.0);

        assert_eq!(c.data(), 0.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn mul() {
        let a = Value::new(3.0);
        let b = Value::new(4.0);
        let c = a.clone() * b.clone();

        c.backward();

        assert_eq!(a.data(), 3.0);
        assert_eq!(a.grad(), 4.0);

        assert_eq!(b.data(), 4.0);
        assert_eq!(b.grad(), 3.0);

        assert_eq!(c.data(), 12.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn mul_self() {
        let a = Value::new(3.0);
        let c = a.clone() * a.clone();

        c.backward();

        assert_eq!(a.data(), 3.0);
        assert_eq!(a.grad(), 6.0);

        assert_eq!(c.data(), 9.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn chain() {
        let a = Value::new(-2.0);
        let b = Value::new(3.0);
        let e = a.clone() + b.clone();
        let d = a.clone() * b.clone();
        let f = e.clone() * d.clone();

        f.backward();

        assert_eq!(a.data(), -2.0);
        assert_eq!(a.grad(), -3.0);

        assert_eq!(b.data(), 3.0);
        assert_eq!(b.grad(), -8.0);

        assert_eq!(e.data(), 1.0);
        assert_eq!(e.grad(), -6.0);

        assert_eq!(d.data(), -6.0);
        assert_eq!(d.grad(), 1.0);

        assert_eq!(f.data(), -6.0);
        assert_eq!(f.grad(), 1.0);
    }

    #[test]
    fn tanh() {
        let a = Value::new(0.8814);
        let c = a.tanh();

        c.backward();

        assert_eq!(a.data(), 0.8814);
        assert_approx_eq!(a.grad(), 0.5, 0.1);

        assert_approx_eq!(c.data(), 0.7071, 0.0001);
        assert_eq!(c.grad(), 1.0);
    }
}
