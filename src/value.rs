use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{Add, Mul};
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

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let data = self.0.borrow().data + rhs.0.borrow().data;
        Self::new_internal(data, 0.0, vec![self, rhs], None, Some(String::from("+")))
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.0.borrow().data * rhs.0.borrow().data;
        Self::new_internal(data, 0.0, vec![self, rhs], None, Some(String::from("*")))
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
            uuid: uuid::Uuid::new_v4(),
            data,
            grad,
            prev,
            label,
            op,
        }
    }
}

#[cfg(test)]
mod tests {}
