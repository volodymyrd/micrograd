use std::collections::HashSet;
use crate::value::{print_computation_graph, Value};
use std::rc::Rc;

mod value;

fn main() {
    let a = Value::new(2.0, "a".to_string(), 0.0, HashSet::new(), None);
    let b = Value::new(-3.0, "b".to_string(), 0.0, HashSet::new(), None);
    let c = Value::new(10.0, "c".to_string(), 0.0, HashSet::new(), None);
    let mut e = a * b; // 6.0
    e.label = "e".to_string();
    let mut d = e + c;
    d.label = "d".to_string();
    let f = Value::new(-2.0, "f".to_string(), 0.0, HashSet::new(), None);
    let mut l = d * f;
    l.label = "L".to_string();
    println!(
        "{}",
        print_computation_graph(Rc::new(l), Some("micrograd.svg"))
    );
}
