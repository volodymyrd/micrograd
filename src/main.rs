use crate::value::{print_computation_graph, Value};
use std::rc::Rc;

mod value;

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(-3.0);
    let c = Value::new(10.0);
    print!("{} * {} + {} = ", a, b, c);
    let d = a * b + c;
    println!("{d}");
    println!(
        "{}",
        print_computation_graph(Rc::new(d), Some("micrograd.svg"))
    );
}
