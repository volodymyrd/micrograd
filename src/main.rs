use crate::value::Value;
use crate::view::print_computation_graph;

mod value;
mod view;

fn main() {
    let a = Value::new(2.0).with_label("a");
    let b = Value::new(-3.0).with_label("b");
    let c = Value::new(10.0).with_label("c");
    let e = (a * b).with_label("e"); // 6.0
    let d = (e + c).with_label("d");
    let f = Value::new(-2.0).with_label("f");
    let l = (d * f).with_label("L");
    l.backward();
    println!("{}", print_computation_graph(&l, Some("micrograd.svg")));
}
