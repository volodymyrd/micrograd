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
    println!("{}", print_computation_graph(&l, Some("micrograd1.svg")));

    let x1 = Value::new(2.0).with_label("x1");
    let x2 = Value::new(0.0).with_label("x2");
    let w1 = Value::new(-3.0).with_label("w1");
    let w2 = Value::new(1.0).with_label("w2");
    let b = Value::new(6.881_373_587_019_543).with_label("b");
    let x1w1 = (x1 * w1).with_label("x1*w1");
    let x2w2 = (x2 * w2).with_label("x2*w2");
    let x1w1x2w2 = (x1w1 + x2w2).with_label("x1*w1 + x2*w2");
    let n = (x1w1x2w2 + b).with_label("n");
    let o = n.tanh().with_label("o");
    o.backward();
    println!("{}", print_computation_graph(&o, Some("micrograd2.svg")));
}
