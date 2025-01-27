use crate::value::Value;

mod value;

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(-3.0);
    let c = Value::new(10.0);
    let d = a * b + c;
    println!("{a} * {b} + {c} = {d}");
}
