use crate::value::Value;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::rng();
        Self::new_internal(
            (0..nin)
                .map(|_| Value::new(rng.random_range(-1.0..1.0)))
                .collect(),
            Value::new(rng.random_range(-1.0..1.0)),
        )
    }

    fn new_internal(weights: Vec<Value>, bias: Value) -> Self {
        Self { weights, bias }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        let v: Value = self
            .weights
            .iter()
            .zip(x.iter())
            .enumerate()
            .map(|(i, (wi, xi))| {
                (wi.clone().with_label(&format!("w{}", i))
                    * xi.clone().with_label(&format!("x{}", i)))
                .with_label(&format!("y{}", i))
            })
            .sum();

        (v + self.bias.clone().with_label("b"))
            .with_label("z")
            .tanh()
            .with_label("a")
    }
}

#[cfg(test)]
mod tests {
    use crate::neuron::Neuron;
    use crate::value::Value;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn forward() {
        let neuron = Neuron::new_internal(vec![Value::new(0.2), Value::new(-0.5)], Value::new(0.1));
        let x = vec![Value::new(0.3), Value::new(0.7)]; // Input values matching the mock!
        let expected_output = (0.2f64 * 0.3f64 + (-0.5f64) * 0.7f64 + 0.1f64).tanh();

        let output = neuron.forward(&x);

        assert_approx_eq!(output.data(), expected_output, 1e-6);
    }
}
