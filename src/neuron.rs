use crate::value::Value;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    activation: bool,
}

impl Neuron {
    pub fn new(nin: usize, activation: bool) -> Self {
        let mut rng = rand::rng();
        Self::new_internal(
            (0..nin)
                .map(|_| Value::new(rng.random_range(-1.0..1.0)))
                .collect(),
            Value::new(rng.random_range(-1.0..1.0)),
            activation,
        )
    }

    pub fn parameters(&self) -> Vec<Value> {
        [&self.weights[..], &[self.bias.clone()]].concat()
    }

    fn new_internal(weights: Vec<Value>, bias: Value, activation: bool) -> Self {
        Self {
            weights,
            bias,
            activation,
        }
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

        let z = (v + self.bias.clone().with_label("b")).with_label("z");
        if self.activation {
            z.tanh().with_label("a")
        } else {
            z.with_label("a")
        }
    }

    pub fn zero_grad(&self) {
        self.bias.zero_grad();
        self.weights.iter().for_each(|w| w.zero_grad());
    }

    pub fn update(&self, learning_rate: f64) {
        self.bias.update(learning_rate);
        self.weights.iter().for_each(|w| w.update(learning_rate));
    }
}

#[cfg(test)]
mod tests {
    use crate::neuron::Neuron;
    use crate::value::Value;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn rand() {
        let neuron = Neuron::new(2, true);
        for p in neuron.parameters() {
            assert!(
                p.data() > -1.0 && p.data() < 1.0,
                "Value {} is out of range!",
                p.data()
            );
        }
    }

    #[test]
    fn params() {
        let neuron = Neuron::new_internal(
            vec![Value::new(0.2), Value::new(-0.5)],
            Value::new(0.1),
            true,
        );

        assert_eq!(
            neuron
                .parameters()
                .iter()
                .map(|e| e.data())
                .collect::<Vec<_>>(),
            vec![0.2, -0.5, 0.1]
        );
    }

    #[test]
    fn forward() {
        let neuron = Neuron::new_internal(
            vec![Value::new(0.2), Value::new(-0.5)],
            Value::new(0.1),
            true,
        );
        let x = vec![Value::new(0.3), Value::new(0.7)]; // Input values matching the mock!
        let expected_output = (0.2f64 * 0.3f64 + (-0.5f64) * 0.7f64 + 0.1f64).tanh();

        let output = neuron.forward(&x);

        assert_approx_eq!(output.data(), expected_output, 1e-6);
    }
}
