use crate::neuron::Neuron;
use crate::value::Value;

#[derive(Clone, Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();
        Self { neurons }
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn zero_grad(&self) {
        self.neurons.iter().for_each(|n| n.zero_grad());
    }

    pub fn update(&self, learning_rate: f64) {
        self.neurons.iter().for_each(|n| n.update(learning_rate));
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::layer::Layer;

    #[test]
    fn parameters() {
        for nin in 50..55 {
            for nout in 90..100 {
                assert_eq!(Layer::new(nin, nout).parameters().len(), nout * (nin + 1));
            }
        }
    }
}
