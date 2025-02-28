use crate::layer::Layer;
use crate::value::Value;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug)]
pub struct Mlp {
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct MlpStat {
    num_layers: usize,
    num_neurons: usize,
    num_weights: usize,
    num_biases: usize,
    num_parameters: usize,
}

impl Display for MlpStat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP Statistics:")?;
        writeln!(f, "  Number of Layers: {}", self.num_layers)?;
        writeln!(f, "  Number of Neurons: {}", self.num_neurons)?;
        writeln!(f, "  Number of Weights: {}", self.num_weights)?;
        writeln!(f, "  Number of Biases: {}", self.num_biases)?;
        writeln!(f, "  Total Number of Parameters: {}", self.num_parameters)?;
        Ok(())
    }
}

impl Mlp {
    pub fn new(nin: usize, nouts: Vec<usize>, activation_last_layer: bool) -> Self {
        let sz = [&[nin], &nouts[..]].concat();
        let layers = (0..nouts.len())
            .map(|i| {
                Layer::new(
                    sz[i],
                    sz[i + 1],
                    activation_last_layer || i != nouts.len() - 1,
                )
            })
            .collect();
        Self { layers }
    }

    pub fn zero_grad(&self) {
        self.layers.iter().for_each(|l| l.zero_grad());
    }

    pub fn update(&self, learning_rate: f64) {
        self.layers.iter().for_each(|l| l.update(learning_rate));
    }

    pub fn train(&self, xs: Vec<Vec<f64>>, ys: Vec<f64>, n: usize, learning_rate: f64) {
        let xs: Vec<Vec<Value>> = xs
            .into_iter()
            .map(|x| {
                x.into_iter()
                    .map(|e| Value::new(e).with_label("XS"))
                    .collect()
            })
            .collect();

        for _ in 0..n {
            // forward pass
            let ypred: Vec<Value> = xs
                .iter()
                .map(|x| self.forward(x.clone())[0].clone())
                .collect();
            let loss: Value = ys
                .iter()
                .map(|y| Value::new(*y).with_label("Y"))
                .zip(ypred)
                .map(|(yout, ygt)| (yout - ygt).pow(&Value::new(2.0)))
                .sum();

            // backward pass
            self.zero_grad();
            loss.backward();

            // update
            self.update(learning_rate);

            println!("loss: {}", loss.data());
        }
    }

    pub fn forward(&self, mut x: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    pub fn stat(&self) -> MlpStat {
        let num_layers = self.layers.len();
        let mut num_neurons = 0;
        let mut num_weights = 0;
        let mut num_biases = 0;

        let mut params = 0;
        for layer in &self.layers {
            num_neurons += layer.len();
            let layer_params = layer.parameters();
            num_weights += layer_params.len() - layer.len();
            num_biases += layer.len();
            params += layer_params.len();
        }
        MlpStat {
            num_layers,
            num_neurons,
            num_weights,
            num_biases,
            num_parameters: params,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_new() {
        let mlp = Mlp::new(3, vec![4, 4, 1], true);
        assert_eq!(mlp.layers.len(), 3);
        assert_eq!(mlp.layers[0].len(), 4);
        assert_eq!(mlp.layers[1].len(), 4);
        assert_eq!(mlp.layers[2].len(), 1);
    }

    #[test]
    fn test_mlp_forward() {
        let mlp = Mlp::new(3, vec![4, 4, 1], true);
        let input = vec![Value::new(0.1), Value::new(0.2), Value::new(0.3)];
        let output = mlp.forward(input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_mlp_forward_with_different_dimensions() {
        let mlp = Mlp::new(2, vec![3, 1], true);
        let input = vec![Value::new(0.5), Value::new(0.8)];
        let output = mlp.forward(input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_mlp_train() {
        let mlp = Mlp::new(3, vec![4, 4, 1], true);
        let xs = vec![
            vec![2.0, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];
        let ys = vec![1.0, -1.0, -1.0, 1.0];
        mlp.train(xs, ys, 10, 0.01);
    }

    #[test]
    fn test_train_small_dataset() {
        let mlp = Mlp::new(2, vec![3, 1], true);
        let xs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let ys = vec![0.0, 1.0, 1.0, 0.0];
        mlp.train(xs, ys, 100, 0.1);
    }

    #[test]
    fn test_stat() {
        let mlp = Mlp::new(3, vec![4, 4, 1], true);
        let stat = mlp.stat();
        assert_eq!(stat.num_layers, 3);
        assert_eq!(stat.num_neurons, 9);
        assert_eq!(stat.num_biases, 9);
        // 12 weights in the first layer + 16 weights in the second + 4 weights in the third
        assert_eq!(stat.num_weights, 32);
        // 32 weights + 9 biases
        assert_eq!(stat.num_parameters, 41);

        let mlp = Mlp::new(2, vec![3, 1], true);
        let stat = mlp.stat();
        assert_eq!(stat.num_layers, 2);
        assert_eq!(stat.num_neurons, 4);
        assert_eq!(stat.num_biases, 4);
        assert_eq!(stat.num_weights, 9);
        assert_eq!(stat.num_parameters, 13);
    }
}
