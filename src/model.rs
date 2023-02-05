use std::{fmt::Debug};

use crate::{
    layers::{Input, Layer, Pipe},
    losses::{Loss, MeanSquaredError},
    optimizers::{Optimizer, SGD},
};

pub struct Model<const M: usize, const N: usize, O = SGD, L = MeanSquaredError, T = f64>
where
    O: Optimizer,
    L: Loss<T>,
{
    weights: Vec<T>,
    biases: Vec<T>,
    layers: Box<dyn Layer<M, N, T>>,
    optimizer: O,
    loss: L,
}

impl<const M: usize, O, L, T> Model<M, M, O, L, T>
where
    O: Optimizer + Default,
    L: Loss<T> + Default,
    T: 'static,
{
    pub fn new() -> Self {
        Self {
            weights: vec![],
            biases: vec![],
            layers: Box::new(Input::new()),
            optimizer: O::default(),
            loss: L::default(),
        }
    }
}

impl<const M: usize, const N: usize, O, L, T> Model<M, N, O, L, T>
where
    O: Optimizer,
    L: Loss<T>,
    T: 'static + Debug,
{
    /// Appends a layer and returns the new shaped model.
    ///
    /// ```
    /// use robit::{Model, layers::Dense};
    ///
    /// let model: Model<2, 4> = Model::new()
    ///     .add_layer(Dense::<2, 3>::new())
    ///     .add_layer(Dense::<3, 4>::new());
    /// ```
    pub fn add_layer<const U: usize, V: Layer<N, U, T> + 'static>(
        self,
        layer: V,
    ) -> Model<M, U, O, L, T> {
        self.weights.append(layer.weights());

        Model {
            weights: self.weights,
            biases: self.biases,
            layers: Box::new(Pipe::new(self.layers.into(), Box::new(layer))),
            optimizer: self.optimizer,
            loss: self.loss
        }
    }

    pub fn predict(&self, input: [T; M]) -> [T; N] {
        self.layers.forward(input)
    }

    pub fn train(&mut self, training_data: Vec<([T; M], [T; N])>) {
        for epoch in 0..5 {
            let weights = self.layers.weights_mut();

            self.optimizer.train(weights, training_data);
            
            // let biases = self.layers.biases_mut();

            println!("{:?}", weights);

            // self.optimizer.train(training_data);
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::layers::Dense;

    use super::*;

    #[test]
    fn train_model() {
        let mut model: Model<1, 2> = Model::new().add_layer(Dense::<1, 2>::new());

        let training_data = vec![([0.0; 1], [0.0; 2]), ([1.0; 1], [1.0; 2])];

        model.train(training_data);
    }
}
