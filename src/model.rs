use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Div, Mul, Sub, SubAssign},
    process::Output,
};

use ndarray::{
    s, Array1, Array2, Axis, Ix1, Ix2, LinalgScalar,
    ScalarOperand
};
use num_traits::{FromPrimitive};

use crate::{
    activations::{Activation, Relu},
    initializers::Initializer,
    losses::{Loss, MeanSquaredError},
};

pub struct Model<A = Relu, L = MeanSquaredError, T = f64>
where
    A: Activation<T>,
    L: Loss<T>,
{
    weights: Vec<Array2<T>>,
    biases: Vec<Array1<T>>,
    activation: A,
    loss: L,
    batch_size: usize,
    learning_rate: T,
}

impl<A, L, T> Model<A, L, T>
where
    A: Activation<T> + Default,
    L: Loss<T> + Default,
    T: Copy + 'static,
{
    pub fn new(learning_rate: T) -> Self {
        Self {
            weights: vec![],
            biases: vec![],
            activation: A::default(),
            loss: L::default(),
            batch_size: 32,
            learning_rate,
        }
    }
}

impl<A, L, T> Model<A, L, T>
where
    A: Activation<T>,
    L: Loss<T>,
    T: Debug,
{
    pub fn add_layer<I: Initializer<T>>(&mut self, shape: (usize, usize), mut init: I) {
        self.weights.push(init.gen(shape));
        self.biases.push(init.gen(shape.1));
    }
}

impl<A, T, L> Model<A, L, T>
where
    A: Activation<T>,
    L: Loss<T>,
    T: LinalgScalar + PartialOrd + FromPrimitive + ScalarOperand + Mul<T> + Debug + SubAssign<T>,
{
    pub fn predict(&self, input: &Array2<T>) -> Array2<T> {
        let mut a = self
            .activation
            .call(&(input.dot(&self.weights[0]) + &self.biases[0]));

        for (w, b) in self.weights[1..].iter().zip(self.biases[1..].iter()) {
            a = self.activation.call(&(a.dot(w) + b));
        }

        a
    }

    pub fn fit(&mut self, X: &Array2<T>, Y: &Array2<T>) {
        let n_samples = X.shape()[0];
        let n_features = X.shape()[1];
        
        for i in (0..n_samples).step_by(self.batch_size) {
            if (i + self.batch_size) >= n_samples {
                continue;
            }

            let x_batch = X.slice(s![i..i + self.batch_size, ..]);
            let y_batch = Y.slice(s![i..i + self.batch_size, ..]);
            let y_pred = self.predict(&x_batch.to_owned());
            let error = self.loss.get(&y_batch.to_owned(), &y_pred);

            self.backpropagate(x_batch.to_owned(), error)

            // let y_pred = self.predict(x);
            // let error = y_pred - &y_batch;

            // self.backpropagate(&x, &error, 0.005);
        }
    }

    fn backpropagate(&mut self, X: Array2<T>, error: Array1<T>) {
        let mut activations = vec![X];
        let mut zs = vec![];

        // forward pass
        for i in 0..self.weights.len() {
            let z = activations[i].dot(&self.weights[i]) + &self.biases[i];
            activations.push(self.activation.call(&z));
            zs.push(z);
        }

        // println!("{:?}", error.sum());
        // println!("{:?}", error);

        // backward pass
        let mut delta = error * self.activation.call_deriv(&zs[zs.len() - 1]);

        for i in (0..self.weights.len()).rev() {
            let weights = self.weights[i].clone();

            let delta_bias = delta.mean_axis(Axis(0)).unwrap();
            let delta_weight = activations[i].t().dot(&delta);

            self.weights[i] -= &(delta_weight * self.learning_rate);
            self.biases[i] -= &(delta_bias * self.learning_rate);

            if i != 0 {
                delta = delta.dot(&weights.t()) * self.activation.call_deriv(&zs[i - 1]);
            }
        }
    }
}

#[cfg(test)]
mod tests {

    // todo ...
}
