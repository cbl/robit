use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, Mul, SubAssign},
};

use num_traits::{Signed, Zero};

use crate::{layers::Layer, losses::Loss};

use super::Optimizer;

pub struct SGD<T = f64> {
    learning_rate: T,
}

impl<T> Optimizer<T> for SGD<T>
where
    T: Copy
        + Zero
        + AddAssign
        + SubAssign
        + From<i32>
        + Div<T, Output = T>
        + Mul<Output = T>
        + Debug
        + Signed,
{
    fn minimize<const M: usize, const N: usize, L>(
        &self,
        weights: &mut Vec<T>,
        biases: &mut Vec<T>,
        training_data: &Vec<([T; M], [T; N])>,
        layer: &dyn Layer<M, N, T>,
        loss: &L,
    ) where
        L: Loss<T>,
    {
        let n_samples = training_data.len();

        training_data.chunks(16).into_iter().for_each(|batch| {
            let mut error = T::zero();

            for (x_train, y_train) in batch {
                let y_pred = layer.predict(weights, biases, x_train);
                error += loss.get(y_train, &y_pred).abs();
            }

            let avg_error: T = error / T::from(16);

            weights
                .iter_mut()
                .for_each(|w| *w += avg_error * self.learning_rate);
            biases
                .iter_mut()
                .for_each(|b| *b += avg_error * self.learning_rate);

            let mut error = T::zero();

            for (x_train, y_train) in batch {
                let y_pred = layer.predict(weights, biases, x_train);
                error += loss.get(y_train, &y_pred).abs();
            }

            let avg_error_2: T = error / T::from(16);

            println!("improved: {:?}", avg_error - avg_error_2);
        });
    }
}

impl Default for SGD<f64> {
    fn default() -> Self {
        Self {
            learning_rate: 0.0025,
        }
    }
}

impl Default for SGD<f32> {
    fn default() -> Self {
        Self {
            learning_rate: 0.0005,
        }
    }
}
