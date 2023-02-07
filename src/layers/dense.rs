use std::{
    fmt::Debug,
    iter::Sum,
    marker::PhantomData,
    ops::{Add, Mul},
};

use ndarray::{Array1, Array2, Ix2, Dimension, Array};

use crate::{
    activations::{Activation, Relu},
    initializers::{Initializer, Zeros},
};

use super::Layer;

/// A densely-connected NN layer.
///
/// # Examples
///
/// Defining the input and output shape by giving the type:
///
/// ```
/// use robit::layers::Dense;
///
/// let layer: Dense<2, 4> = Dense::new();
/// ```
///
/// Use an [Activation] function different from the default [Relu]:
///
/// ```
/// use robit::{layers::Dense, activations::Sigmoid};
///
/// let layer: Dense<2, 4, Sigmoid> = Dense::new();
/// ```
pub struct Dense<A = Relu, I = Zeros, J = Zeros, T = f64>
where
    I: Initializer<T>,
    J: Initializer<T>,
    A: Activation<T>,
{
    weight_initializer: I,
    bias_initializer: J,
    activation: A,
    _phantom: PhantomData<T>,
}

impl<const M: usize, const N: usize, A, I, J, T> Dense<A, I, J, T>
where
    I: Initializer<T> + Default,
    J: Initializer<T> + Default,
    A: Activation<T> + Default,
{
    pub fn new() -> Self {
        Self {
            weight_initializer: I::default(),
            bias_initializer: J::default(),
            activation: A::default(),
            _phantom: PhantomData,
        }
    }
}

impl<const M: usize, const N: usize, A, I, J, T> Dense<A, I, J, T>
where
    I: Initializer<T>,
    J: Initializer<T>,
    A: Activation<T> + Default,
{
    pub fn with_initializers(weight_initializer: I, bias_initializer: J) -> Self {
        Self {
            weight_initializer,
            bias_initializer,
            activation: A::default(),
            _phantom: PhantomData,
        }
    }
}

impl<const M: usize, const N: usize, A, I, J, T> Dense<A, I, J, T>
where
    I: Initializer<T> + Default,
    J: Initializer<T> + Default,
    A: Activation<T>,
{
    /// Returns an instance that uses the given [Activation] function.
    ///
    /// # Example
    ///
    /// ```
    /// use robit::{layers::Dense, activations::Sigmoid};
    ///
    /// let activation = Sigmoid::new(1.05);
    ///
    /// let layer: Dense<2, 4, Sigmoid> = Dense::with_activation(activation);
    /// ```
    pub fn with_activation(activation: A) -> Self {
        Self {
            weight_initializer: I::default(),
            bias_initializer: J::default(),
            activation,
            _phantom: PhantomData,
        }
    }
}

impl<D, A, I, J, T> Layer<D, T> for Dense<A, I, J, T>
where
    D: Dimension,
    I: Initializer<T>,
    J: Initializer<T>,
    T: Add<Output = T> + Mul<Output = T> + Sum + Debug + Copy,
    A: Activation<T>,
{
    fn predict_step(&self, w: &Array<T, D>, b: &Array1<T>, a: Array1<T>) -> Array1<T> {
        self.activation.call(a.dot(w) + b)
    }

    fn gen_weights(&mut self) -> Array<T, D> {
        self.weight_initializer.gen(self.shape())
    }

    fn gen_biases(&mut self) -> Array1<T> {
        self.bias_initializer.gen(self.shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let layer: Dense<2, 3> = Dense::new();

        assert_eq!(2, layer.input_shape());
        assert_eq!(3, layer.output_shape());
    }

    // #[test]
    // fn test_add() {
    //     let weights_a = [0.5, 0.3, 0.4, 0.2, 0.6, 0.7];
    //     let biases_a = [0.5, 0.3, 0.4, 0.2, 0.6, 0.7];
    //     let inputs_a = [2.0, 3.0, 4.0];
    //     let layer_a = Dense::new();
    //     assert_eq!([4.7, 6.5], layer_a.predict(weights_a, biases_a, inputs_a));

    //     let weights_b = [0.5, 0.1, 0.4, 0.2, 0.6, 0.2];
    //     let biases_b = [0.5, 0.4, 0.2, 0.4, 0.4, 0.2];
    //     let inputs_b = [2.0, 3.1];
    //     let layer_b = Dense::from_weights_and_biases(weights_b, biases_b);
    //     assert_eq!([2.21, 2.02, 2.42], layer_b.forward(inputs_b));
    // }
}
