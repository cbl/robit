use std::{
    fmt::Debug,
    iter::Sum,
    marker::PhantomData,
    ops::{Add, Mul},
};

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
pub struct Dense<const M: usize, const N: usize, A = Relu, I = Zeros, J = Zeros, T = f64>
where
    I: Initializer<{ M * N }, T>,
    J: Initializer<{ M * N }, T>,
    A: Activation<T>,
{
    weight_initializer: I,
    bias_initializer: J,
    activation: A,
    _phantom: PhantomData<T>,
}

impl<const M: usize, const N: usize, A, I, J, T> Dense<M, N, A, I, J, T>
where
    I: Initializer<{ M * N }, T> + Default,
    J: Initializer<{ M * N }, T> + Default,
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

impl<const M: usize, const N: usize, A, I, J, T> Dense<M, N, A, I, J, T>
where
    I: Initializer<{ M * N }, T>,
    J: Initializer<{ M * N }, T>,
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

impl<const M: usize, const N: usize, A, I, J, T> Dense<M, N, A, I, J, T>
where
    I: Initializer<{ M * N }, T> + Default,
    J: Initializer<{ M * N }, T> + Default,
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

impl<const M: usize, const N: usize, A, I, J, T> Dense<M, N, A, I, J, T>
where
    [(); M * N]:,
    I: Initializer<{ M * N }, T>,
    J: Initializer<{ M * N }, T>,
    A: Activation<T>,
{
}

impl<const M: usize, const N: usize, A, I, J, T> Layer<M, N, T> for Dense<M, N, A, I, J, T>
where
    [(); M * N]:,
    I: Initializer<{ M * N }, T>,
    J: Initializer<{ M * N }, T>,
    T: Add<Output = T> + Mul<Output = T> + Sum + Debug + Copy,
    A: Activation<T>,
{
    fn predict(&self, weights: &[T], biases: &[T], input: &[T; M]) -> [T; N] {
        weights
            .into_iter()
            .copied()
            .zip(biases.into_iter().copied())
            .collect::<Vec<(T, T)>>()
            .chunks(M)
            .map(|weights| {
                self.activation.call(
                    weights
                        .iter()
                        .zip(input)
                        .map(|((weight, bias), a)| *weight * *a + *bias)
                        .sum::<T>(),
                )
            })
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
    }

    fn gen_weights(&mut self) -> Vec<T> {
        let weights: [T; M * N] = self.weight_initializer.gen();

        weights.into()
    }

    fn gen_biases(&mut self) -> Vec<T> {
        let biases: [T; M * N] = self.bias_initializer.gen();

        biases.into()
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
