use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, Mul},
};

use num_traits::Zero;

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
/// let layer: Dense<2, 4, Sigmoid> = Dense::new().with_activation();
/// ```
pub struct Dense<const M: usize, const N: usize, A = Relu, T = f64>
where
    [(); M * N]:,
    A: Activation<T>,
{
    weights: [T; M * N],
    biases: [T; M * N],
    activation: A,
}

impl<const M: usize, const N: usize, T> Dense<M, N, Relu, T>
where
    [(); M * N]:,
    T: Zero + Copy + PartialOrd,
{
    pub fn new() -> Self {
        Self {
            weights: Zeros::default().gen(),
            biases: Zeros::default().gen(),
            activation: Relu::default(),
        }
    }

    pub fn initialize_with<I: Initializer<{ M * N }, T>, J: Initializer<{ M * N }, T>>(
        mut weight_initializer: I,
        mut bias_initializer: J,
    ) -> Self {
        Self {
            weights: weight_initializer.gen(),
            biases: bias_initializer.gen(),
            activation: Relu::default(),
        }
    }

    pub fn from_weights_and_biases(weights: [T; M * N], biases: [T; M * N]) -> Self {
        Self {
            weights,
            biases,
            activation: Relu::default(),
        }
    }
}

impl<const M: usize, const N: usize, A, T> Dense<M, N, A, T>
where
    [(); M * N]:,
    A: Activation<T>,
{
    /// Returns an instance that uses the given [Activation] function.
    ///
    /// # Example
    ///
    /// ```
    /// use robit::{layers::Dense, activations::Sigmoid};
    ///
    /// let layer: Dense<2, 4, Sigmoid> = Dense::new().with_activation();
    /// ```
    pub fn with_activation<B: Activation<T> + Default>(self) -> Dense<M, N, B, T> {
        Dense {
            weights: self.weights,
            biases: self.biases,
            activation: B::default(),
        }
    }

    /// Returns an instance that uses the given [Activation] function.
    ///
    /// # Example
    ///
    /// ```
    /// use robit::{layers::Dense, activations::Sigmoid};
    ///
    /// let activation = Sigmoid::new(1.05);
    /// let layer: Dense<2, 4, Sigmoid> = Dense::new().set_activation(activation);
    /// ```
    pub fn set_activation<B: Activation<T> + Default>(self, activation: B) -> Dense<M, N, B, T> {
        Dense {
            weights: self.weights,
            biases: self.biases,
            activation,
        }
    }
}

impl<const M: usize, const N: usize, T, A> Layer<M, N, T> for Dense<M, N, A, T>
where
    [(); M * N]:,
    T: Add<Output = T> + Mul<Output = T> + Sum + Debug + Copy,
    A: Activation<T>,
{
    fn forward(&self, input: [T; M]) -> [T; N] {
        self.weights
            .zip(self.biases)
            .chunks(M)
            .map(|weights| {
                self.activation.call(
                    weights
                        .iter()
                        .zip(input)
                        .map(|((weight, bias), a)| *weight * a + *bias)
                        .sum::<T>(),
                )
            })
            .collect::<Vec<T>>()
            .try_into()
            .unwrap()
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

    #[test]
    fn test_add() {
        let weights_a = [0.5, 0.3, 0.4, 0.2, 0.6, 0.7];
        let biases_a = [0.5, 0.3, 0.4, 0.2, 0.6, 0.7];
        let inputs_a = [2.0, 3.0, 4.0];
        let layer_a = Dense::from_weights_and_biases(weights_a, biases_a);
        assert_eq!([4.7, 6.5], layer_a.forward(inputs_a));

        let weights_b = [0.5, 0.1, 0.4, 0.2, 0.6, 0.2];
        let biases_b = [0.5, 0.4, 0.2, 0.4, 0.4, 0.2];
        let inputs_b = [2.0, 3.1];
        let layer_b = Dense::from_weights_and_biases(weights_b, biases_b);
        assert_eq!([2.21, 2.02, 2.42], layer_b.forward(inputs_b));
    }
}
