use std::fmt::Debug;

use rand_distr::WeightedError;

use super::Layer;

pub struct Pipe<const M: usize, const N: usize, const O: usize, T = f64> {
    parent: Box<dyn Layer<M, N, T>>,
    child: Box<dyn Layer<N, O, T>>,
}

impl<const M: usize, const N: usize, const O: usize, T> Pipe<M, N, O, T> {
    /// Constructs a new Pipe.
    ///
    /// # Examples
    ///
    /// ```
    /// use robit::layers::{Pipe, Dense, Layer};
    ///
    /// let layer_a: Dense<2, 3> = Dense::new();
    /// let layer_a_shape = layer_a.shape();
    /// let layer_b: Dense<3, 4> = Dense::new();
    /// let layer_b_shape = layer_b.shape();
    /// let mut connected = Pipe::new(Box::new(layer_a), Box::new(layer_b));
    ///
    /// assert_eq!(
    ///     layer_a_shape + layer_b_shape,
    ///     connected.shape()
    /// );
    /// ```
    pub fn new(parent: Box<dyn Layer<M, N, T>>, child: Box<dyn Layer<N, O, T>>) -> Self {
        Self { parent, child }
    }
}

impl<const M: usize, const N: usize, const O: usize, T: Debug> Layer<M, O, T> for Pipe<M, N, O, T> {
    fn predict(&self, weights: &[T], biases: &[T], input: &[T; M]) -> [T; O] {
        let n_parent = self.parent.shape();

        self.child.predict(
            &weights[n_parent..],
            &biases[n_parent..],
            &self
                .parent
                .predict(&weights[..n_parent], &biases[..n_parent], input),
        )
    }

    fn gen_weights(&mut self) -> Vec<T> {
        self.parent
            .gen_weights()
            .into_iter()
            .chain(self.child.gen_weights().into_iter())
            .collect()
    }

    fn gen_biases(&mut self) -> Vec<T> {
        Vec::new()
    }

    #[inline]
    fn shape(&self) -> usize {
        self.child.shape() + self.parent.shape()
    }
}
