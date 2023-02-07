use std::marker::PhantomData;

use ndarray::{Array, Array1, Array2, Dim, Dimension, Ix, Ix1, Ix2};
use ndarray_rand::RandomExt;

use super::Layer;

pub struct Input<D: Dimension, T = f64> {
    dim: D,
    _phantom: PhantomData<T>,
}

impl<D, T> Layer<D, T> for Input<T>
where
    D: Dimension,
    T: Copy,
{
    /// Predicts the [Input] layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use robit::layers::{Input, Layer};
    ///
    /// let l: Input<2> = Input::new();
    ///
    /// let input = [0.1, 0.2];
    ///
    /// assert_eq!(l.input_shape(), l.output_shape());
    /// assert_eq!(input, l.predict(&[], &[], &input));
    /// ```
    fn predict_step(&self, weights: &Array2<T>, biases: &Array1<T>, input: Array1<T>) -> Array1<T> {
        input
    }

    fn gen_weights(&mut self) -> Array<T, D> {
        Array::from_shape_vec(self.shape(), vec![])
    }

    fn gen_biases(&mut self) -> Array1<T> {
        Array1::from_shape_vec(self.dim().raw_slice()[0], vec![])
    }
}

impl<D, T> Input<T>
where
    D: Dimension,
{
    pub fn new(dim: D) -> Self {
        Self {
            dim,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input() {
        // let layer: Input<2> = Input::new();

        // assert_eq!(layer.output_shape(), layer.input_shape());
        // assert_eq!([0.1, 0.2], layer.predict(&[], &[], &[0.1, 0.2]));
    }
}
