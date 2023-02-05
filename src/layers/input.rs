use std::marker::PhantomData;

use super::Layer;

pub struct Input<const M: usize, T = f64>(PhantomData<T>);

impl<const M: usize, T> Layer<M, M, T> for Input<M, T>
where
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
    fn predict(&self, weights: &[T], biases: &[T], input: &[T; M]) -> [T; M] {
        *input
    }

    fn gen_weights(&mut self) -> Vec<T> {
        Vec::new()
    }

    fn gen_biases(&mut self) -> Vec<T> {
        Vec::new()
    }

    #[inline]
    fn shape(&self) -> usize {
        0
    }
}

impl<const M: usize, T> Input<M, T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input() {
        let layer: Input<2> = Input::new();

        assert_eq!(layer.output_shape(), layer.input_shape());
        assert_eq!([0.1, 0.2], layer.predict(&[], &[], &[0.1, 0.2]));
    }
}
