use ndarray::{Array, Dimension, ShapeBuilder, StrideShape};
use num_traits::Zero;

use super::Initializer;

pub struct Zeros;

impl<T: Zero + Clone> Initializer<T> for Zeros {
    fn gen<S, D>(&self, shape: S) -> Array<T, D>
    where
        D: Dimension,
        S: ShapeBuilder<Dim = D>,
    {
        Array::zeros(shape)
    }
}

impl Default for Zeros {
    fn default() -> Self {
        Self
    }
}
