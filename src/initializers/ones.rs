use ndarray::{Array, Dimension, ShapeBuilder, StrideShape};
use num_traits::One;

use super::Initializer;

pub struct Ones;

impl<T: One + Clone> Initializer<T> for Ones {
    fn gen<S, D>(&self, shape: S) -> Array<T, D>
    where
        D: Dimension,
        S: ShapeBuilder<Dim = D>,
    {
        Array::ones(shape)
    }
}

impl Default for Ones {
    fn default() -> Self {
        Self
    }
}
