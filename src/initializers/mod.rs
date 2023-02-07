mod ones;
mod random_distr;
mod zeros;

use ndarray::{Array, Dimension, ShapeBuilder};
pub use ones::Ones;
pub use random_distr::RandomDistr;
pub use zeros::Zeros;

pub trait Initializer<T = f64> {
    fn gen<S, D>(&self, shape: S) -> Array<T, D>
    where
        D: Dimension,
        S: ShapeBuilder<Dim = D>;
}
