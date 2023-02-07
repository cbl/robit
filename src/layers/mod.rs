mod dense;
mod input;
// mod pipe;

pub use dense::Dense;
pub use input::Input;
use ndarray::{Array, Array1, Array2, Dimension, Shape};
// pub use pipe::Pipe;

pub trait Layer<T = f64> {
    fn predict_step(&self, w: &Array2<T>, b: &Array1<T>, a: Array1<T>) -> Array1<T>;

    fn gen_weights(&mut self) -> Array2<T>;

    fn gen_biases(&mut self) -> Array1<T>;
}
