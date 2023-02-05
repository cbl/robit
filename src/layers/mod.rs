mod dense;
mod input;
mod pipe;

pub use dense::Dense;
pub use input::Input;
pub use pipe::Pipe;

pub trait Layer<const M: usize, const N: usize, T = f64> {
    fn predict(&self, weights: &[T], biases: &[T], input: &[T; M]) -> [T; N];

    fn gen_weights(&mut self) -> Vec<T>;

    fn gen_biases(&mut self) -> Vec<T>;

    #[inline]
    fn shape(&self) -> usize {
        M * N
    }

    #[inline]
    fn input_shape(&self) -> usize {
        M
    }

    #[inline]
    fn output_shape(&self) -> usize {
        N
    }
}
