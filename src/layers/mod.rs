mod dense;
mod input;
mod pipe;

pub use dense::Dense;
pub use input::Input;
pub use pipe::Pipe;

pub trait Layer<const M: usize, const N: usize, T = f64> {
    fn forward(&self, input: [T; M]) -> [T; N];

    #[inline]
    fn input_shape(&self) -> usize {
        M
    }

    #[inline]
    fn output_shape(&self) -> usize {
        N
    }
}
