mod mse;

pub use mse::MeanSquaredError;

pub trait Loss<T> {
    fn get<const M: usize>(&self, y_true: [T; M], y_pred: [T; M]) -> T;
}
