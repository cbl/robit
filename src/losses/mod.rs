mod mse;

pub use mse::MeanSquaredError;
use ndarray::{Array1, Array2, ArrayBase, Ix1, Ix2, ViewRepr};

pub trait Loss<T> {
    fn get(&self, y_true: &Array2<T>, y_pred: &Array2<T>) -> Array1<T>;
}
