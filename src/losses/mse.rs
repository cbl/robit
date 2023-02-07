use std::{
    iter::Sum,
    ops::{Div, Mul, Sub}, fmt::Debug,
};

use ndarray::{Array, Array1, Array2, ArrayBase, Ix1, Ix2, ViewRepr, ScalarOperand};
use num_traits::{FromPrimitive, One, Pow, Zero};

use super::Loss;

pub struct MeanSquaredError;

impl<T> Loss<T> for MeanSquaredError
where
     T: Zero + Clone + Copy + Div<Output = T> + Mul<T, Output = T> + Sub<Output = T> + From<u16> + FromPrimitive + ScalarOperand + Debug,
{
    #[inline]
    fn get(&self, y_true: &Array2<T>, y_pred: &Array2<T>) -> Array1<T> {
        let diff = y_pred - y_true;

        // diff.mapv(|x| x * x).mean_axis(ndarray::Axis(0)).unwrap() / (T::from(y_pred.shape()[1].to_owned() as f64))
        diff.sum_axis(ndarray::Axis(0)) / (T::from(y_pred.shape()[1].to_owned() as u16))
    }
}

impl Default for MeanSquaredError {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_f64() {
        let y_true = [0.6, 0.3];
        let y_pred = [0.3, 0.1];

        let mse = MeanSquaredError::default();

        // assert_eq!(0.125, mse.get(&y_true, &y_pred));
    }

    #[test]
    fn test_f64_zero() {
        let y_true = [0.6, 0.3];
        let y_pred = [0.6, 0.3];

        let mse = MeanSquaredError::default();

        // assert_eq!(0.0, mse.get(&y_true, &y_pred));
    }

    #[test]
    fn test_decimal() {
        let y_true = [dec!(0.6), dec!(0.3)];
        let y_pred = [dec!(0.3), dec!(0.1)];

        let mse = MeanSquaredError::default();

        // assert_eq!(dec!(0.125), mse.get(&y_true, &y_pred));
    }
}
