use std::{
    iter::Sum,
    ops::{Div, Mul, Sub},
};

use num_traits::{One, Pow};

use super::Loss;

pub struct MeanSquaredError;

impl<T> Loss<T> for MeanSquaredError
where
    T: One
        + Copy
        + Div<Output = T>
        + Mul<T>
        + Pow<f64, Output = T>
        + From<u32>
        + Sub<Output = T>
        + Sum,
{
    #[inline]
    fn get<const M: usize>(&self, y_true: [T; M], y_pred: [T; M]) -> T {
        let sum: T = (0..M).map(|i| y_pred[i] - y_true[i]).sum();

        (T::one() / T::from(M as u32)) * Pow::pow(sum, 2.0)
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

        assert_eq!(0.125, mse.get(y_true, y_pred));
    }

    #[test]
    fn test_decimal() {
        let y_true = [dec!(0.6), dec!(0.3)];
        let y_pred = [dec!(0.3), dec!(0.1)];

        let mse = MeanSquaredError::default();

        assert_eq!(dec!(0.125), mse.get(y_true, y_pred));
    }
}
