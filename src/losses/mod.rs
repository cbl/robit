use std::{
    iter::Sum,
    ops::{Div, Mul, Sub},
};

use nalgebra::DVector;
use num_traits::{One, Pow};

pub trait Loss<T, U> {
    fn get(&self, y_true: DVector<T>, y_pred: DVector<T>) -> U;
}

pub struct MeanSquaredError();

impl<T, U> Loss<T, U> for MeanSquaredError
where
    T: One + From<usize> + Div<Output = U> + Sum + Sub<Output = T> + Pow<u64, Output = U> + Copy,
    U: Mul<Output = U>,
{
    #[inline]
    fn get(&self, y_true: DVector<T>, y_pred: DVector<T>) -> U {
        let sum: T = (0..y_true.len()).map(|i| y_pred[i] - y_true[i]).sum();

        (T::one() / T::from(y_true.len())) * Pow::pow(sum, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_add() {
        let y_true = DVector::from_vec(vec![dec!(0.5), dec!(0.4)]);
        let y_pred = DVector::from_vec(vec![dec!(0.6), dec!(0.4)]);

        let mse = MeanSquaredError();

        assert_eq!(dec!(0.005), mse.get(y_true, y_pred));
    }
}
