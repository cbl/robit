use std::{cmp::Ordering, ops::Div};

use num_traits::{Float, One, Zero};

pub trait Activation<T> {
    fn call(&self, a: T) -> T;
}

/// Rectified Linear Unit (ReLU).
///
/// [Nair, Vinod and Geoffrey E. Hinton. “Rectified Linear Units Improve
/// Restricted Boltzmann Machines.” International Conference on Machine Learning
/// (2010).](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)
pub struct Relu;

impl<T> Activation<T> for Relu
where
    T: Zero + PartialOrd,
{
    fn call(&self, a: T) -> T {
        match T::zero().partial_cmp(&a) {
            None => T::zero(),
            Some(ordering) => match ordering {
                Ordering::Less | Ordering::Equal => a,
                Ordering::Greater => T::zero(),
            },
        }
    }
}

impl Default for Relu {
    fn default() -> Self {
        Self
    }
}

pub struct Sigmoid<T = f64> {
    β: T,
}

impl<T: Float> Sigmoid<T> {
    pub fn new(β: T) -> Self {
        Self { β }
    }
}

impl<T> Activation<T> for Sigmoid<T>
where
    T: Div + One + Float,
{
    fn call(&self, a: T) -> T {
        T::one().div(T::one() + (-a * self.β).exp())
    }
}

impl Default for Sigmoid<f64> {
    fn default() -> Self {
        Sigmoid { β: 1.0 }
    }
}
