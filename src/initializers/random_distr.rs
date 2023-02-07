use std::{marker::PhantomData, ops::Range};

use ndarray::{Array, Dimension, ShapeBuilder, StrideShape};
use ndarray_rand::RandomExt;
use num_traits::Float;
use rand::{rngs::ThreadRng, thread_rng, Rng};
use rand_distr::{
    uniform::SampleUniform, Distribution, Normal, NormalError, StandardNormal, Uniform,
};

use super::Initializer;

const DEFAULT_NORMAL_MEAN: f64 = 0.0;
const DEFAULT_NORMAL_STD_DEV: f64 = 0.05;

const DEFAULT_UNIFORM_LOW: f64 = -0.05;
const DEFAULT_UNIFORM_HIGH: f64 = 0.05;

/// Initializer that generates values with distribution.
///
/// # Examples
///
/// The default is a [Normal] distribution with the values:
///
/// - mean = `0.0`
/// - std_dev = `0.5`
///
/// ```
/// use robit::initializers::{RandomDistr, Initializer};
///
/// let mut initializer = RandomDistr::default();
///
/// assert_eq!(0.0, initializer.dist().mean());
/// assert_eq!(0.05, initializer.dist().std_dev());
///
/// let values: [_; 2] = initializer.gen();
/// ```
pub struct RandomDistr<D, T = f64>
where
    D: Distribution<T>,
{
    /// Distribution creates random instances of `T`.
    dist: D,

    _phantom: PhantomData<T>,
}

impl<Distr, T> Initializer<T> for RandomDistr<Distr, T>
where
    Distr: Distribution<T> + Copy,
{
    fn gen<S, Dim>(&self, shape: S) -> Array<T, Dim>
    where
        Dim: Dimension,
        S: ShapeBuilder<Dim = Dim>,
    {
        Array::random::<S, Distr>(shape, self.dist)
    }
}

impl<T> RandomDistr<Normal<T>, T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    /// Returns the instance of the [Normal] distribution.
    pub fn dist(&self) -> Normal<T> {
        self.dist
    }
}

impl Default for RandomDistr<Normal<f64>, f64> {
    fn default() -> Self {
        Self::normal()
    }
}

impl<D, T> RandomDistr<D, T>
where
    D: Distribution<T>,
{
    /// Initializer that generates values with the given distribution and random
    /// generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use robit::initializers::{RandomDistr, Initializer};
    /// use rand_distr::{Uniform, Distribution};
    ///
    /// let dist = Uniform::new(-0.05, 0.05);
    /// let mut initializer = RandomDistr::new(dist);
    ///
    /// let values: [_; 2] = initializer.gen();
    /// ```
    pub fn new(dist: D) -> Self {
        Self {
            dist,
            _phantom: PhantomData,
        }
    }
}

impl RandomDistr<Normal<f64>, f64> {
    /// Initializer that generates values with a normal distribution the and
    /// default values:
    ///
    /// - mean = `0.0`
    /// - std_dev = `0.05`
    ///
    /// # Examples
    ///
    /// ```
    /// use robit::initializers::{RandomDistr, Initializer};
    ///
    /// let mut initializer = RandomDistr::normal();
    ///
    /// assert_eq!(0.0, initializer.dist().mean());
    /// assert_eq!(0.05, initializer.dist().std_dev());
    ///
    /// let values: [_; 2] = initializer.gen();
    /// ```
    pub fn normal() -> Self {
        Self {
            dist: Normal::new(DEFAULT_NORMAL_MEAN, DEFAULT_NORMAL_STD_DEV).unwrap(),
            _phantom: PhantomData,
        }
    }
}

impl<T> RandomDistr<Normal<T>, T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    /// Initializer that generates values with a normal distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use robit::initializers::{RandomDistr, Initializer};
    ///
    /// let mut initializer = RandomDistr::normal_with(0.01, 1.0).unwrap();
    ///
    /// assert_eq!(0.01, initializer.dist().mean());
    /// assert_eq!(1.0, initializer.dist().std_dev());
    ///
    /// let values: [_; 2] = initializer.gen();
    /// ```
    pub fn normal_with(mean: T, std_dev: T) -> Result<Self, NormalError> {
        Ok(Self {
            dist: Normal::new(mean, std_dev)?,
            _phantom: PhantomData,
        })
    }
}

impl RandomDistr<Uniform<f64>, f64> {
    /// Initializer that generates values with a uniform distribution the and
    /// default values:
    ///
    /// - low = `-0.05`
    /// - high = `0.05`
    ///
    /// # Examples
    ///
    /// ```
    /// use robit::initializers::{RandomDistr, Initializer};
    ///
    /// let mut initializer = RandomDistr::uniform();
    ///
    /// let values: [_; 2] = initializer.gen();
    /// ```
    pub fn uniform() -> Self {
        Self {
            dist: Uniform::new(DEFAULT_UNIFORM_LOW, DEFAULT_UNIFORM_HIGH),
            _phantom: PhantomData,
        }
    }
}

impl<T> RandomDistr<Uniform<T>, T>
where
    T: SampleUniform,
{
    /// Initializer that generates values with a uniform distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use robit::initializers::{RandomDistr, Initializer};
    ///
    /// let mut initializer = RandomDistr::uniform_with(-0.05, 0.05);
    ///
    /// let values: [_; 2] = initializer.gen();
    /// ```
    pub fn uniform_with(low: T, high: T) -> Self {
        Self {
            dist: Uniform::new(low, high),
            _phantom: PhantomData,
        }
    }
}
