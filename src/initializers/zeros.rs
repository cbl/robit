use num_traits::Zero;

use super::Initializer;

pub struct Zeros;

impl<const M: usize, T: Zero + Copy> Initializer<M, T> for Zeros {
    fn gen(&mut self) -> [T; M] {
        [T::zero(); M]
    }
}

impl Default for Zeros {
    fn default() -> Self {
        Self
    }
}
