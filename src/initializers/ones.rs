use num_traits::One;

use super::Initializer;

pub struct Ones;

impl<const M: usize, T: One + Copy> Initializer<M, T> for Ones {
    fn gen(&mut self) -> [T; M] {
        [T::one(); M]
    }
}

impl Default for Ones {
    fn default() -> Self {
        Self
    }
}
