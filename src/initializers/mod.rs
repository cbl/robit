mod ones;
mod random_distr;
mod zeros;

pub use ones::Ones;
pub use random_distr::RandomDistr;
pub use zeros::Zeros;

pub trait Initializer<const M: usize, T = f64> {
    fn gen(&mut self) -> [T; M];
}
