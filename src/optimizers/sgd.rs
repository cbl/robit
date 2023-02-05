use crate::Model;

use super::Optimizer;

pub struct SGD;

impl Optimizer for SGD {
    fn train<const M: usize, const N: usize, T>(&self, training_data: Vec<([T; M], [T; N])>) {
        //
    }
}

impl Default for SGD {
    fn default() -> Self {
        Self
    }
}