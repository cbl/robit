use super::Layer;

pub struct Pipe<const M: usize, const N: usize, const O: usize, T = f64> {
    parent: Box<dyn Layer<M, N, T>>,
    child: Box<dyn Layer<N, O, T>>,
}

impl<const M: usize, const N: usize, const O: usize, T> Pipe<M, N, O, T> {
    /// Constructs a new Pipe.
    ///
    /// # Examples
    ///
    /// ```
    /// use robit::layers::{Pipe, Dense, Layer};
    ///
    /// let layer_a: Dense<2, 3> = Dense::new();
    /// let layer_b: Dense<3, 4> = Dense::new();
    /// let connected = Pipe::new(Box::new(layer_a), Box::new(layer_b));
    ///
    /// let result: [_; 4] = connected.forward([0.0; 2]);
    /// ```
    pub fn new(parent: Box<dyn Layer<M, N, T>>, child: Box<dyn Layer<N, O, T>>) -> Self {
        Self { parent, child }
    }
}

impl<const M: usize, const N: usize, const O: usize, T> Layer<M, O, T> for Pipe<M, N, O, T> {
    fn forward(&self, input: [T; M]) -> [T; O] {
        self.child.forward(self.parent.forward(input))
    }
}
