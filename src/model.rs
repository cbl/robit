use std::collections::HashMap;

use crate::layers::{Input, Layer, Pipe};

pub struct Model<const M: usize, const N: usize, T = f64> {
    layers: Box<dyn Layer<M, N, T>>,
}

impl<const M: usize, T: 'static> Model<M, M, T> {
    pub fn new() -> Self {
        Self {
            layers: Box::new(Input::new()),
        }
    }
}

impl<const M: usize, const N: usize, T: 'static> Model<M, N, T> {
    /// Appends a layer and returns the new shaped model.
    ///
    /// ```
    /// use robit::{Model, layers::Dense};
    ///
    /// let model: Model<2, 4> = Model::new()
    ///     .add_layer(Dense::<2, 3>::new())
    ///     .add_layer(Dense::<3, 4>::new());
    /// ```
    pub fn add_layer<const O: usize, L: Layer<N, O, T> + 'static>(
        self,
        layer: L,
    ) -> Model<M, O, T> {
        Model {
            layers: Box::new(Pipe::new(self.layers.into(), Box::new(layer))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // todo...
}
