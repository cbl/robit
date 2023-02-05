mod sgd;

pub use sgd::SGD;

use crate::Model;

pub trait Optimizer {
    fn train<const M: usize, const N: usize, T>(&self, training_data: Vec<([T; M], [T; N])>);
}

// pub struct SGD<R, T> {
//     learning_rate: f64,
//     mini_batch_size: usize,
//     rng: R,
//     _phantom: PhantomData<T>,
// }

// impl<R, T> SGD<R, T> {
//     pub fn new(rng: R, learning_rate: f64, mini_batch_size: usize) -> Self {
//         Self {
//             learning_rate,
//             mini_batch_size,
//             rng,
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<R, T, const N: usize> OptIter for SGD<R, T, N> {
//     type Iter = SGDIterator<R, T, N>;

//     fn into_iter(self, training_data: [T; N]) -> SGDIterator<R, T, N> {
//         SGDIterator::new(self.learning_rate, self.rng, training_data)
//     }
// }

// pub struct SGDIterator<R, T, const N: usize> {
//     rng: R,
//     training_data: [T; N],
//     epoch: usize,
//     learning_rate: f64,
//     mini_batch_size: usize,
// }

// impl<R: Rng, T, const N: usize> SGDIterator<R, T, N> {
//     pub fn new(rng: R, training_data: [T; N], learning_rate: f64, mini_batch_size: usize) -> Self {
//         Self {
//             training_data,
//             rng,
//             learning_rate,
//             mini_batch_size,
//             epoch: 0,
//         }
//     }

//     fn update_mini_batch(&mut self, mini_batch: &[T]) {
//         let nabla_b = [0; 10];
//         let nabla_w = [0; 10];

//         for t in mini_batch {
//             let (delta_nabla_b, delta_nabla_w) = self.backprop(t);
//         }
//     }

//     fn backprop(&mut self, training_data: &T) {
//         let biases = [0; 10];
//         let weights = [0; 10];

//         let nabla_b = [0; 10];
//         let nabla_w = [0; 10];

//         let activation = training_data;
//         let activations = [training_data];
//         let mut zs = [0; N];

//         let activations = biases
//             .zip(weights)
//             .map(|(bias, weights)| {
//                 let z = weights * activation + bias;
//                 zs.push(z);
//                 z
//             })
//             .map(|z| z) // activation e.g.: sigmoid(z)
//             .collect();

//         let delta = self.cost_deriviative(activations, y);
//     }
// }

// impl<R: Rng, T, const N: usize> Iterator for SGDIterator<R, T, N> {
//     type Item = ();

//     fn next(&mut self) -> Option<()> {
//         self.epoch += 1;

//         self.training_data.shuffle(&mut self.rng);

//         for mini_batch in self.training_data.chunks(self.mini_batch_size) {
//             self.update_mini_batch(mini_batch);
//         }

//         None
//     }
// }

// impl<R: Rng, U> Optimizer for SGD<R, U> {
//     fn run<T, const N: usize>(&mut self, mut training_data: [T; N], test_data: Option<DVector<T>>) {
//         let epochs = 5;
//         let mini_batch_size = 10;
//         let n_train = training_data.len();
//         // let n =

//         for j in 0..epochs {
//             println!("Epoch {}: Done", j);
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use rand::thread_rng;
//     use rust_decimal_macros::dec;

//     #[test]
//     fn test_add() {
//         let y_true = DVector::from_vec(vec![dec!(0.5), dec!(0.4)]);
//         let y_pred = DVector::from_vec(vec![dec!(0.6), dec!(0.4)]);

//         let training_data = [0.5, 0.2];

//         let mut optimizer = SGD::new(0.001, thread_rng(), 3);

//         let foo = optimizer.run(training_data, None);

//         dbg!(foo);

//         assert_eq!(true, false);
//     }
// }
