use std::{marker::PhantomData, ops::SubAssign, time::Instant};

use itertools::Itertools;
use mnist::*;
use ndarray::{
    arr1, arr2, linalg::Dot, s, Array, Array1, Array2, Array3, ArrayBase, ArrayD, Axis, Data, Ix1,
    Ix2, LinalgScalar,
};
use num_traits::One;
use rand_distr::{Normal, StandardNormal};
use robit::{
    activations::Relu,
    initializers::RandomDistr,
    losses::{Loss, MeanSquaredError},
    Model,
};

fn main() {
    type T = f32;

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_labels = Array2::from_shape_vec(
        (50_000, 10),
        trn_lbl
            .iter()
            .map(|x| {
                let mut r = [0.0; 10];
                r[*x as usize] = 1.0;
                r
            })
            .flatten()
            .collect::<Vec<T>>(),
    )
    .expect("Error converting labels to Array2 struct");

    let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as T / 256.0);

    let test_labels = Array2::from_shape_vec(
        (10_000, 10),
        tst_lbl
            .iter()
            .map(|x| {
                let mut r = [0.0; 10];
                r[*x as usize] = 1.0;
                r
            })
            .flatten()
            .collect::<Vec<T>>(),
    )
    .expect("Error converting labels to Array2 struct");
    
    let test_data = Array2::from_shape_vec((10_000, 784), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as T / 256.0);

    // println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));

    // println!("train_data.len() = {:?}", train_data.len());

    type Init = RandomDistr<Normal<T>, T>;

    let mut model: Model<Relu, MeanSquaredError, T> = Model::new(0.00015);

    // model.add_layer((784, 28), Init::default());
    model.add_layer((784, 10), Init::normal_with(0.0, 0.05).unwrap());
    // model.add_layer((28, 10), Init::default());

    // let image_num = 0;
    // let x_example: Array2<f32> = train_data
    //     .slice(s![image_num..(image_num + 1), ..])
    //     .to_owned();
    // let y_example: Array2<f32> = train_labels
    //     .slice(s![image_num..(image_num + 1), ..])
    //     .to_owned();

    let now = Instant::now();

    let mse = MeanSquaredError::default();

    for epoch in 0..200 {
        model.fit(&train_data, &train_labels);

        let test_predict = model.predict(&test_data);
        let loss = mse.get(&test_labels, &test_predict);

        println!("[Epoch:{}] loss = {:?}", epoch, loss.mean().unwrap());
    }

    println!("duration = {:?}", now.elapsed());

    
}
