#![feature(generic_const_exprs)]
#![feature(array_zip)]

pub mod activations;
pub mod initializers;
pub mod layers;
pub mod losses;

mod model;
// pub mod optimizers;

pub use model::Model;
