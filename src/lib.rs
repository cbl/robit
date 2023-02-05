#![feature(generic_const_exprs)]
#![feature(array_zip)]

pub mod activations;
pub mod initializers;
pub mod layers;
pub mod losses;
pub mod optimizers;

mod model;

pub use model::Model;
