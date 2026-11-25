//! Fit and sweep regression and one-step forecasting models on prepared matrices.

mod dlinear;
mod input;
mod lightgbm;
mod lightgbm_api;
mod model;
mod ridge;
mod sweep;
mod tide;

pub use input::{read_matrix, Matrix};
pub use model::{fit_predict, FitPrediction, Model};
pub use sweep::{sweep, SweepResult, SweepWinner};

#[cfg(test)]
mod correctness_tests;
