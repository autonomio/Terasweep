//! Fit models and run factorized parameter sweeps on prepared matrices.

mod dlinear;
mod input;
mod lightgbm;
mod lightgbm_api;
mod model;
mod optimized;
mod results;
mod ridge;
mod sweep;
mod tide;

pub use input::{read_matrix, read_prices, Matrix};
pub use model::{fit_predict, FitPrediction, Model};
pub use results::{Prices, ScoredRow, SignalRule, SweepSummary};
pub use sweep::{sweep, SweepResult};

#[cfg(test)]
mod correctness_tests;
