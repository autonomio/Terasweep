use std::error::Error;
use std::time::Instant;

use crate::{dlinear, lightgbm, ridge, tide, Matrix};

/// Settings for a supported regression or one-step forecasting model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Model {
    Ridge {
        alpha: f64,
        fit_intercept: bool,
        solver_eps: f64,
    },
    DLinear {
        alpha: f64,
        lookback: usize,
        kernel: usize,
        solver_eps: f64,
    },
    LightGbm {
        learning_rate: f64,
        num_trees: usize,
        num_leaves: usize,
        max_bins: usize,
        lambda_l2: f64,
        min_data_in_leaf: usize,
        seed: u32,
    },
    Tide {
        lookback: usize,
        hidden_width: usize,
        learning_rate: f64,
        epochs: usize,
        seed: u32,
    },
}

pub struct FitPrediction {
    pub predictions: Vec<f64>,
    pub fit_predict_seconds: f64,
}

pub(crate) fn measure(
    fit: impl FnOnce() -> Result<Vec<f64>, Box<dyn Error>>,
) -> Result<FitPrediction, Box<dyn Error>> {
    let start = Instant::now();
    let predictions = fit()?;
    Ok(FitPrediction {
        predictions,
        fit_predict_seconds: start.elapsed().as_secs_f64(),
    })
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn valid_matrix(matrix: &Matrix) -> bool {
    matrix.has_shape() && matrix.x.iter().chain(&matrix.y).all(|x| x.is_finite())
}

/// Fit a model on `train` and predict one value per row of `test`.
pub fn fit_predict(
    model: &Model,
    train: &Matrix,
    test: &Matrix,
) -> Result<FitPrediction, Box<dyn Error>> {
    check_matrices(train, test)?;
    let prediction = match model.training(train.cols)? {
        Training::Ridge(config) => {
            measure(|| ridge::fit_predict(&ridge::precompute_stats(train), test, config))?
        }
        Training::DLinear(config) => measure(|| {
            dlinear::DesignCache::new(train, test, config.lookback, config.kernel)?
                .fit_predict(config)
        })?,
        Training::LightGbm(config, seed) => lightgbm::fit_one(train, test, config, seed)?,
        Training::Tide(config, seed) => tide::fit_batch(train, test, &[config], seed)?
            .pop()
            .ok_or("TiDE returned no fit")?,
    };
    check_predictions(&prediction.predictions, test.rows)?;
    Ok(prediction)
}

#[derive(Clone, Copy)]
pub(crate) enum Training {
    Ridge(ridge::TrainConfig),
    DLinear(dlinear::TrainConfig),
    LightGbm(lightgbm::TrainConfig, u32),
    Tide(tide::TrainConfig, u32),
}

#[derive(Eq, Hash, PartialEq)]
pub(crate) enum TrainingKey {
    Ridge(u64, bool, u64),
    DLinear(u64, usize, usize, u64),
    LightGbm(u64, usize, usize, usize, u64, usize, u32),
    Tide(usize, usize, u64, usize, u32),
}

impl Training {
    pub(crate) fn key(self) -> TrainingKey {
        match self {
            Self::Ridge(c) => {
                TrainingKey::Ridge(c.alpha.to_bits(), c.fit_intercept, c.solver_eps.to_bits())
            }
            Self::DLinear(c) => TrainingKey::DLinear(
                c.alpha.to_bits(),
                c.lookback,
                c.kernel,
                c.solver_eps.to_bits(),
            ),
            Self::LightGbm(c, seed) => TrainingKey::LightGbm(
                c.learning_rate.to_bits(),
                c.num_trees,
                c.num_leaves,
                c.max_bins,
                c.lambda_l2.to_bits(),
                c.min_data_in_leaf,
                seed,
            ),
            Self::Tide(c, seed) => TrainingKey::Tide(
                c.lookback,
                c.hidden_width,
                c.learning_rate.to_bits(),
                c.epochs,
                seed,
            ),
        }
    }
}

impl Model {
    pub(crate) fn training(self, columns: usize) -> Result<Training, Box<dyn Error>> {
        if matches!(self, Self::DLinear { .. } | Self::Tide { .. }) && columns != 1 {
            return Err("DLinear and TiDE require a single observation column".into());
        }
        Ok(match self {
            Self::Ridge {
                alpha,
                fit_intercept,
                solver_eps,
            } => {
                if !nonnegative(alpha) || !nonnegative(solver_eps) {
                    return Err("Ridge alpha and solver_eps must be finite and nonnegative".into());
                }
                Training::Ridge(ridge::TrainConfig {
                    alpha,
                    fit_intercept,
                    solver_eps,
                })
            }
            Self::DLinear {
                alpha,
                lookback,
                kernel,
                solver_eps,
            } => {
                if !positive(alpha) || lookback == 0 || kernel % 2 == 0 || !nonnegative(solver_eps)
                {
                    return Err("DLinear requires positive alpha and lookback, a positive odd kernel, and nonnegative solver_eps".into());
                }
                Training::DLinear(dlinear::TrainConfig {
                    alpha,
                    lookback,
                    kernel,
                    solver_eps,
                })
            }
            Self::LightGbm {
                learning_rate,
                num_trees,
                num_leaves,
                max_bins,
                lambda_l2,
                min_data_in_leaf,
                seed,
            } => {
                if !positive(learning_rate)
                    || num_trees == 0
                    || num_leaves < 2
                    || max_bins < 2
                    || !nonnegative(lambda_l2)
                    || min_data_in_leaf == 0
                {
                    return Err("Invalid LightGBM regression settings".into());
                }
                Training::LightGbm(
                    lightgbm::TrainConfig {
                        learning_rate,
                        num_trees,
                        num_leaves,
                        max_bins,
                        lambda_l2,
                        min_data_in_leaf,
                    },
                    seed & i32::MAX as u32,
                )
            }
            Self::Tide {
                lookback,
                hidden_width,
                learning_rate,
                epochs,
                seed,
            } => {
                if lookback == 0 || hidden_width == 0 || !positive(learning_rate) || epochs == 0 {
                    return Err(
                        "TiDE lookback, hidden width, epochs and learning rate must be positive"
                            .into(),
                    );
                }
                Training::Tide(
                    tide::TrainConfig {
                        lookback,
                        hidden_width,
                        learning_rate,
                        epochs,
                    },
                    seed & i32::MAX as u32,
                )
            }
        })
    }
}

fn nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

pub(crate) fn check_matrices(train: &Matrix, test: &Matrix) -> Result<(), Box<dyn Error>> {
    if !valid_matrix(train) || !valid_matrix(test) || test.cols != train.cols {
        return Err(
            "Train and test need finite, nonempty matrices with the same column count".into(),
        );
    }
    Ok(())
}

pub(crate) fn check_predictions(predictions: &[f64], rows: usize) -> Result<(), Box<dyn Error>> {
    if predictions.len() != rows || predictions.iter().any(|x| !x.is_finite()) {
        return Err("Model produced an invalid prediction vector".into());
    }
    Ok(())
}
