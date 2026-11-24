use std::error::Error;
use std::time::Instant;

use crate::{dlinear, lightgbm, ridge, tide, Matrix};

#[derive(Clone, Copy, Debug)]
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

pub fn fit_predict(
    model: &Model,
    train: &Matrix,
    test: &Matrix,
) -> Result<FitPrediction, Box<dyn Error>> {
    if !valid_matrix(train) || !valid_matrix(test) || test.cols != train.cols {
        return Err(
            "Train and test need finite, nonempty matrices with the same column count".into(),
        );
    }
    if matches!(model, Model::DLinear { .. } | Model::Tide { .. }) && train.cols != 1 {
        return Err("DLinear and TiDE require a single observed return column".into());
    }
    let prediction = match *model {
        Model::Ridge {
            alpha,
            fit_intercept,
            solver_eps,
        } => {
            if !alpha.is_finite() || alpha < 0.0 || !solver_eps.is_finite() || solver_eps < 0.0 {
                return Err("Ridge alpha and solver_eps must be finite and nonnegative".into());
            }
            measure(|| {
                ridge::fit_one(
                    train,
                    test,
                    ridge::TrainConfig {
                        alpha,
                        fit_intercept,
                        solver_eps,
                    },
                )
            })?
        }
        Model::DLinear {
            alpha,
            lookback,
            kernel,
            solver_eps,
        } => {
            if !positive(alpha)
                || lookback == 0
                || kernel == 0
                || kernel % 2 == 0
                || !solver_eps.is_finite()
                || solver_eps < 0.0
            {
                return Err("DLinear requires positive alpha and lookback, a positive odd kernel, and nonnegative solver_eps".into());
            }
            measure(|| {
                dlinear::fit_one(
                    train,
                    test,
                    dlinear::TrainConfig {
                        alpha,
                        fit_intercept: true,
                        solver_eps,
                        lookback,
                        kernel,
                    },
                )
            })?
        }
        Model::LightGbm {
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
                || !lambda_l2.is_finite()
                || lambda_l2 < 0.0
                || min_data_in_leaf == 0
            {
                return Err("Invalid LightGBM regression settings".into());
            }
            lightgbm::fit_one(
                train,
                test,
                lightgbm::TrainConfig {
                    learning_rate,
                    num_trees,
                    num_leaves,
                    max_bins,
                    lambda_l2,
                    min_data_in_leaf,
                },
                u64::from(seed),
            )?
        }
        Model::Tide {
            lookback,
            hidden_width,
            learning_rate,
            epochs,
            seed,
        } => tide::fit_one(
            train,
            test,
            tide::TrainConfig {
                lookback,
                hidden_width,
                learning_rate,
                epochs,
            },
            u64::from(seed),
        )?,
    };
    if prediction.predictions.len() != test.rows
        || prediction.predictions.iter().any(|x| !x.is_finite())
    {
        return Err("Model produced an invalid prediction vector".into());
    }
    Ok(prediction)
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
    matrix.rows > 0
        && matrix.cols > 0
        && matrix.rows.checked_mul(matrix.cols) == Some(matrix.x.len())
        && matrix.y.len() == matrix.rows
        && matrix.x.iter().chain(&matrix.y).all(|x| x.is_finite())
}
