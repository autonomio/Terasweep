use std::collections::{hash_map::Entry, BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::time::Instant;

use crate::model::{check_matrices, check_predictions, Training};
use crate::{dlinear, lightgbm, ridge, tide, Matrix, Model};

/// The lowest validation MAE from a completed sweep.
#[derive(Debug)]
pub struct SweepWinner {
    /// Zero-based position in the supplied plan.
    pub trial: usize,
    pub model: Model,
    pub mae: f64,
    /// Predictions in validation row order.
    pub predictions: Vec<f64>,
}

#[derive(Debug)]
pub struct SweepResult {
    pub trials: usize,
    pub unique_fits: usize,
    pub elapsed_seconds: f64,
    pub best: SweepWinner,
}

/// Evaluate an ordered model plan by mean absolute error on labeled validation data.
///
/// Identical training configurations share one fit and score. Ridge shares
/// training statistics across settings, DLinear shares designs per window
/// shape, and LightGBM and TiDE reuse backend setup. Ties choose the first
/// supplied trial. Invalid models, fit errors and nonfinite scores return errors.
pub fn sweep(
    models: &[Model],
    train: &Matrix,
    validation: &Matrix,
) -> Result<SweepResult, Box<dyn Error>> {
    if models.is_empty() {
        return Err("Sweep requires at least one model".into());
    }
    check_matrices(train, validation)?;
    let start = Instant::now();
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for (trial, model) in models.iter().enumerate() {
        let training = model
            .training(train.cols)
            .map_err(|error| format!("Trial {trial}: {error}"))?;
        if seen.insert(training.key()) {
            unique.push((trial, training));
        }
    }
    let unique_fits = unique.len();

    let mut ridge_stats = None;
    let mut designs = HashMap::new();
    let mut lightgbm_backend = None;
    let mut tide_groups = BTreeMap::new();
    let mut best = None;

    for (trial, training) in unique {
        let predictions = match training {
            Training::Ridge(config) => {
                let stats = ridge_stats.get_or_insert_with(|| ridge::precompute_stats(train));
                ridge::fit_predict(stats, validation, config)?
            }
            Training::DLinear(config) => {
                let design = match designs.entry((config.lookback, config.kernel)) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(dlinear::DesignCache::new(
                        train,
                        validation,
                        config.lookback,
                        config.kernel,
                    )?),
                };
                design.fit_predict(config)?
            }
            Training::LightGbm(config, seed) => {
                let backend = match &lightgbm_backend {
                    Some(backend) => backend,
                    None => lightgbm_backend.insert(lightgbm::Backend::new(train)?),
                };
                backend.fit_predict(train, validation, config, seed)?
            }
            Training::Tide(config, seed) => {
                let (trials, configs): &mut (Vec<_>, Vec<_>) = tide_groups.entry(seed).or_default();
                trials.push(trial);
                configs.push(config);
                continue;
            }
        };
        consider(&mut best, trial, models[trial], predictions, validation)?;
    }
    for (seed, (trials, configs)) in tide_groups {
        let fits = tide::fit_batch(train, validation, &configs, seed)?;
        for (trial, fit) in trials.into_iter().zip(fits) {
            consider(&mut best, trial, models[trial], fit.predictions, validation)?;
        }
    }

    Ok(SweepResult {
        trials: models.len(),
        unique_fits,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        best: best.ok_or("Sweep returned no predictions")?,
    })
}

fn consider(
    best: &mut Option<SweepWinner>,
    trial: usize,
    model: Model,
    predictions: Vec<f64>,
    validation: &Matrix,
) -> Result<(), Box<dyn Error>> {
    check_predictions(&predictions, validation.rows)?;
    let mae = predictions
        .iter()
        .zip(&validation.y)
        .map(|(prediction, target)| (prediction - target).abs())
        .sum::<f64>()
        / validation.rows as f64;
    if !mae.is_finite() {
        return Err(format!("Trial {trial} produced nonfinite MAE").into());
    }
    if best
        .as_ref()
        .map_or(true, |prior| (mae, trial) < (prior.mae, prior.trial))
    {
        *best = Some(SweepWinner {
            trial,
            model,
            mae,
            predictions,
        });
    }
    Ok(())
}
