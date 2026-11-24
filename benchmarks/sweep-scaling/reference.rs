//! Benchmark-only bindings to the production sampler and uncached scalar scorer.
#![allow(dead_code)]

use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;

#[path = "../../src/lightgbm_api.rs"]
mod lightgbm_api;
#[path = "../../src/model.rs"]
mod model;
#[path = "../../src/optimized/mod.rs"]
mod optimized;
#[path = "../../src/results.rs"]
mod results;

use model::FitPrediction;
use results::{CommonSpace, PriceStats, Prices, ScoreConfig, ScoredRow, SignalRule, SweepSummary};

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Trial {
    train: [f64; 6],
    threshold: f64,
    fee: f64,
    slippage: f64,
    rule: usize,
}

impl Trial {
    fn new(train: &[f64], score: ScoreConfig) -> Self {
        let mut key = [0.0; 6];
        key[..train.len()].copy_from_slice(train);
        Self {
            train: key,
            threshold: score.threshold,
            fee: score.fee_bps,
            slippage: score.slippage_bps,
            rule: score.signal_rule.index(),
        }
    }
}

// Each module uses its production ModelSpace and sampler, including RNG order.
macro_rules! sampler {
    ($name:ident, $path:literal, $key:expr) => {
        mod $name {
            include!($path);

            pub(super) fn plan(
                source: &str,
                common: &crate::CommonSpace,
                output: &mut [crate::Trial],
            ) -> Result<(), Box<dyn std::error::Error>> {
                let model = ModelSpace::from_json(source)?;
                for (id, slot) in output.iter_mut().enumerate() {
                    let parameter = model.sample(common, id);
                    *slot = crate::Trial::new(&$key(parameter.train), parameter.score);
                }
                Ok(())
            }
        }
    };
}

sampler!(ridge, "../../src/ridge.rs", |key: TrainConfig| [
    key.alpha,
    f64::from(key.fit_intercept as u8),
    key.solver_eps,
]);
sampler!(dlinear, "../../src/dlinear.rs", |key: TrainConfig| [
    key.alpha,
    f64::from(key.fit_intercept as u8),
    key.solver_eps,
    key.lookback as f64,
    key.kernel as f64,
]);
sampler!(lightgbm, "../../src/lightgbm.rs", |key: TrainConfig| [
    key.learning_rate,
    key.num_trees as f64,
    key.num_leaves as f64,
    key.max_bins as f64,
    key.lambda_l2,
    key.min_data_in_leaf as f64,
]);
sampler!(tide, "../../src/tide.rs", |key: TrainConfig| [
    key.lookback as f64,
    key.hidden_width as f64,
    key.learning_rate,
    key.epochs as f64,
]);

#[no_mangle]
pub unsafe extern "C" fn sweep_plan(
    source: *const c_char,
    count: usize,
    output: *mut Trial,
) -> i32 {
    let run = || -> Result<(), Box<dyn std::error::Error>> {
        let source = CStr::from_ptr(source).to_str()?;
        let common = CommonSpace::from_json(source)?;
        let output = slice::from_raw_parts_mut(output, count);
        match results::model_name(source)?.as_str() {
            "ridge" => ridge::plan(source, &common, output),
            "dlinear" => dlinear::plan(source, &common, output),
            "lightgbm" => lightgbm::plan(source, &common, output),
            "tide" => tide::plan(source, &common, output),
            other => Err(format!("Unknown model {other}").into()),
        }
    };
    match run() {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("Sweep plan: {error}");
            1
        }
    }
}

pub struct Scorer {
    prices: PriceStats,
    target: Vec<f64>,
    summary: SweepSummary,
}

#[no_mangle]
pub unsafe extern "C" fn sweep_scorer(
    open: *const f64,
    close: *const f64,
    target: *const f64,
    rows: usize,
) -> *mut Scorer {
    let prices = Prices {
        open: slice::from_raw_parts(open, rows).to_vec(),
        close: slice::from_raw_parts(close, rows).to_vec(),
    };
    Box::into_raw(Box::new(Scorer {
        prices: results::precompute_price_stats(&prices),
        target: slice::from_raw_parts(target, rows).to_vec(),
        summary: SweepSummary::new(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn sweep_record(
    scorer: *mut Scorer,
    predictions: *const f64,
    trial: *const Trial,
    id: usize,
) {
    let scorer = &mut *scorer;
    let trial = &*trial;
    let predictions = slice::from_raw_parts(predictions, scorer.target.len());
    let signal_rule = [
        SignalRule::Gt,
        SignalRule::Gte,
        SignalRule::Lt,
        SignalRule::Lte,
        SignalRule::AbsGt,
        SignalRule::AbsGte,
    ][trial.rule];
    let base =
        results::compute_base_score(predictions, trial.threshold, signal_rule, &scorer.prices);
    let (signal_rate, net_return, sharpe) = results::apply_costs(
        &base,
        predictions.len(),
        ScoreConfig {
            threshold: trial.threshold,
            fee_bps: trial.fee,
            slippage_bps: trial.slippage,
            signal_rule,
        },
    );
    scorer.summary.record(ScoredRow {
        id,
        signal_rule,
        fit_failed: false,
        mae: results::round_to(results::mae(predictions, &scorer.target), 3),
        signal_rate,
        net_return,
        sharpe,
    });
}

#[no_mangle]
pub unsafe extern "C" fn sweep_finish(scorer: *mut Scorer, output: *mut f64) {
    let scorer = Box::from_raw(scorer);
    let output = slice::from_raw_parts_mut(output, 15);
    for (best, fields) in [
        scorer.summary.best_return(),
        scorer.summary.best_sharpe(),
        scorer.summary.best_mae(),
    ]
    .into_iter()
    .zip(output.chunks_exact_mut(5))
    {
        if let Some(row) = best {
            fields.copy_from_slice(&[
                row.id as f64,
                row.mae,
                row.signal_rate,
                row.net_return,
                row.sharpe,
            ]);
        } else {
            fields.fill(f64::NAN);
        }
    }
}
