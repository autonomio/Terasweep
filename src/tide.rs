use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::process::{Command, Stdio};

use crate::optimized::{self, CachedFit, CachedTrial, Level};
use crate::results::{
    self, CommonSpace, PriceStats, RunOutput, ScoreConfig, ScoredRow, SplitMix64, SweepSummary,
};
use crate::Matrix;

#[derive(Clone, Copy)]
pub struct TrainConfig {
    pub lookback: usize,
    pub hidden_width: usize,
    pub learning_rate: f64,
    pub epochs: usize,
}

pub struct FitOut {
    pub config: TrainConfig,
    pub preds: Vec<f64>,
    pub mae: f64,
    pub fit_failed: bool,
    fit_predict_seconds: f64,
}

type TrainKey = (usize, usize, u64, usize);

struct ModelSpace {
    lookbacks: Vec<usize>,
    hidden_widths: Vec<usize>,
    learning_rates: Vec<f64>,
    epochs: Vec<usize>,
}

struct Param {
    train: TrainConfig,
    score: ScoreConfig,
    threshold_ordinal: usize,
}

pub fn run(
    config_json: &str,
    train: &Matrix,
    test: &Matrix,
    price_stats: &PriceStats,
    runs: usize,
) -> Result<RunOutput, Box<dyn std::error::Error>> {
    let level = Level::from_env()?;
    if matches!(level, Level::Naive | Level::Stats) {
        return Err("naive and stats execution levels currently require Ridge".into());
    }
    let common = CommonSpace::from_json(config_json)?;
    let model = ModelSpace::from_json(config_json)?;
    let train_configs = model.train_configs();
    let train_cache = precompute_train_cache(common.seed, train, test, &train_configs)?;
    let train_index: HashMap<TrainKey, usize> = train_cache
        .iter()
        .enumerate()
        .map(|(idx, entry)| (cache_key(entry.config), idx))
        .collect();
    let mut summary = SweepSummary::new();
    let fits: Vec<_> = train_cache
        .iter()
        .map(|fit| CachedFit {
            preds: &fit.preds,
            mae: fit.mae,
            fit_failed: fit.fit_failed,
        })
        .collect();
    let score_cache_entries = optimized::sweep_cached(
        level,
        runs,
        |id| {
            let param = model.sample(&common, id);
            CachedTrial {
                train_idx: train_index[&cache_key(param.train)],
                threshold_ordinal: param.threshold_ordinal,
                score: param.score,
            }
        },
        &common,
        &fits,
        price_stats,
        &mut summary,
    )?;

    results::export_predictions(train_cache.iter().map(|fit| {
        let c = fit.config;
        (
            format!(
                r#"{{"lookback":{},"hidden_width":{},"learning_rate":{},"epochs":{}}}"#,
                c.lookback, c.hidden_width, c.learning_rate, c.epochs
            ),
            fit.fit_failed,
            fit.preds.as_slice(),
        )
    }))?;

    let best_return_json = row_json(
        summary.best_return(),
        &model,
        &common,
        &train_index,
        &train_cache,
        price_stats,
    );
    let best_sharpe_json = row_json(
        summary.best_sharpe(),
        &model,
        &common,
        &train_index,
        &train_cache,
        price_stats,
    );
    let best_mae_json = row_json(
        summary.best_mae(),
        &model,
        &common,
        &train_index,
        &train_cache,
        price_stats,
    );
    Ok(RunOutput {
        unique_alphas: model.learning_rates.len(),
        unique_train_configs: train_cache.len(),
        failed_train_configs: train_cache.iter().filter(|entry| entry.fit_failed).count(),
        score_cache_entries,
        best_return_json,
        best_sharpe_json,
        best_mae_json,
        summary,
    })
}

impl ModelSpace {
    fn from_json(src: &str) -> Result<Self, Box<dyn std::error::Error>> {
        if ["l2", "init_scale", "activation"]
            .iter()
            .any(|key| results::has_json_key(src, key))
        {
            return Err("Google Research TiDE does not expose the prototype's l2, init_scale, or activation settings".into());
        }
        Ok(Self {
            lookbacks: results::json_usize_array(src, "lookbacks")?,
            hidden_widths: results::json_usize_array(src, "hidden_width")?,
            learning_rates: results::json_f64_array(src, "learning_rate")?,
            epochs: results::json_usize_array(src, "epochs")?,
        })
    }

    fn train_configs(&self) -> Vec<TrainConfig> {
        let mut configs = Vec::new();
        for &lookback in &self.lookbacks {
            for &hidden_width in &self.hidden_widths {
                for &learning_rate in &self.learning_rates {
                    for &epochs in &self.epochs {
                        configs.push(TrainConfig {
                            lookback,
                            hidden_width,
                            learning_rate,
                            epochs,
                        });
                    }
                }
            }
        }
        configs
    }

    fn sample(&self, common: &CommonSpace, id: usize) -> Param {
        let mut rng = SplitMix64::new(common.seed ^ results::mix_id(id as u64));
        let signal_rule = common.signal_rules[rng.index(common.signal_rules.len())];
        let thresholds = if signal_rule.is_abs() {
            &common.threshold_abs
        } else {
            &common.threshold_signed
        };
        let train = TrainConfig {
            lookback: self.lookbacks[rng.index(self.lookbacks.len())],
            hidden_width: self.hidden_widths[rng.index(self.hidden_widths.len())],
            learning_rate: self.learning_rates[rng.index(self.learning_rates.len())],
            epochs: self.epochs[rng.index(self.epochs.len())],
        };
        let threshold_ordinal = rng.index(thresholds.len());
        Param {
            train,
            threshold_ordinal,
            score: ScoreConfig {
                threshold: thresholds[threshold_ordinal],
                fee_bps: common.fee_bps[rng.index(common.fee_bps.len())],
                slippage_bps: common.slippage_bps[rng.index(common.slippage_bps.len())],
                signal_rule,
            },
        }
    }
}

fn row_json(
    row: Option<ScoredRow>,
    model: &ModelSpace,
    common: &CommonSpace,
    train_index: &HashMap<TrainKey, usize>,
    train_cache: &[FitOut],
    price_stats: &PriceStats,
) -> String {
    match row {
        Some(row) => {
            let param = model.sample(common, row.id);
            let cache_idx = train_index[&cache_key(param.train)];
            let detail = results::compute_backtest_stats(
                &train_cache[cache_idx].preds,
                param.score,
                price_stats,
                common,
            );
            format!(
                "{{\"model\": \"tide\", \"lookback\": {}, \"hidden_width\": {}, \"learning_rate\": {}, \"epochs\": {}, \"id\": {}, \"threshold\": {}, \"fee_bps\": {}, \"slippage_bps\": {}, \"signal_rule\": \"{}\", \"mae\": {}, \"signal_rate_pct\": {}, \"backtest_total_return_net_pct\": {}, \"backtest_sharpe_per_bar\": {}, {}}}",
                param.train.lookback,
                param.train.hidden_width,
                results::json_float(param.train.learning_rate),
                param.train.epochs,
                row.id,
                results::json_float(param.score.threshold),
                results::json_float(param.score.fee_bps),
                results::json_float(param.score.slippage_bps),
                param.score.signal_rule.as_str(),
                results::json_float(row.mae),
                results::json_float(row.signal_rate),
                results::json_float(row.net_return),
                results::json_float(row.sharpe),
                results::backtest_stats_json(detail)
            )
        }
        None => "null".to_string(),
    }
}

fn cache_key(config: TrainConfig) -> TrainKey {
    (
        config.lookback,
        config.hidden_width,
        config.learning_rate.to_bits(),
        config.epochs,
    )
}

fn precompute_train_cache(
    seed: u64,
    train: &Matrix,
    test: &Matrix,
    configs: &[TrainConfig],
) -> Result<Vec<FitOut>, Box<dyn std::error::Error>> {
    if train.cols != 1 || test.cols != 1 || train.rows == 0 || test.rows == 0 {
        return Err("TiDE requires one nonempty observed return column in each split".into());
    }
    let mut seen = HashSet::new();
    let configs: Vec<_> = configs
        .iter()
        .copied()
        .filter(|&c| seen.insert(cache_key(c)))
        .collect();
    if configs.iter().any(|c| {
        c.lookback == 0
            || c.hidden_width == 0
            || c.epochs == 0
            || !c.learning_rate.is_finite()
            || c.learning_rate <= 0.0
    }) {
        return Err(
            "TiDE lookback, hidden width, epochs and learning rate must be positive".into(),
        );
    }
    let python = std::env::var_os("TERASWEEP_TIDE_PYTHON")
        .ok_or("Set TERASWEEP_TIDE_PYTHON to a Python interpreter with TensorFlow 2.16.2")?;
    if std::env::var_os("TERASWEEP_TIDE_SOURCE").is_none() {
        return Err("Set TERASWEEP_TIDE_SOURCE to the pinned Google Research models.py".into());
    }
    let mut child = Command::new(python)
        .args(["-B", "-c", include_str!("../scripts/tide_backend.py")])
        .env("TF_USE_LEGACY_KERAS", "0")
        .env("OMP_NUM_THREADS", "1")
        .env("TF_NUM_INTEROP_THREADS", "1")
        .env("TF_NUM_INTRAOP_THREADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let sent = (|| -> std::io::Result<()> {
        let mut input = std::io::BufWriter::new(child.stdin.take().unwrap());
        write!(
            input,
            "[{},{},{},[",
            seed & i32::MAX as u64,
            train.rows,
            test.rows
        )?;
        for (index, c) in configs.iter().enumerate() {
            if index > 0 {
                write!(input, ",")?;
            }
            write!(
                input,
                "[{},{},{},{}]",
                c.lookback, c.hidden_width, c.learning_rate, c.epochs
            )?;
        }
        writeln!(input, "]]")?;
        for value in train.x.iter().chain(&train.y).chain(&test.x) {
            input.write_all(&value.to_le_bytes())?;
        }
        input.flush()
    })();
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!("TiDE reference backend failed: {}", output.status).into());
    }
    sent?;
    let row_bytes = test
        .rows
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(8))
        .ok_or("TiDE prediction size overflow")?;
    if row_bytes.checked_mul(configs.len()) != Some(output.stdout.len()) {
        return Err("TiDE reference backend returned an unexpected prediction count".into());
    }
    let mut cache = Vec::with_capacity(configs.len());
    for (config, data) in configs
        .into_iter()
        .zip(output.stdout.chunks_exact(row_bytes))
    {
        let fit_predict_seconds = f64::from_le_bytes(data[..8].try_into()?);
        let preds: Vec<_> = data[8..]
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let mae = preds
            .iter()
            .zip(&test.y)
            .map(|(pred, target)| (pred - target).abs())
            .sum::<f64>()
            / test.rows as f64;
        cache.push(FitOut {
            config,
            preds,
            mae,
            fit_failed: false,
            fit_predict_seconds,
        });
    }
    Ok(cache)
}

pub(crate) fn fit_one(
    train: &Matrix,
    test: &Matrix,
    config: TrainConfig,
    seed: u64,
) -> Result<crate::FitPrediction, Box<dyn std::error::Error>> {
    let fit = precompute_train_cache(seed, train, test, &[config])?
        .pop()
        .ok_or("TiDE returned no fit")?;
    Ok(crate::FitPrediction {
        predictions: fit.preds,
        fit_predict_seconds: fit.fit_predict_seconds,
    })
}
