use std::collections::{HashMap, HashSet};

use crate::lightgbm_api::LightGbm;
use crate::results::{
    self, BaseScore, CommonSpace, PriceStats, RunOutput, ScoreConfig, ScoreKey, ScoredRow,
    SplitMix64, SweepSummary,
};
use crate::Matrix;

const MAE_DECIMALS: i32 = 3;

#[derive(Clone, Copy)]
pub struct TrainConfig {
    pub learning_rate: f64,
    pub num_trees: usize,
    pub num_leaves: usize,
    pub max_bins: usize,
    pub lambda_l2: f64,
    pub min_data_in_leaf: usize,
}

pub struct FitOut {
    pub config: TrainConfig,
    pub preds: Vec<f64>,
    pub mae: f64,
    pub fit_failed: bool,
}

pub type TrainKey = (u64, usize, usize, usize, u64, usize);

struct ModelSpace {
    learning_rates: Vec<f64>,
    num_trees: Vec<usize>,
    num_leaves: Vec<usize>,
    max_bins: Vec<usize>,
    lambda_l2: Vec<f64>,
    min_data_in_leaf: Vec<usize>,
}

struct Param {
    id: usize,
    train: TrainConfig,
    score: ScoreConfig,
}

pub fn run(
    config_json: &str,
    train: &Matrix,
    test: &Matrix,
    price_stats: &PriceStats,
    runs: usize,
) -> Result<RunOutput, Box<dyn std::error::Error>> {
    let common = CommonSpace::from_json(config_json)?;
    let model = ModelSpace::from_json(config_json)?;
    let train_configs = model.train_configs();
    let train_cache = precompute_train_cache(train, test, &train_configs, common.seed)?;
    let train_index: HashMap<TrainKey, usize> = train_cache
        .iter()
        .enumerate()
        .map(|(idx, entry)| (cache_key(entry.config), idx))
        .collect();
    let mut score_cache: HashMap<ScoreKey, BaseScore> = HashMap::new();
    let mut summary = SweepSummary::new();

    for id in 0..runs {
        let param = model.sample(&common, id);
        let row = evaluate(
            &param,
            &train_index,
            &train_cache,
            &mut score_cache,
            price_stats,
        )?;
        summary.record(row);
    }

    results::export_predictions(train_cache.iter().map(|fit| {
        let config = fit.config;
        (
            format!(
                "{{\"learning_rate\": {}, \"num_trees\": {}, \"num_leaves\": {}, \"max_bins\": {}, \"lambda_l2\": {}, \"min_data_in_leaf\": {}}}",
                results::json_float(config.learning_rate),
                config.num_trees,
                config.num_leaves,
                config.max_bins,
                results::json_float(config.lambda_l2),
                config.min_data_in_leaf,
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
        score_cache_entries: score_cache.len(),
        best_return_json,
        best_sharpe_json,
        best_mae_json,
        summary,
    })
}

impl ModelSpace {
    fn from_json(src: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            learning_rates: results::json_f64_array(src, "learning_rate")?,
            num_trees: results::json_usize_array(src, "num_trees")?,
            num_leaves: results::json_usize_array(src, "num_leaves")?,
            max_bins: results::json_usize_array(src, "max_bins")?,
            lambda_l2: results::json_f64_array(src, "lambda_l2")?,
            min_data_in_leaf: results::json_usize_array(src, "min_data_in_leaf")?,
        })
    }

    fn train_configs(&self) -> Vec<TrainConfig> {
        let mut configs = Vec::with_capacity(
            self.learning_rates.len()
                * self.num_trees.len()
                * self.num_leaves.len()
                * self.max_bins.len()
                * self.lambda_l2.len()
                * self.min_data_in_leaf.len(),
        );
        for &learning_rate in &self.learning_rates {
            for &num_trees in &self.num_trees {
                for &num_leaves in &self.num_leaves {
                    for &max_bins in &self.max_bins {
                        for &lambda_l2 in &self.lambda_l2 {
                            for &min_data_in_leaf in &self.min_data_in_leaf {
                                configs.push(TrainConfig {
                                    learning_rate,
                                    num_trees,
                                    num_leaves,
                                    max_bins,
                                    lambda_l2,
                                    min_data_in_leaf,
                                });
                            }
                        }
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
        let learning_rate = self.learning_rates[rng.index(self.learning_rates.len())];
        let threshold = thresholds[rng.index(thresholds.len())];
        let num_trees = self.num_trees[rng.index(self.num_trees.len())];
        let num_leaves = self.num_leaves[rng.index(self.num_leaves.len())];
        let max_bins = self.max_bins[rng.index(self.max_bins.len())];
        let lambda_l2 = self.lambda_l2[rng.index(self.lambda_l2.len())];
        let min_data_in_leaf = self.min_data_in_leaf[rng.index(self.min_data_in_leaf.len())];
        let fee_bps = common.fee_bps[rng.index(common.fee_bps.len())];
        let slippage_bps = common.slippage_bps[rng.index(common.slippage_bps.len())];
        Param {
            id,
            train: TrainConfig {
                learning_rate,
                num_trees,
                num_leaves,
                max_bins,
                lambda_l2,
                min_data_in_leaf,
            },
            score: ScoreConfig {
                threshold,
                fee_bps,
                slippage_bps,
                signal_rule,
            },
        }
    }
}

fn precompute_train_cache(
    train: &Matrix,
    test: &Matrix,
    configs: &[TrainConfig],
    seed: u64,
) -> Result<Vec<FitOut>, Box<dyn std::error::Error>> {
    let reference = LightGbm::load()?;
    let labels: Vec<f32> = train.y.iter().map(|&value| value as f32).collect();
    let mut cache = Vec::new();
    let mut seen: HashSet<TrainKey> = HashSet::new();
    for &config in configs {
        if !seen.insert(cache_key(config)) {
            continue;
        }
        let fit_failed = config.num_leaves < 2
            || config.num_trees == 0
            || config.max_bins < 2
            || config.min_data_in_leaf == 0;
        let (preds, mae) = if fit_failed {
            (Vec::new(), f64::NAN)
        } else {
            let preds = reference.fit_predict(train, &labels, test, config, seed)?;
            let mae = preds
                .iter()
                .zip(&test.y)
                .map(|(pred, target)| (pred - target).abs())
                .sum::<f64>()
                / test.rows as f64;
            (preds, mae)
        };
        cache.push(FitOut {
            config,
            preds,
            mae,
            fit_failed,
        });
    }
    Ok(cache)
}

fn cache_key(config: TrainConfig) -> TrainKey {
    (
        config.learning_rate.to_bits(),
        config.num_trees,
        config.num_leaves,
        config.max_bins,
        config.lambda_l2.to_bits(),
        config.min_data_in_leaf,
    )
}

fn evaluate(
    param: &Param,
    train_index: &HashMap<TrainKey, usize>,
    train_cache: &[FitOut],
    score_cache: &mut HashMap<ScoreKey, BaseScore>,
    price_stats: &PriceStats,
) -> Result<ScoredRow, Box<dyn std::error::Error>> {
    let cache_idx = *train_index
        .get(&cache_key(param.train))
        .ok_or("missing cache entry")?;
    let entry = &train_cache[cache_idx];
    let score = if entry.fit_failed {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        results::score_cached(
            cache_idx,
            &entry.preds,
            param.score,
            score_cache,
            price_stats,
        )
    };
    Ok(ScoredRow {
        id: param.id,
        signal_rule: param.score.signal_rule,
        fit_failed: entry.fit_failed,
        mae: results::round_to(entry.mae, MAE_DECIMALS),
        signal_rate: score.0,
        net_return: score.1,
        sharpe: score.2,
    })
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
                "{{\"model\": \"lightgbm\", \"learning_rate\": {}, \"num_trees\": {}, \"num_leaves\": {}, \"max_bins\": {}, \"lambda_l2\": {}, \"min_data_in_leaf\": {}, \"id\": {}, \"threshold\": {}, \"fee_bps\": {}, \"slippage_bps\": {}, \"signal_rule\": \"{}\", \"mae\": {}, \"signal_rate_pct\": {}, \"backtest_total_return_net_pct\": {}, \"backtest_sharpe_per_bar\": {}, {}}}",
                results::json_float(param.train.learning_rate),
                param.train.num_trees,
                param.train.num_leaves,
                param.train.max_bins,
                results::json_float(param.train.lambda_l2),
                param.train.min_data_in_leaf,
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
