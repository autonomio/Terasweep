use std::collections::{HashMap, HashSet};

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

struct BinnedData {
    rows: usize,
    cols: usize,
    train_bins: Vec<u16>,
    test_bins: Vec<u16>,
}

struct Node {
    feature: usize,
    bin: u16,
    left: usize,
    right: usize,
    value: f64,
    is_leaf: bool,
}

struct Tree {
    nodes: Vec<Node>,
}

struct Split {
    leaf: usize,
    feature: usize,
    bin: u16,
    gain: f64,
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
    let train_cache = precompute_train_cache(train, test, &train_configs);
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

fn precompute_train_cache(train: &Matrix, test: &Matrix, configs: &[TrainConfig]) -> Vec<FitOut> {
    let mut cache = Vec::new();
    let mut seen: HashSet<TrainKey> = HashSet::new();
    let mut bins: HashMap<usize, BinnedData> = HashMap::new();
    for &config in configs {
        let key = cache_key(config);
        if !seen.insert(key) {
            continue;
        }
        if !bins.contains_key(&config.max_bins) {
            bins.insert(config.max_bins, build_bins(train, test, config.max_bins));
        }
        let binned = bins.get(&config.max_bins).unwrap();
        let (preds, mae, fit_failed) = fit_predict(train, test, binned, config);
        cache.push(FitOut {
            config,
            preds,
            mae,
            fit_failed,
        });
    }
    cache
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

fn fit_predict(
    train: &Matrix,
    test: &Matrix,
    bins: &BinnedData,
    config: TrainConfig,
) -> (Vec<f64>, f64, bool) {
    if config.num_leaves < 2
        || config.num_trees == 0
        || config.max_bins < 2
        || config.min_data_in_leaf == 0
    {
        return (Vec::new(), f64::NAN, true);
    }
    let init = train.y.iter().sum::<f64>() / train.rows as f64;
    let mut train_pred = vec![init; train.rows];
    let mut test_pred = vec![init; test.rows];
    let mut residual = vec![0.0; train.rows];

    for _ in 0..config.num_trees {
        for row in 0..train.rows {
            residual[row] = train.y[row] - train_pred[row];
        }
        let tree = build_tree(bins, &residual, config);
        for row in 0..train.rows {
            train_pred[row] += config.learning_rate * tree.predict_train(bins, row);
        }
        for row in 0..test.rows {
            test_pred[row] += config.learning_rate * tree.predict_test(bins, row);
        }
    }

    let mut mae = 0.0;
    for (pred, target) in test_pred.iter().zip(test.y.iter()) {
        mae += (pred - target).abs();
    }
    (test_pred, mae / test.rows as f64, false)
}

fn build_bins(train: &Matrix, test: &Matrix, max_bins: usize) -> BinnedData {
    let rows = train.rows;
    let cols = train.cols;
    let mut train_bins = vec![0u16; train.rows * cols];
    let mut test_bins = vec![0u16; test.rows * cols];
    let mut values = Vec::with_capacity(train.rows);

    for col in 0..cols {
        values.clear();
        for row in 0..train.rows {
            values.push(train.x[row * cols + col]);
        }
        values.sort_by(|a, b| a.total_cmp(b));
        values.dedup();
        let mut cuts = Vec::new();
        if values.len() > 1 {
            for bin in 1..max_bins {
                let idx = bin * (values.len() - 1) / max_bins;
                cuts.push(values[idx]);
            }
            cuts.dedup();
        }
        for row in 0..train.rows {
            train_bins[row * cols + col] = bin_value(train.x[row * cols + col], &cuts);
        }
        for row in 0..test.rows {
            test_bins[row * cols + col] = bin_value(test.x[row * cols + col], &cuts);
        }
    }

    BinnedData {
        rows,
        cols,
        train_bins,
        test_bins,
    }
}

fn bin_value(value: f64, cuts: &[f64]) -> u16 {
    let mut bin = 0usize;
    while bin < cuts.len() && value > cuts[bin] {
        bin += 1;
    }
    bin as u16
}

fn build_tree(bins: &BinnedData, residual: &[f64], config: TrainConfig) -> Tree {
    let root_value = leaf_value(
        (0..bins.rows).map(|row| residual[row]).sum(),
        bins.rows,
        config,
    );
    let mut tree = Tree {
        nodes: vec![Node {
            feature: 0,
            bin: 0,
            left: 0,
            right: 0,
            value: root_value,
            is_leaf: true,
        }],
    };
    let mut leaves = vec![0usize];
    let mut leaf_rows = vec![(0..bins.rows).collect::<Vec<_>>()];

    while leaves.len() < config.num_leaves {
        let split = match best_split(bins, residual, config, &leaves, &leaf_rows) {
            Some(split) => split,
            None => break,
        };
        if split.gain <= 0.0 {
            break;
        }
        let node_idx = leaves[split.leaf];
        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();
        for &row in &leaf_rows[split.leaf] {
            if train_bin(bins, row, split.feature) <= split.bin {
                left_rows.push(row);
            } else {
                right_rows.push(row);
            }
        }
        if left_rows.len() < config.min_data_in_leaf || right_rows.len() < config.min_data_in_leaf {
            break;
        }

        let left_sum = left_rows.iter().map(|&row| residual[row]).sum::<f64>();
        let right_sum = right_rows.iter().map(|&row| residual[row]).sum::<f64>();
        let left_idx = tree.nodes.len();
        let right_idx = left_idx + 1;
        tree.nodes[node_idx] = Node {
            feature: split.feature,
            bin: split.bin,
            left: left_idx,
            right: right_idx,
            value: 0.0,
            is_leaf: false,
        };
        tree.nodes.push(Node {
            feature: 0,
            bin: 0,
            left: 0,
            right: 0,
            value: leaf_value(left_sum, left_rows.len(), config),
            is_leaf: true,
        });
        tree.nodes.push(Node {
            feature: 0,
            bin: 0,
            left: 0,
            right: 0,
            value: leaf_value(right_sum, right_rows.len(), config),
            is_leaf: true,
        });
        leaves[split.leaf] = left_idx;
        leaf_rows[split.leaf] = left_rows;
        leaves.push(right_idx);
        leaf_rows.push(right_rows);
    }

    tree
}

fn best_split(
    bins: &BinnedData,
    residual: &[f64],
    config: TrainConfig,
    leaves: &[usize],
    leaf_rows: &[Vec<usize>],
) -> Option<Split> {
    let mut best: Option<Split> = None;
    let max_bins = config.max_bins;
    let mut bin_sum = vec![0.0; max_bins];
    let mut bin_count = vec![0usize; max_bins];

    for leaf_idx in 0..leaves.len() {
        let rows = &leaf_rows[leaf_idx];
        if rows.len() < config.min_data_in_leaf * 2 {
            continue;
        }
        let total_sum = rows.iter().map(|&row| residual[row]).sum::<f64>();
        let parent_gain = total_sum * total_sum / (rows.len() as f64 + config.lambda_l2);

        for feature in 0..bins.cols {
            bin_sum.fill(0.0);
            bin_count.fill(0);
            for &row in rows {
                let bin = train_bin(bins, row, feature) as usize;
                bin_sum[bin] += residual[row];
                bin_count[bin] += 1;
            }

            let mut left_sum = 0.0;
            let mut left_count = 0usize;
            for bin in 0..max_bins - 1 {
                left_sum += bin_sum[bin];
                left_count += bin_count[bin];
                let right_count = rows.len() - left_count;
                if left_count < config.min_data_in_leaf || right_count < config.min_data_in_leaf {
                    continue;
                }
                let right_sum = total_sum - left_sum;
                let gain = left_sum * left_sum / (left_count as f64 + config.lambda_l2)
                    + right_sum * right_sum / (right_count as f64 + config.lambda_l2)
                    - parent_gain;
                if best.as_ref().map_or(true, |old| gain > old.gain) {
                    best = Some(Split {
                        leaf: leaf_idx,
                        feature,
                        bin: bin as u16,
                        gain,
                    });
                }
            }
        }
    }

    best
}

fn leaf_value(sum: f64, count: usize, config: TrainConfig) -> f64 {
    sum / (count as f64 + config.lambda_l2)
}

fn train_bin(bins: &BinnedData, row: usize, feature: usize) -> u16 {
    bins.train_bins[row * bins.cols + feature]
}

fn test_bin(bins: &BinnedData, row: usize, feature: usize) -> u16 {
    bins.test_bins[row * bins.cols + feature]
}

impl Tree {
    fn predict_train(&self, bins: &BinnedData, row: usize) -> f64 {
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf {
                return node.value;
            }
            idx = if train_bin(bins, row, node.feature) <= node.bin {
                node.left
            } else {
                node.right
            };
        }
    }

    fn predict_test(&self, bins: &BinnedData, row: usize) -> f64 {
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf {
                return node.value;
            }
            idx = if test_bin(bins, row, node.feature) <= node.bin {
                node.left
            } else {
                node.right
            };
        }
    }
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
