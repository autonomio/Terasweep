use std::collections::{HashMap, HashSet};

use crate::results::{
    self, BaseScore, CommonSpace, PriceStats, RunOutput, ScoreConfig, ScoreKey, ScoredRow,
    SplitMix64, SweepSummary,
};
use crate::Matrix;

const MAE_DECIMALS: i32 = 3;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub enum Activation {
    Tanh,
    Relu,
    Linear,
}

#[derive(Clone, Copy)]
pub struct TrainConfig {
    pub lookback: usize,
    pub hidden_width: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub l2: f64,
    pub init_scale: f64,
    pub activation: Activation,
}

struct WindowData {
    train_rows: usize,
    test_rows: usize,
    dims: usize,
    train_x: Vec<f64>,
    test_x: Vec<f64>,
    train_y: Vec<f64>,
    y_mean: f64,
    y_scale: f64,
}

struct Network {
    w1: Vec<f64>,
    b1: Vec<f64>,
    v: Vec<f64>,
    skip: Vec<f64>,
    out_bias: f64,
}

struct Adam {
    mw1: Vec<f64>,
    vw1: Vec<f64>,
    mb1: Vec<f64>,
    vb1: Vec<f64>,
    mv: Vec<f64>,
    vv: Vec<f64>,
    mskip: Vec<f64>,
    vskip: Vec<f64>,
    mout: f64,
    vout: f64,
}

pub struct FitOut {
    pub config: TrainConfig,
    pub preds: Vec<f64>,
    pub mae: f64,
    pub fit_failed: bool,
}

pub type TrainKey = (usize, usize, u64, usize, u64, u64, usize);

struct ModelSpace {
    lookbacks: Vec<usize>,
    hidden_widths: Vec<usize>,
    learning_rates: Vec<f64>,
    epochs: Vec<usize>,
    l2s: Vec<f64>,
    init_scales: Vec<f64>,
    activations: Vec<Activation>,
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
    let train_cache = precompute_train_cache(common.seed, train, test, &train_configs);
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
            lookbacks: results::json_usize_array(src, "lookbacks")?,
            hidden_widths: results::json_usize_array(src, "hidden_width")?,
            learning_rates: results::json_f64_array(src, "learning_rate")?,
            epochs: results::json_usize_array(src, "epochs")?,
            l2s: results::json_f64_array(src, "l2")?,
            init_scales: results::json_f64_array(src, "init_scale")?,
            activations: results::json_string_array(src, "activation")?
                .iter()
                .map(|value| Activation::from_str(value))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn train_configs(&self) -> Vec<TrainConfig> {
        let mut configs = Vec::with_capacity(
            self.lookbacks.len()
                * self.hidden_widths.len()
                * self.learning_rates.len()
                * self.epochs.len()
                * self.l2s.len()
                * self.init_scales.len()
                * self.activations.len(),
        );
        for &lookback in &self.lookbacks {
            for &hidden_width in &self.hidden_widths {
                for &learning_rate in &self.learning_rates {
                    for &epochs in &self.epochs {
                        for &l2 in &self.l2s {
                            for &init_scale in &self.init_scales {
                                for &activation in &self.activations {
                                    configs.push(TrainConfig {
                                        lookback,
                                        hidden_width,
                                        learning_rate,
                                        epochs,
                                        l2,
                                        init_scale,
                                        activation,
                                    });
                                }
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
        Param {
            id,
            train: TrainConfig {
                lookback: self.lookbacks[rng.index(self.lookbacks.len())],
                hidden_width: self.hidden_widths[rng.index(self.hidden_widths.len())],
                learning_rate: self.learning_rates[rng.index(self.learning_rates.len())],
                epochs: self.epochs[rng.index(self.epochs.len())],
                l2: self.l2s[rng.index(self.l2s.len())],
                init_scale: self.init_scales[rng.index(self.init_scales.len())],
                activation: self.activations[rng.index(self.activations.len())],
            },
            score: ScoreConfig {
                threshold: thresholds[rng.index(thresholds.len())],
                fee_bps: common.fee_bps[rng.index(common.fee_bps.len())],
                slippage_bps: common.slippage_bps[rng.index(common.slippage_bps.len())],
                signal_rule,
            },
        }
    }
}

impl Activation {
    fn from_str(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tanh" => Ok(Self::Tanh),
            "relu" => Ok(Self::Relu),
            "linear" => Ok(Self::Linear),
            other => Err(format!("bad tide activation: {other}").into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Tanh => "tanh",
            Self::Relu => "relu",
            Self::Linear => "linear",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Tanh => 0,
            Self::Relu => 1,
            Self::Linear => 2,
        }
    }

    fn forward(self, value: f64) -> f64 {
        match self {
            Self::Tanh => value.tanh(),
            Self::Relu => value.max(0.0),
            Self::Linear => value,
        }
    }

    fn derivative(self, activated: f64) -> f64 {
        match self {
            Self::Tanh => 1.0 - activated * activated,
            Self::Relu => {
                if activated > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Linear => 1.0,
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
                "{{\"model\": \"tide\", \"lookback\": {}, \"hidden_width\": {}, \"learning_rate\": {}, \"epochs\": {}, \"l2\": {}, \"init_scale\": {}, \"activation\": \"{}\", \"id\": {}, \"threshold\": {}, \"fee_bps\": {}, \"slippage_bps\": {}, \"signal_rule\": \"{}\", \"mae\": {}, \"signal_rate_pct\": {}, \"backtest_total_return_net_pct\": {}, \"backtest_sharpe_per_bar\": {}, {}}}",
                param.train.lookback,
                param.train.hidden_width,
                results::json_float(param.train.learning_rate),
                param.train.epochs,
                results::json_float(param.train.l2),
                results::json_float(param.train.init_scale),
                param.train.activation.as_str(),
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

fn precompute_train_cache(
    seed: u64,
    train: &Matrix,
    test: &Matrix,
    configs: &[TrainConfig],
) -> Vec<FitOut> {
    let mut cache = Vec::new();
    let mut seen: HashSet<TrainKey> = HashSet::new();
    let mut windows: HashMap<usize, WindowData> = HashMap::new();
    for &config in configs {
        let key = cache_key(config);
        if !seen.insert(key) {
            continue;
        }
        if !windows.contains_key(&config.lookback) {
            windows.insert(
                config.lookback,
                build_window_data(train, test, config.lookback),
            );
        }
        let window = windows.get(&config.lookback).unwrap();
        let (preds, mae, fit_failed) = fit_predict(seed, window, test, config);
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
        config.lookback,
        config.hidden_width,
        config.learning_rate.to_bits(),
        config.epochs,
        config.l2.to_bits(),
        config.init_scale.to_bits(),
        config.activation.index(),
    )
}

fn build_window_data(train: &Matrix, test: &Matrix, lookback: usize) -> WindowData {
    let dims = lookback * train.cols;
    let mut row = vec![0.0; dims];
    let mut mean = vec![0.0; dims];
    let mut scale = vec![0.0; dims];

    for idx in 0..train.rows {
        fill_window(train, idx, lookback, &mut row);
        for col in 0..dims {
            mean[col] += row[col];
        }
    }
    for value in &mut mean {
        *value /= train.rows as f64;
    }

    for idx in 0..train.rows {
        fill_window(train, idx, lookback, &mut row);
        for col in 0..dims {
            let diff = row[col] - mean[col];
            scale[col] += diff * diff;
        }
    }
    for value in &mut scale {
        *value = (*value / train.rows as f64).sqrt().max(1e-12);
    }

    let y_mean = train.y.iter().sum::<f64>() / train.rows as f64;
    let mut y_scale = 0.0;
    for &value in &train.y {
        let diff = value - y_mean;
        y_scale += diff * diff;
    }
    y_scale = (y_scale / train.rows as f64).sqrt().max(1e-12);

    let mut train_x = vec![0.0; train.rows * dims];
    for idx in 0..train.rows {
        fill_window(train, idx, lookback, &mut row);
        normalize_row(
            &row,
            &mean,
            &scale,
            &mut train_x[idx * dims..(idx + 1) * dims],
        );
    }

    let mut test_x = vec![0.0; test.rows * dims];
    for idx in 0..test.rows {
        fill_window(test, idx, lookback, &mut row);
        normalize_row(
            &row,
            &mean,
            &scale,
            &mut test_x[idx * dims..(idx + 1) * dims],
        );
    }

    let train_y = train
        .y
        .iter()
        .map(|value| (value - y_mean) / y_scale)
        .collect();
    WindowData {
        train_rows: train.rows,
        test_rows: test.rows,
        dims,
        train_x,
        test_x,
        train_y,
        y_mean,
        y_scale,
    }
}

fn normalize_row(raw: &[f64], mean: &[f64], scale: &[f64], out: &mut [f64]) {
    for idx in 0..raw.len() {
        out[idx] = (raw[idx] - mean[idx]) / scale[idx];
    }
}

fn fill_window(data: &Matrix, row: usize, lookback: usize, out: &mut [f64]) {
    let cols = data.cols;
    for pos in 0..lookback {
        let source_row = window_row(row, pos, lookback);
        let source = source_row * cols;
        let target = pos * cols;
        out[target..target + cols].copy_from_slice(&data.x[source..source + cols]);
    }
}

fn window_row(row: usize, pos: usize, lookback: usize) -> usize {
    let available = (row + 1).min(lookback);
    let pad = lookback - available;
    if pos < pad {
        0
    } else {
        row + 1 - available + pos - pad
    }
}

fn fit_predict(
    seed: u64,
    data: &WindowData,
    test: &Matrix,
    config: TrainConfig,
) -> (Vec<f64>, f64, bool) {
    if config.lookback == 0 || config.hidden_width == 0 || config.epochs == 0 {
        return (Vec::new(), f64::NAN, true);
    }
    let mut net = init_network(seed, data.dims, config);
    train_network(&mut net, data, config);

    let mut preds = Vec::with_capacity(data.test_rows);
    let mut mae = 0.0;
    let mut hidden = vec![0.0; config.hidden_width];
    for row in 0..data.test_rows {
        let x = &data.test_x[row * data.dims..(row + 1) * data.dims];
        let pred =
            predict_scaled(&net, x, &mut hidden, config.activation) * data.y_scale + data.y_mean;
        mae += (pred - test.y[row]).abs();
        preds.push(pred);
    }
    (preds, mae / data.test_rows as f64, false)
}

fn init_network(seed: u64, dims: usize, config: TrainConfig) -> Network {
    let mut rng = SplitMix64::new(seed ^ config_seed(config));
    let hidden = config.hidden_width;
    let scale = config.init_scale / (dims as f64).sqrt();
    let mut w1 = vec![0.0; hidden * dims];
    let mut v = vec![0.0; hidden];
    let skip = vec![0.0; dims];
    for value in &mut w1 {
        *value = rand_weight(&mut rng, scale);
    }
    for value in &mut v {
        *value = rand_weight(&mut rng, config.init_scale / (hidden as f64).sqrt());
    }
    Network {
        w1,
        b1: vec![0.0; hidden],
        v,
        skip,
        out_bias: 0.0,
    }
}

fn train_network(net: &mut Network, data: &WindowData, config: TrainConfig) {
    let dims = data.dims;
    let hidden_len = config.hidden_width;
    let mut adam = Adam::new(dims, hidden_len);
    let mut hidden = vec![0.0; hidden_len];
    let mut grad_w1 = vec![0.0; hidden_len * dims];
    let mut grad_b1 = vec![0.0; hidden_len];
    let mut grad_v = vec![0.0; hidden_len];
    let mut grad_skip = vec![0.0; dims];
    let inv_rows = 1.0 / data.train_rows as f64;

    for step in 1..=config.epochs {
        grad_w1.fill(0.0);
        grad_b1.fill(0.0);
        grad_v.fill(0.0);
        grad_skip.fill(0.0);
        let mut grad_out = 0.0;

        for row in 0..data.train_rows {
            let x = &data.train_x[row * dims..(row + 1) * dims];
            fill_hidden(net, x, &mut hidden, config.activation);
            let pred = output_from_hidden(net, x, &hidden);
            let err = pred - data.train_y[row];
            grad_out += err;

            for col in 0..dims {
                grad_skip[col] += err * x[col];
            }
            for hid in 0..hidden_len {
                grad_v[hid] += err * hidden[hid];
                let dz = err * net.v[hid] * config.activation.derivative(hidden[hid]);
                grad_b1[hid] += dz;
                let base = hid * dims;
                for col in 0..dims {
                    grad_w1[base + col] += dz * x[col];
                }
            }
        }

        for idx in 0..grad_w1.len() {
            grad_w1[idx] = grad_w1[idx] * inv_rows + config.l2 * net.w1[idx];
        }
        for idx in 0..grad_b1.len() {
            grad_b1[idx] *= inv_rows;
        }
        for idx in 0..grad_v.len() {
            grad_v[idx] = grad_v[idx] * inv_rows + config.l2 * net.v[idx];
        }
        for idx in 0..grad_skip.len() {
            grad_skip[idx] = grad_skip[idx] * inv_rows + config.l2 * net.skip[idx];
        }
        grad_out *= inv_rows;

        adam.update(
            net,
            &grad_w1,
            &grad_b1,
            &grad_v,
            &grad_skip,
            grad_out,
            config.learning_rate,
            step,
        );
    }
}

impl Adam {
    fn new(dims: usize, hidden: usize) -> Self {
        Self {
            mw1: vec![0.0; hidden * dims],
            vw1: vec![0.0; hidden * dims],
            mb1: vec![0.0; hidden],
            vb1: vec![0.0; hidden],
            mv: vec![0.0; hidden],
            vv: vec![0.0; hidden],
            mskip: vec![0.0; dims],
            vskip: vec![0.0; dims],
            mout: 0.0,
            vout: 0.0,
        }
    }

    fn update(
        &mut self,
        net: &mut Network,
        grad_w1: &[f64],
        grad_b1: &[f64],
        grad_v: &[f64],
        grad_skip: &[f64],
        grad_out: f64,
        lr: f64,
        step: usize,
    ) {
        adam_update_slice(&mut net.w1, grad_w1, &mut self.mw1, &mut self.vw1, lr, step);
        adam_update_slice(&mut net.b1, grad_b1, &mut self.mb1, &mut self.vb1, lr, step);
        adam_update_slice(&mut net.v, grad_v, &mut self.mv, &mut self.vv, lr, step);
        adam_update_slice(
            &mut net.skip,
            grad_skip,
            &mut self.mskip,
            &mut self.vskip,
            lr,
            step,
        );
        adam_update_scalar(
            &mut net.out_bias,
            grad_out,
            &mut self.mout,
            &mut self.vout,
            lr,
            step,
        );
    }
}

fn predict_scaled(net: &Network, x: &[f64], hidden: &mut [f64], activation: Activation) -> f64 {
    fill_hidden(net, x, hidden, activation);
    output_from_hidden(net, x, hidden)
}

fn fill_hidden(net: &Network, x: &[f64], hidden: &mut [f64], activation: Activation) {
    let dims = x.len();
    for (hid, out) in hidden.iter_mut().enumerate() {
        let base = hid * dims;
        let mut value = net.b1[hid];
        for col in 0..dims {
            value += net.w1[base + col] * x[col];
        }
        *out = activation.forward(value);
    }
}

fn output_from_hidden(net: &Network, x: &[f64], hidden: &[f64]) -> f64 {
    let mut value = net.out_bias;
    for col in 0..x.len() {
        value += net.skip[col] * x[col];
    }
    for (hid, &activation) in hidden.iter().enumerate() {
        value += net.v[hid] * activation;
    }
    value
}

fn adam_update_slice(
    values: &mut [f64],
    grads: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    lr: f64,
    step: usize,
) {
    for idx in 0..values.len() {
        adam_update_scalar(
            &mut values[idx],
            grads[idx],
            &mut m[idx],
            &mut v[idx],
            lr,
            step,
        );
    }
}

fn adam_update_scalar(value: &mut f64, grad: f64, m: &mut f64, v: &mut f64, lr: f64, step: usize) {
    let beta1 = 0.9;
    let beta2 = 0.999;
    *m = beta1 * *m + (1.0 - beta1) * grad;
    *v = beta2 * *v + (1.0 - beta2) * grad * grad;
    let m_hat = *m / (1.0 - beta1.powi(step as i32));
    let v_hat = *v / (1.0 - beta2.powi(step as i32));
    *value -= lr * m_hat / (v_hat.sqrt() + 1e-8);
}

fn rand_weight(rng: &mut SplitMix64, scale: f64) -> f64 {
    let value = rng.index(2_000_001) as f64 / 1_000_000.0 - 1.0;
    value * scale
}

fn config_seed(config: TrainConfig) -> u64 {
    let mut seed = config.lookback as u64;
    seed ^= (config.hidden_width as u64).rotate_left(7);
    seed ^= config.learning_rate.to_bits().rotate_left(13);
    seed ^= (config.epochs as u64).rotate_left(19);
    seed ^= config.l2.to_bits().rotate_left(29);
    seed ^= config.init_scale.to_bits().rotate_left(37);
    seed ^ (config.activation.index() as u64).rotate_left(43)
}
