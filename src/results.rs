use std::collections::HashMap;
use std::fs;
use std::io::Write;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SignalRule {
    Gt,
    Gte,
    Lt,
    Lte,
    AbsGt,
    AbsGte,
}

pub struct Prices {
    pub open: Vec<f64>,
    pub close: Vec<f64>,
}

pub struct PriceStats {
    pub(crate) tradable: Vec<bool>,
    pub(crate) r_entry: Vec<f64>,
    pub(crate) r_cont: Vec<f64>,
}

pub struct RunOutput {
    pub summary: SweepSummary,
    pub unique_alphas: usize,
    pub unique_train_configs: usize,
    pub failed_train_configs: usize,
    pub score_cache_entries: usize,
    pub best_return_json: String,
    pub best_sharpe_json: String,
    pub best_mae_json: String,
}

pub struct CommonSpace {
    pub seed: u64,
    pub signal_rules: Vec<SignalRule>,
    pub threshold_signed: Vec<f64>,
    pub threshold_abs: Vec<f64>,
    pub fee_bps: Vec<f64>,
    pub slippage_bps: Vec<f64>,
    pub rolling_window_bars: usize,
    pub bar_minutes: f64,
}

#[derive(Clone, Copy)]
pub struct ScoreConfig {
    pub threshold: f64,
    pub fee_bps: f64,
    pub slippage_bps: f64,
    pub signal_rule: SignalRule,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct ScoreKey {
    pub train_idx: usize,
    pub threshold_bits: u64,
    pub rule_idx: usize,
}

#[derive(Clone, Copy)]
pub struct BaseScore {
    pub(crate) signal_count: usize,
    pub(crate) eq_gross: f64,
    pub(crate) entry_count: i32,
    pub(crate) exit_count: i32,
    pub(crate) sharpe_count: usize,
    pub(crate) cat_count: [usize; 4],
    pub(crate) cat_sum_a: [f64; 4],
    pub(crate) cat_sum_a2: [f64; 4],
}

#[derive(Clone, Copy)]
pub struct ScoredRow {
    pub id: usize,
    pub signal_rule: SignalRule,
    pub fit_failed: bool,
    pub mae: f64,
    pub signal_rate: f64,
    pub net_return: f64,
    pub sharpe: f64,
}

#[derive(Clone, Copy)]
pub struct BacktestStats {
    pub edge_per_signal_n: usize,
    pub edge_per_signal_bps_p5: f64,
    pub edge_per_signal_bps_p50: f64,
    pub edge_per_signal_bps_p95: f64,
    pub trade_n: usize,
    pub trade_pnl_net_bps_p5: f64,
    pub trade_pnl_net_bps_p50: f64,
    pub trade_pnl_net_bps_p95: f64,
    pub cost_drag_n: usize,
    pub cost_drag_bps_p5: f64,
    pub cost_drag_bps_p50: f64,
    pub cost_drag_bps_p95: f64,
    pub rolling_return_n: usize,
    pub rolling_return_net_pct_p5: f64,
    pub rolling_return_net_pct_p50: f64,
    pub rolling_return_net_pct_p95: f64,
    pub return_on_exposure_n: usize,
    pub return_on_exposure_p5: f64,
    pub return_on_exposure_p50: f64,
    pub return_on_exposure_p95: f64,
    pub drawdown_n: usize,
    pub drawdown_depth_pct_p5: f64,
    pub drawdown_depth_pct_p50: f64,
    pub drawdown_depth_pct_p95: f64,
    pub drawdown_duration_days_p5: f64,
    pub drawdown_duration_days_p50: f64,
    pub drawdown_duration_days_p95: f64,
    pub cvar_95_return_pct: f64,
}

pub struct SweepSummary {
    rows: usize,
    failed_rows: usize,
    signal_rule_counts: [usize; 6],
    best_return: Option<ScoredRow>,
    best_sharpe: Option<ScoredRow>,
    best_mae: Option<ScoredRow>,
}

impl SweepSummary {
    pub fn new() -> Self {
        Self {
            rows: 0,
            failed_rows: 0,
            signal_rule_counts: [0; 6],
            best_return: None,
            best_sharpe: None,
            best_mae: None,
        }
    }

    pub fn record(&mut self, row: ScoredRow) {
        self.rows += 1;
        self.signal_rule_counts[row.signal_rule.index()] += 1;
        if row.fit_failed {
            self.failed_rows += 1;
            return;
        }
        if row.net_return.is_finite()
            && self
                .best_return
                .map_or(true, |best| row.net_return > best.net_return)
        {
            self.best_return = Some(row);
        }
        if row.sharpe.is_finite()
            && self
                .best_sharpe
                .map_or(true, |best| row.sharpe > best.sharpe)
        {
            self.best_sharpe = Some(row);
        }
        if row.mae.is_finite() && self.best_mae.map_or(true, |best| row.mae < best.mae) {
            self.best_mae = Some(row);
        }
    }

    pub fn best_return(&self) -> Option<ScoredRow> {
        self.best_return
    }

    pub fn best_sharpe(&self) -> Option<ScoredRow> {
        self.best_sharpe
    }

    pub fn best_mae(&self) -> Option<ScoredRow> {
        self.best_mae
    }
}

impl CommonSpace {
    pub fn from_json(src: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let signed_section = json_object(src, "threshold_signed")?;
        let abs_section = json_object(src, "threshold_abs")?;
        Ok(Self {
            seed: json_u64(src, "seed")?,
            signal_rules: json_signal_rule_array(src, "signal_rule")?,
            threshold_signed: grid_values(
                json_f64(signed_section, "min")?,
                json_f64(signed_section, "max")?,
                json_f64(signed_section, "step")?,
            ),
            threshold_abs: grid_values(
                json_f64(abs_section, "min")?,
                json_f64(abs_section, "max")?,
                json_f64(abs_section, "step")?,
            ),
            fee_bps: json_f64_array(src, "fee_bps")?,
            slippage_bps: json_f64_array(src, "slippage_bps")?,
            rolling_window_bars: json_usize_optional(src, "rolling_window_bars")?.unwrap_or(96),
            bar_minutes: json_f64_optional(src, "bar_minutes")?.unwrap_or(15.0),
        })
    }
}

impl SignalRule {
    pub fn as_str(self) -> &'static str {
        match self {
            SignalRule::Gt => "gt",
            SignalRule::Gte => "gte",
            SignalRule::Lt => "lt",
            SignalRule::Lte => "lte",
            SignalRule::AbsGt => "abs_gt",
            SignalRule::AbsGte => "abs_gte",
        }
    }

    pub fn index(self) -> usize {
        match self {
            SignalRule::Gt => 0,
            SignalRule::Gte => 1,
            SignalRule::Lt => 2,
            SignalRule::Lte => 3,
            SignalRule::AbsGt => 4,
            SignalRule::AbsGte => 5,
        }
    }

    pub fn is_abs(self) -> bool {
        matches!(self, SignalRule::AbsGt | SignalRule::AbsGte)
    }
}

pub fn precompute_price_stats(prices: &Prices) -> PriceStats {
    let n = prices.open.len();
    let mut tradable = vec![false; n];
    let mut r_entry = vec![0.0; n];
    let mut r_cont = vec![0.0; n];

    for i in 0..n {
        tradable[i] =
            prices.open[i].is_finite() && prices.close[i].is_finite() && prices.open[i] != 0.0;
        if tradable[i] {
            r_entry[i] = (prices.close[i] - prices.open[i]) / prices.open[i];
        }
        if i > 0 && prices.close[i - 1].is_finite() && prices.close[i - 1] != 0.0 {
            r_cont[i] = prices.close[i] / prices.close[i - 1] - 1.0;
        }
    }

    PriceStats {
        tradable,
        r_entry,
        r_cont,
    }
}

pub fn compute_base_score(
    preds: &[f64],
    threshold: f64,
    rule: SignalRule,
    prices: &PriceStats,
) -> BaseScore {
    let n = preds.len();
    let mut base = BaseScore {
        signal_count: 0,
        eq_gross: 1.0,
        entry_count: 0,
        exit_count: 0,
        sharpe_count: 0,
        cat_count: [0; 4],
        cat_sum_a: [0.0; 4],
        cat_sum_a2: [0.0; 4],
    };

    for pred in preds {
        if signal_active(*pred, threshold, rule) {
            base.signal_count += 1;
        }
    }

    let mut prev_pos = false;
    for i in 0..n {
        let pos = if i > 0 && prices.tradable[i] {
            signal_active(preds[i - 1], threshold, rule)
        } else {
            false
        };
        let next_pos = if i + 1 < n && prices.tradable[i + 1] {
            signal_active(preds[i], threshold, rule)
        } else {
            false
        };
        let entry = pos && !prev_pos;
        let cont = pos && prev_pos;
        let exit = pos && !next_pos;

        let mut r_gross = 0.0;
        if entry {
            r_gross += prices.r_entry[i];
            base.entry_count += 1;
        }
        if cont {
            r_gross += prices.r_cont[i];
        }
        if exit {
            base.exit_count += 1;
        }

        let a = 1.0 + r_gross;
        base.eq_gross *= a;
        if i > 0 && prices.tradable[i] {
            let cat = (entry as usize) | ((exit as usize) << 1);
            base.sharpe_count += 1;
            base.cat_count[cat] += 1;
            base.cat_sum_a[cat] += a;
            base.cat_sum_a2[cat] += a * a;
        }
        prev_pos = pos;
    }

    base
}

pub fn apply_costs(base: &BaseScore, pred_count: usize, config: ScoreConfig) -> (f64, f64, f64) {
    let fee = config.fee_bps / 10_000.0;
    let slip = config.slippage_bps / 10_000.0;
    let entry_mult = (1.0 - fee) / (1.0 + slip);
    let exit_mult = (1.0 - fee) * (1.0 - slip);
    let cat_mult = [1.0, entry_mult, exit_mult, entry_mult * exit_mult];

    let eq_net =
        base.eq_gross * entry_mult.powi(base.entry_count) * exit_mult.powi(base.exit_count);
    let net_return = round_to((eq_net - 1.0) * 100.0, 1);
    let signal_rate = round_to(base.signal_count as f64 * 100.0 / pred_count as f64, 3);

    let sharpe = if base.sharpe_count > 1 {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for (idx, mult) in cat_mult.iter().enumerate() {
            let count = base.cat_count[idx] as f64;
            if count == 0.0 {
                continue;
            }
            sum += mult * base.cat_sum_a[idx] - count;
            sum_sq += mult * mult * base.cat_sum_a2[idx] - 2.0 * mult * base.cat_sum_a[idx] + count;
        }
        let count = base.sharpe_count as f64;
        let mean = sum / count;
        let var = (sum_sq - count * mean * mean) / (count - 1.0);
        let sd = var.max(0.0).sqrt();
        if sd > 0.0 {
            round_to(mean / sd, 2)
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    (signal_rate, net_return, sharpe)
}

pub fn score_cached(
    cache_idx: usize,
    preds: &[f64],
    score: ScoreConfig,
    score_cache: &mut HashMap<ScoreKey, BaseScore>,
    prices: &PriceStats,
) -> (f64, f64, f64) {
    let score_key = ScoreKey {
        train_idx: cache_idx,
        threshold_bits: score.threshold.to_bits(),
        rule_idx: score.signal_rule.index(),
    };
    let base = match score_cache.get(&score_key) {
        Some(base) => *base,
        None => {
            let base = compute_base_score(preds, score.threshold, score.signal_rule, prices);
            score_cache.insert(score_key, base);
            base
        }
    };
    apply_costs(&base, preds.len(), score)
}

pub fn compute_backtest_stats(
    preds: &[f64],
    score: ScoreConfig,
    prices: &PriceStats,
    common: &CommonSpace,
) -> BacktestStats {
    let n = preds.len();
    let fee = score.fee_bps / 10_000.0;
    let slip = score.slippage_bps / 10_000.0;
    let entry_mult = (1.0 - fee) / (1.0 + slip);
    let exit_mult = (1.0 - fee) * (1.0 - slip);
    let cat_mult = [1.0, entry_mult, exit_mult, entry_mult * exit_mult];

    let mut edge_bps = Vec::new();
    let mut trade_net_bps = Vec::new();
    let mut cost_drag_bps = Vec::new();
    let mut net_returns = vec![0.0; n];
    let mut exposure = vec![false; n];
    let mut equity = Vec::with_capacity(n + 1);
    equity.push(1.0);

    let mut in_trade = false;
    let mut trade_gross = 1.0;
    let mut trade_net = 1.0;
    let mut prev_pos = false;

    for i in 0..n {
        let pos = if i > 0 && prices.tradable[i] {
            signal_active(preds[i - 1], score.threshold, score.signal_rule)
        } else {
            false
        };
        let next_pos = if i + 1 < n && prices.tradable[i + 1] {
            signal_active(preds[i], score.threshold, score.signal_rule)
        } else {
            false
        };
        let entry = pos && !prev_pos;
        let cont = pos && prev_pos;
        let exit = pos && !next_pos;
        let cat = (entry as usize) | ((exit as usize) << 1);

        let mut gross_r = 0.0;
        if entry {
            gross_r += prices.r_entry[i];
        }
        if cont {
            gross_r += prices.r_cont[i];
        }
        let net_r = (1.0 + gross_r) * cat_mult[cat] - 1.0;
        net_returns[i] = net_r;
        exposure[i] = pos && prices.tradable[i];
        equity.push(equity[i] * (1.0 + net_r));

        if exposure[i] {
            edge_bps.push(gross_r * 10_000.0);
            if entry {
                in_trade = true;
                trade_gross = 1.0;
                trade_net = 1.0;
            }
            if in_trade {
                trade_gross *= 1.0 + gross_r;
                trade_net *= 1.0 + net_r;
            }
            if exit && in_trade {
                trade_net_bps.push((trade_net - 1.0) * 10_000.0);
                cost_drag_bps.push((trade_gross - trade_net) * 10_000.0);
                in_trade = false;
            }
        }
        prev_pos = pos;
    }

    let (rolling_return_net_pct, return_on_exposure) =
        rolling_returns(&net_returns, &exposure, common.rolling_window_bars.max(1));
    let (drawdown_depth_pct, drawdown_duration_days) =
        drawdowns(&equity, common.bar_minutes.max(0.0));

    BacktestStats {
        edge_per_signal_n: edge_bps.len(),
        edge_per_signal_bps_p5: round_to(percentile(&mut edge_bps, 0.05), 3),
        edge_per_signal_bps_p50: round_to(percentile(&mut edge_bps, 0.50), 3),
        edge_per_signal_bps_p95: round_to(percentile(&mut edge_bps, 0.95), 3),
        trade_n: trade_net_bps.len(),
        trade_pnl_net_bps_p5: round_to(percentile(&mut trade_net_bps, 0.05), 3),
        trade_pnl_net_bps_p50: round_to(percentile(&mut trade_net_bps, 0.50), 3),
        trade_pnl_net_bps_p95: round_to(percentile(&mut trade_net_bps, 0.95), 3),
        cost_drag_n: cost_drag_bps.len(),
        cost_drag_bps_p5: round_to(percentile(&mut cost_drag_bps, 0.05), 3),
        cost_drag_bps_p50: round_to(percentile(&mut cost_drag_bps, 0.50), 3),
        cost_drag_bps_p95: round_to(percentile(&mut cost_drag_bps, 0.95), 3),
        rolling_return_n: rolling_return_net_pct.len(),
        rolling_return_net_pct_p5: round_to(
            percentile(&mut rolling_return_net_pct.clone(), 0.05),
            3,
        ),
        rolling_return_net_pct_p50: round_to(
            percentile(&mut rolling_return_net_pct.clone(), 0.50),
            3,
        ),
        rolling_return_net_pct_p95: round_to(
            percentile(&mut rolling_return_net_pct.clone(), 0.95),
            3,
        ),
        return_on_exposure_n: return_on_exposure.len(),
        return_on_exposure_p5: round_to(percentile(&mut return_on_exposure.clone(), 0.05), 3),
        return_on_exposure_p50: round_to(percentile(&mut return_on_exposure.clone(), 0.50), 3),
        return_on_exposure_p95: round_to(percentile(&mut return_on_exposure.clone(), 0.95), 3),
        drawdown_n: drawdown_depth_pct.len(),
        drawdown_depth_pct_p5: round_to(percentile(&mut drawdown_depth_pct.clone(), 0.05), 3),
        drawdown_depth_pct_p50: round_to(percentile(&mut drawdown_depth_pct.clone(), 0.50), 3),
        drawdown_depth_pct_p95: round_to(percentile(&mut drawdown_depth_pct.clone(), 0.95), 3),
        drawdown_duration_days_p5: round_to(
            percentile(&mut drawdown_duration_days.clone(), 0.05),
            3,
        ),
        drawdown_duration_days_p50: round_to(
            percentile(&mut drawdown_duration_days.clone(), 0.50),
            3,
        ),
        drawdown_duration_days_p95: round_to(
            percentile(&mut drawdown_duration_days.clone(), 0.95),
            3,
        ),
        cvar_95_return_pct: round_to(cvar_95(&rolling_return_net_pct), 3),
    }
}

pub fn backtest_stats_json(stats: BacktestStats) -> String {
    format!(
        "\"edge_per_signal_n\": {}, \"edge_per_signal_bps_p5\": {}, \"edge_per_signal_bps_p50\": {}, \"edge_per_signal_bps_p95\": {}, \"trade_n\": {}, \"trade_pnl_net_bps_p5\": {}, \"trade_pnl_net_bps_p50\": {}, \"trade_pnl_net_bps_p95\": {}, \"cost_drag_n\": {}, \"cost_drag_bps_p5\": {}, \"cost_drag_bps_p50\": {}, \"cost_drag_bps_p95\": {}, \"rolling_return_n\": {}, \"rolling_return_net_pct_p5\": {}, \"rolling_return_net_pct_p50\": {}, \"rolling_return_net_pct_p95\": {}, \"return_on_exposure_n\": {}, \"return_on_exposure_p5\": {}, \"return_on_exposure_p50\": {}, \"return_on_exposure_p95\": {}, \"drawdown_n\": {}, \"drawdown_depth_pct_p5\": {}, \"drawdown_depth_pct_p50\": {}, \"drawdown_depth_pct_p95\": {}, \"drawdown_duration_days_p5\": {}, \"drawdown_duration_days_p50\": {}, \"drawdown_duration_days_p95\": {}, \"cvar_95_return_pct\": {}",
        stats.edge_per_signal_n,
        json_float(stats.edge_per_signal_bps_p5),
        json_float(stats.edge_per_signal_bps_p50),
        json_float(stats.edge_per_signal_bps_p95),
        stats.trade_n,
        json_float(stats.trade_pnl_net_bps_p5),
        json_float(stats.trade_pnl_net_bps_p50),
        json_float(stats.trade_pnl_net_bps_p95),
        stats.cost_drag_n,
        json_float(stats.cost_drag_bps_p5),
        json_float(stats.cost_drag_bps_p50),
        json_float(stats.cost_drag_bps_p95),
        stats.rolling_return_n,
        json_float(stats.rolling_return_net_pct_p5),
        json_float(stats.rolling_return_net_pct_p50),
        json_float(stats.rolling_return_net_pct_p95),
        stats.return_on_exposure_n,
        json_float(stats.return_on_exposure_p5),
        json_float(stats.return_on_exposure_p50),
        json_float(stats.return_on_exposure_p95),
        stats.drawdown_n,
        json_float(stats.drawdown_depth_pct_p5),
        json_float(stats.drawdown_depth_pct_p50),
        json_float(stats.drawdown_depth_pct_p95),
        json_float(stats.drawdown_duration_days_p5),
        json_float(stats.drawdown_duration_days_p50),
        json_float(stats.drawdown_duration_days_p95),
        json_float(stats.cvar_95_return_pct)
    )
}

pub fn round_to(value: f64, decimals: i32) -> f64 {
    let factor = 10_f64.powi(decimals);
    (value * factor).round() / factor
}

fn rolling_returns(net_returns: &[f64], exposure: &[bool], window: usize) -> (Vec<f64>, Vec<f64>) {
    if net_returns.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let w = window.min(net_returns.len()).max(1);
    let mut returns = Vec::new();
    let mut roe = Vec::new();
    for start in 0..=net_returns.len() - w {
        let mut eq = 1.0;
        let mut exposed = 0usize;
        for idx in start..start + w {
            eq *= 1.0 + net_returns[idx];
            if exposure[idx] {
                exposed += 1;
            }
        }
        let pct = (eq - 1.0) * 100.0;
        returns.push(pct);
        if exposed > 0 {
            roe.push(pct / (exposed as f64 / w as f64));
        }
    }
    (returns, roe)
}

fn drawdowns(equity: &[f64], bar_minutes: f64) -> (Vec<f64>, Vec<f64>) {
    let mut depths = Vec::new();
    let mut durations = Vec::new();
    if equity.len() < 2 {
        return (depths, durations);
    }

    let mut peak = equity[0];
    let mut peak_idx = 0usize;
    let mut in_drawdown = false;
    let mut max_depth = 0.0;

    for (idx, &value) in equity.iter().enumerate().skip(1) {
        if value >= peak {
            if in_drawdown {
                depths.push(max_depth * 100.0);
                durations.push((idx - peak_idx) as f64 * bar_minutes / 1440.0);
                in_drawdown = false;
                max_depth = 0.0;
            }
            peak = value;
            peak_idx = idx;
        } else if peak > 0.0 {
            in_drawdown = true;
            let depth = 1.0 - value / peak;
            if depth > max_depth {
                max_depth = depth;
            }
        }
    }

    if in_drawdown {
        depths.push(max_depth * 100.0);
        durations.push((equity.len() - 1 - peak_idx) as f64 * bar_minutes / 1440.0);
    }

    (depths, durations)
}

fn percentile(values: &mut Vec<f64>, p: f64) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    if values.len() == 1 {
        return values[0];
    }
    let pos = p.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let weight = pos - lo as f64;
        values[lo] * (1.0 - weight) + values[hi] * weight
    }
}

fn cvar_95(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    let count = ((sorted.len() as f64) * 0.05).ceil().max(1.0) as usize;
    sorted.iter().take(count).sum::<f64>() / count as f64
}

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    pub fn index(&mut self, len: usize) -> usize {
        ((self.next_u64() as u128 * len as u128) >> 64) as usize
    }
}

pub fn mix_id(id: u64) -> u64 {
    let mut rng = SplitMix64::new(id);
    rng.next_u64()
}

pub fn model_name(src: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(json_string_optional(src, "model")?.unwrap_or_else(|| "ridge".to_string()))
}

pub fn logspace(log_min: f64, log_max: f64, count: usize) -> Vec<f64> {
    if count == 1 {
        return vec![10_f64.powf(log_min)];
    }
    (0..count)
        .map(|idx| {
            let t = idx as f64 / (count - 1) as f64;
            10_f64.powf(log_min + (log_max - log_min) * t)
        })
        .collect()
}

pub fn grid_values(min: f64, max: f64, step: f64) -> Vec<f64> {
    let mut values = Vec::new();
    let mut value = min;
    while value <= max + step * 0.5 {
        values.push(round_to(value, 12));
        value += step;
    }
    values
}

pub fn json_object<'a>(src: &'a str, key: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    let start = json_value_start(src, key)?;
    let open = src[start..]
        .find('{')
        .ok_or(format!("json key {key} is not an object"))?
        + start;
    let close = matching_bracket(src, open, '{', '}')?;
    Ok(&src[open..=close])
}

pub fn json_f64(src: &str, key: &str) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(json_number_text(src, key)?.parse()?)
}

pub fn json_usize(src: &str, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(json_number_text(src, key)?.parse()?)
}

pub fn json_u64(src: &str, key: &str) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(json_number_text(src, key)?.parse()?)
}

pub fn json_f64_optional(src: &str, key: &str) -> Result<Option<f64>, Box<dyn std::error::Error>> {
    if has_json_key(src, key) {
        Ok(Some(json_f64(src, key)?))
    } else {
        Ok(None)
    }
}

pub fn json_usize_optional(
    src: &str,
    key: &str,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    if has_json_key(src, key) {
        Ok(Some(json_usize(src, key)?))
    } else {
        Ok(None)
    }
}

pub fn json_f64_array(src: &str, key: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    json_array(src, key)?
        .split(',')
        .map(|value| Ok(value.trim().parse()?))
        .collect()
}

pub fn json_usize_array(src: &str, key: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    json_array(src, key)?
        .split(',')
        .map(|value| Ok(value.trim().parse()?))
        .collect()
}

pub fn json_bool_array(src: &str, key: &str) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
    json_array(src, key)?
        .split(',')
        .map(|value| parse_bool(value.trim()))
        .collect()
}

pub fn json_string_array(src: &str, key: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    json_array(src, key)?
        .split(',')
        .map(|value| {
            let value = value.trim();
            if value.len() < 2 || !value.starts_with('"') || !value.ends_with('"') {
                return Err(format!("json array value for {key} is not a string").into());
            }
            Ok(value[1..value.len() - 1].to_string())
        })
        .collect()
}

fn has_json_key(src: &str, key: &str) -> bool {
    src.contains(&format!("\"{key}\""))
}

fn json_array<'a>(src: &'a str, key: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    let start = json_value_start(src, key)?;
    let open = src[start..]
        .find('[')
        .ok_or(format!("json key {key} is not an array"))?
        + start;
    let close = matching_bracket(src, open, '[', ']')?;
    Ok(&src[open + 1..close])
}

fn json_value_start(src: &str, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let needle = format!("\"{key}\"");
    let key_pos = src
        .find(&needle)
        .ok_or(format!("json missing key: {key}"))?;
    let colon = src[key_pos + needle.len()..]
        .find(':')
        .ok_or(format!("json key missing colon: {key}"))?
        + key_pos
        + needle.len();
    Ok(colon + 1)
}

fn json_string_optional(
    src: &str,
    key: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if !has_json_key(src, key) {
        return Ok(None);
    }
    let mut idx = json_value_start(src, key)?;
    let bytes = src.as_bytes();
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx >= bytes.len() || bytes[idx] != b'"' {
        return Err(format!("json key is not a string: {key}").into());
    }
    idx += 1;
    let start = idx;
    while idx < bytes.len() && bytes[idx] != b'"' {
        idx += 1;
    }
    if idx >= bytes.len() {
        return Err(format!("json string did not close: {key}").into());
    }
    Ok(Some(src[start..idx].to_string()))
}

fn matching_bracket(
    src: &str,
    open: usize,
    left: char,
    right: char,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (offset, ch) in src[open..].char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
        } else if ch == left {
            depth += 1;
        } else if ch == right {
            depth -= 1;
            if depth == 0 {
                return Ok(open + offset);
            }
        }
    }
    Err("json bracket did not close".into())
}

fn json_number_text<'a>(src: &'a str, key: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    let mut idx = json_value_start(src, key)?;
    let bytes = src.as_bytes();
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    let start = idx;
    while idx < bytes.len() {
        let ch = bytes[idx] as char;
        if ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E') {
            idx += 1;
        } else {
            break;
        }
    }
    if start == idx {
        return Err(format!("json key is not numeric: {key}").into());
    }
    Ok(&src[start..idx])
}

fn parse_bool(value: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        other => Err(format!("bad bool value: {other}").into()),
    }
}

fn json_signal_rule_array(
    src: &str,
    key: &str,
) -> Result<Vec<SignalRule>, Box<dyn std::error::Error>> {
    json_array(src, key)?
        .split(',')
        .map(|value| parse_signal_rule(value.trim().trim_matches('"')))
        .collect()
}

fn parse_signal_rule(value: &str) -> Result<SignalRule, Box<dyn std::error::Error>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "gt" | ">" => Ok(SignalRule::Gt),
        "gte" | ">=" => Ok(SignalRule::Gte),
        "lt" | "<" => Ok(SignalRule::Lt),
        "lte" | "<=" => Ok(SignalRule::Lte),
        "abs_gt" | "abs>" => Ok(SignalRule::AbsGt),
        "abs_gte" | "abs>=" => Ok(SignalRule::AbsGte),
        other => Err(format!("bad signal_rule value: {other}").into()),
    }
}

fn signal_active(pred: f64, threshold: f64, rule: SignalRule) -> bool {
    match rule {
        SignalRule::Gt => pred > threshold,
        SignalRule::Gte => pred >= threshold,
        SignalRule::Lt => pred < threshold,
        SignalRule::Lte => pred <= threshold,
        SignalRule::AbsGt => pred.abs() > threshold,
        SignalRule::AbsGte => pred.abs() >= threshold,
    }
}

pub fn write_summary_json(
    path: &str,
    run: &RunOutput,
    hot_seconds: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = fs::File::create(path)?;
    let summary = &run.summary;
    writeln!(out, "{{")?;
    writeln!(out, "  \"rows\": {},", summary.rows)?;
    writeln!(out, "  \"failed_rows\": {},", summary.failed_rows)?;
    writeln!(
        out,
        "  \"valid_rows\": {},",
        summary.rows - summary.failed_rows
    )?;
    writeln!(out, "  \"hot_loop_seconds\": {:.12},", hot_seconds)?;
    writeln!(
        out,
        "  \"hot_loop_iterations_per_second\": {:.6},",
        summary.rows as f64 / hot_seconds
    )?;
    writeln!(out, "  \"unique_alphas\": {},", run.unique_alphas)?;
    writeln!(
        out,
        "  \"unique_train_configs\": {},",
        run.unique_train_configs
    )?;
    writeln!(
        out,
        "  \"failed_train_configs\": {},",
        run.failed_train_configs
    )?;
    writeln!(
        out,
        "  \"score_cache_entries\": {},",
        run.score_cache_entries
    )?;
    writeln!(
        out,
        "  \"signal_rule_counts\": {{\"gt\": {}, \"gte\": {}, \"lt\": {}, \"lte\": {}, \"abs_gt\": {}, \"abs_gte\": {}}},",
        summary.signal_rule_counts[0],
        summary.signal_rule_counts[1],
        summary.signal_rule_counts[2],
        summary.signal_rule_counts[3],
        summary.signal_rule_counts[4],
        summary.signal_rule_counts[5]
    )?;
    writeln!(out, "  \"best_return\": {},", run.best_return_json)?;
    writeln!(out, "  \"best_sharpe\": {},", run.best_sharpe_json)?;
    writeln!(out, "  \"best_mae\": {}", run.best_mae_json)?;
    writeln!(out, "}}")?;
    Ok(())
}

pub fn json_float(value: f64) -> String {
    if value.is_finite() {
        format!("{}", value)
    } else {
        "null".to_string()
    }
}
