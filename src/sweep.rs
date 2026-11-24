use std::io::{self, Write};
use std::time::Instant;

use crate::results::{self, RunOutput};
use crate::{dlinear, lightgbm, ridge, tide, Matrix, Prices, SweepSummary};

/// The winning trials, cache counts, and timing of a completed sweep.
pub struct SweepResult {
    output: RunOutput,
    hot_seconds: f64,
}

impl SweepResult {
    /// Trial counts and per-metric winners.
    pub fn summary(&self) -> &SweepSummary {
        &self.output.summary
    }

    /// Number of distinct fitted configurations in the configured pool.
    pub fn training_configurations(&self) -> usize {
        self.output.unique_train_configs
    }

    /// Number of fits in the configured pool that failed.
    pub fn failed_training_configurations(&self) -> usize {
        self.output.failed_train_configs
    }

    /// Number of distinct training-key, threshold, and rule combinations cached.
    pub fn score_cache_entries(&self) -> usize {
        self.output.score_cache_entries
    }

    /// Seconds spent fitting and running the sweep, including winner diagnostics.
    pub fn elapsed_seconds(&self) -> f64 {
        self.hot_seconds
    }

    /// Write the same JSON receipt as the command-line runner.
    pub fn write_json(&self, output: impl Write) -> io::Result<()> {
        results::write_summary_json(output, &self.output, self.hot_seconds)
    }
}

/// Run the configured sweep on prepared matrices and aligned test prices.
///
/// The JSON fields and process environment use the command-line runner's
/// configuration, including `TERASWEEP_LEVEL` and backend paths.
pub fn sweep(
    config_json: &str,
    train: &Matrix,
    test: &Matrix,
    prices: &Prices,
    runs: usize,
) -> Result<SweepResult, Box<dyn std::error::Error>> {
    if !train.has_shape() || !test.has_shape() {
        return Err("Sweep matrices must have valid nonempty shapes".into());
    }
    if test.cols != train.cols {
        return Err(format!("test cols {} != train cols {}", test.cols, train.cols).into());
    }
    if prices.open.len() != test.rows || prices.close.len() != test.rows {
        return Err(format!(
            "price rows ({} open, {} close) != test rows {}",
            prices.open.len(),
            prices.close.len(),
            test.rows,
        )
        .into());
    }

    let price_stats = results::precompute_price_stats(prices);
    let hot_start = Instant::now();
    let run = match results::model_name(config_json)?
        .to_ascii_lowercase()
        .as_str()
    {
        "ridge" => ridge::run(config_json, train, test, &price_stats, runs)?,
        "dlinear" => dlinear::run(config_json, train, test, &price_stats, runs)?,
        "lightgbm" => lightgbm::run(config_json, train, test, &price_stats, runs)?,
        "tide" => tide::run(config_json, train, test, &price_stats, runs)?,
        other => return Err(format!("bad model value: {other}").into()),
    };
    let hot_seconds = hot_start.elapsed().as_secs_f64();

    Ok(SweepResult {
        output: run,
        hot_seconds,
    })
}
