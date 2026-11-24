# Retained sweep inputs

## Ridge and LightGBM matrices

Frozen Limen-prepared BTCUSDT inputs used by the original Ridge sweep and the
bundled LightGBM configuration.

- Source: `HistoricalData.get_spot_klines(kline_size=900, row_count_limit=6000)`.
- Raw rows: 6000 15-minute bars.
- Raw window: `2026-03-13 12:00:00+00:00` to `2026-05-14 23:45:00+00:00`.
- Split: `7/1/2`.
- Train shape: `4136 x 55`.
- Validation shape in Limen prep: `536 x 55`; not consumed by this runner.
- Test shape: `1136 x 55`.
- Backtest price rows: `1136`.
- Backtest window: `2026-05-03 03:45:00+00:00` to `2026-05-14 23:30:00+00:00`.
- Target: `NextReturnTarget(periods=1, scale=100.0)`.
- Ridge model: sampled `alpha`, `fit_intercept`, and `solver_eps`.
- LightGBM regression: sampled `learning_rate`, `num_trees`, `num_leaves`, `max_bins`, `lambda_l2`, and `min_data_in_leaf`.

The frozen binaries let the sweep run without fetching and preparing the Limen
data again.

## DLinear and TiDE return series

Both use a single observed return column and a next-bar return target in
percentage points. DLinear fits an intercept and samples `alpha`, `solver_eps`,
`lookback`, and `kernel`. TiDE samples `lookback`, `hidden_width`,
`learning_rate`, and `epochs`. [Input preparation](../backends.md#return-inputs)

## Shared scoring

The model's [sweep configuration](../../configs/) defines the search space,
including `signal_rule`, `threshold`, `fee_bps`, and `slippage_bps`. `RUNS`
supplies the trial count. Outputs are rounded to 3 decimals for MAE and signal
rate, 1 for return, and 2 for Sharpe.

The [matched sweep benchmark](../../benchmarks/sweep-scaling/README.md) slices
these inputs to 256 training and 128 test rows per model and records the exact
fixture hashes and grids.
