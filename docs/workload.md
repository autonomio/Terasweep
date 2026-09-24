# Workload

Frozen Limen-prepared BTCUSDT ridge sweep input.

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
- DLinear model: sampled `alpha`, `fit_intercept`, `solver_eps`, `lookback`, and `kernel`.
- LightGBM-style model: sampled `learning_rate`, `num_trees`, `num_leaves`, `max_bins`, `lambda_l2`, and `min_data_in_leaf`.
- Score knobs: `signal_rule`, `threshold`, `fee_bps`, `slippage_bps`.
- Run count is supplied by `RUNS`; `configs/sweep.json` defines the fixed search space.
- Output rounding is fixed: MAE `3`, signal rate `3`, return `1`, Sharpe `2`.

The frozen binaries are intentionally included so the sweep can be repeated without re-fetching or re-preparing Limen data.
