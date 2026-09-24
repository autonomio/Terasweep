# Terasweep

Minimal Rust sweep runner for trillion-scale experiment permutations.

Upstream data is prepared by Limen; Terasweep only runs the hot sweep loop.

This package contains the minimum surface needed to repeat the cached sampled sweep:

- `src/sweep.rs`: orchestration, config, sampling, and summary output.
- `src/results.rs`: scoring, price accounting, fees, slippage, and metrics.
- `src/ridge.rs`: first model architecture.
- `src/dlinear.rs`: closed-form DLinear architecture.
- `src/lightgbm.rs`: native histogram GBDT architecture.
- `src/tide.rs`: minimal dense temporal encoder-decoder architecture.
- `data/train.bin`: frozen training matrix and target.
- `data/test.bin`: frozen test matrix and target.
- `data/price.bin`: frozen open/close backtest prices.
- `configs/sweep.json`: deterministic sampled sweep space.
- `configs/dlinear.json`: deterministic DLinear sampled sweep space.
- `configs/lightgbm.json`: deterministic LightGBM-style sampled sweep space.
- `configs/tide.json`: deterministic TiDE sampled sweep space.
- `golden/1m.json`: stable canonical 1M validation result.

Build and smoke test:

```sh
make smoke
```

Run a larger sweep:

```sh
make run RUNS=1000000 OUT=runs/1m
make run RUNS=10000000 OUT=runs/10m
make run RUNS=100000000 OUT=runs/100m
make run RUNS=1000000000 OUT=runs/1b
```

Run DLinear on the same frozen data:

```sh
make run RUNS=1000000 OUT=runs/dlinear-1m CONFIG=configs/dlinear.json
```

Run LightGBM-style GBDT on the same frozen data:

```sh
make run RUNS=1000000 OUT=runs/lightgbm-1m CONFIG=configs/lightgbm.json
```

Run TiDE on the same frozen data:

```sh
make run RUNS=1000000 OUT=runs/tide-1m CONFIG=configs/tide.json
```

Golden validation:

```sh
make verify
```

The runner writes `summary.json` and `time.txt` under the selected run directory.
