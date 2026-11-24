# Matched sweep scaling

This experiment measures Terasweep’s compact scalar sweep on the M1 Max and an uncached model reference on the same input matrices, trial order, signal rules, and costs. Each reference trial initializes and fits a fresh canonical model, predicts the test rows, and runs Terasweep’s scalar scorer without a score cache. The reference uses scikit-learn Ridge through Limen, Limen DLinear, Python LightGBM regression, or Google Research TiDE, depending on the card.

## One billion sampled trials

| Model | Terasweep measured | Reference projected | Reference calibration |
| --- | ---: | ---: | ---: |
| [Ridge](../../docs/models/ridge.md#sweep-scaling) | 51.380 s | 2.57 days | 10,000 |
| [DLinear](../../docs/models/dlinear.md#sweep-scaling) | 72.456 s | 2.38 days | 10,000 |
| [LightGBM](../../docs/models/lightgbm.md#sweep-scaling) | 77.479 s | 36.51 days | 10,000 |
| [TiDE](../../docs/models/tide.md#sweep-scaling) | 75.606 s | 4,018.83 days | 1,000 |

## Workload

Each case uses the first 256 training and 128 test rows of its retained market-data fixture. Ridge uses the archived Limen matrix, LightGBM uses the bundled 55-feature matrix, and DLinear and TiDE use the retained single-return series. Test prices are sliced to the same rows. The benchmark receipt records the fixture hashes and exact grids. The production samplers draw with replacement from those grids using seed `20260514`. Each requested size is a prefix of the same deterministic draw sequence.

The benchmark’s Rust bindings call the production sampler and scalar scorer directly. Validation checks reference predictions, the signal vectors for every distinct fitted model, rule and threshold in each measured reference prefix, and the IDs and rounded metrics for the winning return, Sharpe and MAE trials. The reference comparison uses `rtol=atol=1e-10` and additionally requires bit-identical LightGBM and TiDE predictions. The receipt records each validation result.

## Timing and projections

Each timing comes from a separate process. Measured sweep seconds exclude loading the prepared matrices and prices, preparing price statistics, and writing the summary. They include backend initialization, fitting, predicting, scoring, and reducing the trial results. The Terasweep time also includes its winner diagnostics. Reference measurements record the backend setup and the trial loop separately. Both paths run serially, with the scalar CPU scorer and single-threaded model backends on this host. The shortest measurements include disproportionate backend startup time, which is recorded separately for the reference in the receipt.

The reference is measured at 10, 100, and 1,000 trials. The 10,000-trial reference runs when the 1,000-trial calibration predicts a runtime of at most 150 seconds. Larger reference times keep the setup time constant and extend the per-trial rate observed in the largest measured calibration. Terasweep was measured at every size from 10 through one billion on the same workload. A billion draws reused 4 to 18 distinct fits and 12,036 to 36,108 cached scoring entries, depending on the model. The receipt records both counts for every case.

## Reproduce

Set up the reference Python dependencies from the [model reference audit](../../docs/research/model-reference-parity.md) and the [TiDE backend](../../docs/backends.md#tide), then run these commands from the repository root on the measured platform.

```sh
"$TERASWEEP_TIDE_PYTHON" -m pip install 'threadpoolctl==3.6.0'
export TERASWEEP_BENCH_RUSTC="$(rustup which --toolchain 1.100.0 rustc)"
mkdir -p runs/sweep-scaling
make build RUSTC="$TERASWEEP_BENCH_RUSTC" BIN=runs/sweep-scaling/terasweep
"$TERASWEEP_BENCH_RUSTC" --edition=2021 -O -C panic=abort -C target-cpu=native \
  -C lto=fat -C codegen-units=1 --crate-type cdylib \
  benchmarks/sweep-scaling/reference.rs -o runs/sweep-scaling/reference.dylib
reference_python=runs/reference-venv/bin/python
"$reference_python" -B benchmarks/sweep-scaling/run.py prepare
"$reference_python" -B benchmarks/sweep-scaling/run.py native --binary runs/sweep-scaling/terasweep
"$reference_python" -B benchmarks/sweep-scaling/run.py reference --binary runs/sweep-scaling/terasweep
"$reference_python" -B benchmarks/sweep-scaling/run.py report --binary runs/sweep-scaling/terasweep \
  > runs/sweep-scaling/comparison.json
```

The stages retain the raw timing logs and summaries one point at a time under `runs/sweep-scaling/`. The [measurement receipt](results-m1-max.json) preserves the recorded timings, calibration rates, fixture hashes, and parity checks.
