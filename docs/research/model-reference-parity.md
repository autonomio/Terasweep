# Model reference parity

The current audit checks predictions and validation-MAE sweep winners on public
concrete strength, abalone age, and bike demand fixtures. Each comparison uses
identical feature values, targets, and model settings on both sides.
[Fixture provenance](../../tests/fixtures/README.md)

## Fast checks

`make test` runs the Rust and Python checks. These cover ten-decimal prediction
agreement against reference values for Ridge and DLinear, exact sweep
agreement with separately fitted models, stable first-winner ties, duplicate
fit reuse, input validation, and the CLI and MCP contracts. No external model
backend is needed.

## Live audit

The optional audit calls scikit-learn Ridge, the official LightGBM regressor,
and the pinned Google Research TiDE model directly. It solves DLinear's
regularized linear heads with SciPy moving averages and NumPy SVD. The audit
checks all predictions at `atol=rtol=1e-10` and compares their ten-decimal
formatted values. LightGBM and TiDE require bit equality on the tested
environment. Each sweep contains the same model settings as the individual
fits plus a duplicate, and must return the reference's winning index and score.
Its winning predictions must be bit-identical to those from a separate
Terasweep single fit.

From the checkout, install the audit dependencies and configure the
[TiDE backend](../backends.md#tide). On macOS, LightGBM also needs the
[OpenMP runtime](../backends.md#lightgbm).

```sh
python3.12 -m venv runs/reference-venv
runs/reference-venv/bin/python -m pip install 'numpy==2.2.6' 'scipy==1.15.3' \
  'scikit-learn==1.7.2' 'lightgbm==4.6.0' 'threadpoolctl==3.8.0'
cargo build --release --locked
runs/reference-venv/bin/python -B scripts/verify_model_parity.py
```

The full audit checks eight settings and 512 predictions per model. Use
`--quick` for one setting per model, or give one or more model names to run
a subset. The JSON receipt reports maximum error, bit equality, and sweep
results. The [historical research record](README.md) preserves earlier results
on their original inputs.

## Recorded result

The full audit passed on Apple M1 Max for all 32 configurations. Each of the
four nine-entry plans completed eight fits and matched its reference winner.

| Model | Maximum absolute prediction error | Bit-identical predictions |
| --- | ---: | --- |
| Ridge | `9.95e-14` | No |
| DLinear | `1.78e-14` | No |
| LightGBM | `0` | Yes |
| TiDE | `0` | Yes |

## Single-fit performance

Each sample fits a fresh model from the fixture arrays in memory and predicts
all 64 validation rows once. The timer includes model construction, feature
windows where needed, fitting, and prediction. It excludes process startup,
imports, backend library loading, matrix decoding, and output serialization.
Terasweep's fit timer starts after its input and model validation. The public
Python APIs time any validation they perform inside fit and predict. The run
uses single-threaded backends and checks every prediction for parity.

Local medians on Apple M1 Max, macOS 26.6.2, Rust 1.63.0, and Python 3.12.
The dependency versions are pinned above and in the TiDE backend instructions.

| Model | Direct reference | Terasweep | Reference | Speed ratio | Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| Ridge | scikit-learn Cholesky | 0.019 ms | 1.031 ms | 54.53× | 21 |
| DLinear | SciPy moving average and NumPy SVD | 0.039 ms | 0.626 ms | 15.96× | 21 |
| LightGBM | Python `LGBMRegressor.fit` and `Booster.predict` | 1.693 ms | 2.949 ms | 1.74× | 21 |
| TiDE | Google Research `TideModel` in TensorFlow eager mode | 307.561 ms | 321.047 ms | 1.04× | 7 |

The TiDE sample ranges overlap. Both paths use the same upstream model and
training loop. All four comparisons use 192 training rows and 64 validation
rows. [Fixture provenance](../../tests/fixtures/README.md)

```sh
runs/reference-venv/bin/python -B scripts/verify_model_parity.py --single-fit-samples 21 ridge dlinear lightgbm
runs/reference-venv/bin/python -B scripts/verify_model_parity.py --single-fit-samples 7 tide
```

The commands use the first configuration for each model and report all sample
times in seconds. The speed ratio divides the reference median by the
Terasweep median.
