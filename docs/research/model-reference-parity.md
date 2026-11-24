# Model reference parity

Terasweep applies the Ridge experiment's validation method to all four models.
Each comparison holds the prepared inputs and fit settings fixed, checks every
prediction at `rtol=1e-10` and `atol=1e-10`, and requires exact agreement on the
`prediction > 0` signal vector. LightGBM and TiDE also require bitwise equality
on the tested host. For each model the audit checks identical scalar and compact
summaries and prediction exports on the same ordered trials.

## Reproduce

```sh
python3.12 -m venv runs/reference-venv
runs/reference-venv/bin/python -m pip install 'vaquum-limen[data,stats]==6.0.1' \
  'numpy==2.2.6' 'scipy==1.15.3' 'scikit-learn==1.7.2' 'polars==1.42.1' \
  'pyarrow==23.0.1' 'lightgbm==4.6.0'

python3.12 -m venv runs/tide-reference
runs/tide-reference/bin/python -m pip install 'tensorflow==2.16.2' \
  'keras==3.15.1' 'numpy==1.26.4' 'tqdm==4.67.1'
curl -fsSL https://raw.githubusercontent.com/google-research/google-research/5b09c22d73a9d35eb6c5d2a99b95677a45053466/tide/models.py \
  -o runs/tide-reference/models.py
export TERASWEEP_TIDE_PYTHON="$PWD/runs/tide-reference/bin/python"
export TERASWEEP_TIDE_SOURCE="$PWD/runs/tide-reference/models.py"

make build
TERASWEEP_CHECK_BASE=1 runs/reference-venv/bin/python scripts/verify_model_parity.py
```

The audit runs the full bundled training grids for DLinear, LightGBM and TiDE.
It prints per-fit counts and errors as JSON and exits nonzero on a mismatch. Pass model names such as `ridge dlinear` to run a subset. The CLI
loads LightGBM only for `lightgbm` and starts TensorFlow only for `tide`.
The TiDE adapter verifies the upstream source digest before importing it.

## Results on Apple M1 Max

| Model | Fits | Predictions | Maximum absolute error | Signal differences |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 6 | 8,586 | `3.86e-14` | 0 |
| DLinear | 18 | 33,102 | `1.03e-16` | 0 |
| LightGBM | 1,440 | 1,635,840 | `0` | 0 |
| TiDE | 16 | 29,424 | `0` | 0 |

Ridge replays the archived 1,014-row ordered plan and refits the current Limen
Ridge reference. DLinear refits Limen's SVD reference with the same single-channel
return windows and fitted intercept. LightGBM compares the official C API with
the Python regression estimator on the same frozen arrays. The original custom
GBDT changed 1,683 signals in the initial eight-fit comparison. TiDE uses the pinned Google Research
model in a single-series, one-step configuration with no covariates and compares
with an independent process supplied with the same pre-windowed observations.
The three new comparisons also verify 1,000 sampled rows per model across both
scalar and compact scoring. [Ridge experiment](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/reports/terasweep_reverse_funnel_report.md#5-correctness-and-repeatability) · [Limen model contracts](https://docs.vaquum.fi/limen/reference/reference-architecture/) · [Official LightGBM regression](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html) · [Upstream TiDE model](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/tide/models.py)

All models share `train`, `score`, `bounds`, `compact`, `intrinsics` and
`assembly` scoring levels. The `naive` and `stats` training controls currently
require Ridge. The model checks above cover the recorded inputs and fit settings.
The [matched sweep benchmark](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/README.md) compares
timings and winners on Terasweep's scorer. Native Limen uses its own backtest
accounting, which is outside that comparison.

The benchmark evidence verifier lives with the archived experiments. The
regression fixtures used here retain their exact bytes independently.
[Research record](README.md) · [Fixture provenance](../../tests/fixtures/README.md)
