# Model reference parity

Run the prediction audit before comparing performance against an upstream model.
The audit compares every test prediction with `rtol=1e-10` and `atol=1e-10`,
and requires exact agreement on the full `prediction > 0` signal vector.
LightGBM additionally requires bit-for-bit equality. The audit exits nonzero
if a requested reference comparison fails or is blocked.

```sh
python3.12 -m venv runs/reference-venv
runs/reference-venv/bin/python -m pip install 'vaquum-limen[data,stats]==6.0.1' 'numpy==2.2.6' \
  'scipy==1.15.3' 'scikit-learn==1.7.2' 'polars==1.42.1' 'pyarrow==23.0.1' 'lightgbm==4.6.0'
make build
runs/reference-venv/bin/python scripts/verify_model_parity.py
```

Pass model names to run a subset, for example
`runs/reference-venv/bin/python scripts/verify_model_parity.py ridge dlinear`. The command
prints per-fit counts and errors as JSON. It uses retained real-market inputs
and creates temporary files for runs and prediction exports.

| Model | Result on Apple M1 Max | Comparison |
| --- | --- | --- |
| Ridge | Pass | Limen's sklearn-backed Ridge on the archived 1,014-row ordered plan. Six fits, 8,586 predictions, maximum absolute difference `3.86e-14`, zero signal differences. |
| DLinear | Pass on aligned windows | Limen's SVD-based DLinear on identical single-channel return windows, with a fitted intercept. Twelve fits, 22,068 predictions, maximum absolute difference `1.70e-17`, zero signal differences. |
| LightGBM | Pass | Native C API and Python reference use official LightGBM 4.6.0 on identical frozen arrays. All 9,088 predictions across eight fits are bit-identical. |
| TiDE-style regressor | Blocked | The Rust smoke case fits and scores. The upstream model has separate residual encoder, decoder and temporal decoder blocks and a covariate input contract. The Rust model has one hidden layer and a linear skip, so there is no corresponding upstream fit to compare. |

The Ridge check reuses the archived plan and prepared matrices and refits the
current Limen reference. DLinear derives the return channel from the archived
parquet and builds lag windows within each split with the same edge padding on
both sides. This covers the aligned univariate fit. The default 55-channel
Terasweep workload and intercept-free cases use a different input or fit
contract. LightGBM uses the official regression estimator because the frozen
target is a continuous next return. Limen's built-in LightGBM is a binary
classifier. The original custom GBDT differed on all eight fits, changed 1,683
signals, and had a maximum absolute prediction difference of `0.1047`. [Ridge experiment](../benchmarks/reports/terasweep_reverse_funnel_report.md#5-correctness-and-repeatability) · [Limen model contracts](https://docs.vaquum.fi/limen/reference/reference-architecture/) · [Official LightGBM regression](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html) · [Upstream TiDE model](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/tide/models.py)

The tolerance applies to the recorded inputs and parameter cases. Backtest
accounting and performance require separate matched comparisons.

The benchmark evidence verifier checks the measured source against its
preserved archive. Use `scripts/verify_evidence.py --check-current-source`
with Python to also require the current checkout to match that source.
