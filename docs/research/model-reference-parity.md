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

The optional audit compares predictions with Limen Ridge and DLinear, the
official LightGBM regressor, and the pinned Google Research TiDE model. It
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
runs/reference-venv/bin/python -m pip install 'vaquum-limen[data,stats]==6.0.1' \
  'numpy==2.2.6' 'scipy==1.15.3' 'scikit-learn==1.7.2' 'polars==1.42.1' \
  'pyarrow==23.0.1' 'lightgbm==4.6.0'
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
