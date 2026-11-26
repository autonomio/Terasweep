# LightGBM

## Input

Prepared tabular features and a continuous regression target. The native adapter uses the official LightGBM shared library with single-threaded deterministic regression.

## Output

One continuous prediction per validation or test row, in input order. The
reference case uses abalone age with 192 training rows and 64 validation rows,
each with 10 feature columns.
[Fixture provenance](../../tests/fixtures/README.md)

## Reference and equivalence

Official Python LightGBM 4.6.0 regression. The audit checks eight separately fitted
configurations and all 512 predictions to ten decimal places. Predictions were bit-identical.
[Reproduce](../research/model-reference-parity.md#live-audit) · [Single-fit performance](../research/model-reference-parity.md#single-fit-performance)

## Sweep

Reuses the loaded backend and converted labels. Each distinct model configuration trains a fresh booster.

The audit's nine-entry ordered plan completes eight distinct fits. Its
winner and validation MAE agree with the reference. The recorded winner's
MAE is 1.9730 rings. Use a separate holdout for model
quality estimates. [Plan format](../sweeps.md#model-plan) ·
[Historical measurements](../research/README.md)
