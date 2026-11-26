# DLinear

## Input

One observed series and a next-value target per row. Each split is windowed independently with left-edge replication. The model decomposes each window into moving-average and residual components and fits an intercept.

## Output

One continuous prediction per validation or test row, in input order. The
reference case uses bike demand with 192 training rows and 64 validation rows,
each with 1 feature column.
[Fixture provenance](../../tests/fixtures/README.md)

## Reference and equivalence

SciPy moving-average decomposition and NumPy SVD solve the same regularized
two-head regression objective. The audit checks eight separately fitted
configurations and all 512 predictions to ten decimal places. Maximum absolute prediction difference was `1.78e-14`.
[Reproduce](../research/model-reference-parity.md#live-audit) · [Single-fit performance](../research/model-reference-parity.md#single-fit-performance)

## Sweep

Reuses its training statistics and validation design when lookback and moving-average kernel match.

The audit's nine-entry ordered plan completes eight distinct fits. Its
winner and validation MAE agree with the reference. The recorded winner's
MAE is 0.7475 thousand rentals. Use a separate holdout for model
quality estimates. [Plan format](../sweeps.md#model-plan) ·
[Historical measurements](../research/README.md)
