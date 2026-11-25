# Ridge

## Input

Prepared tabular features with one continuous target per training row. The fit uses Cholesky decomposition with optional intercept fitting and a configurable pivot cutoff.

## Output

One continuous prediction per validation or test row, in input order. The
reference case uses concrete strength with 192 training rows and 64 validation rows,
each with 8 feature columns.
[Fixture provenance](../../tests/fixtures/README.md)

## Reference and equivalence

Limen Ridge with scikit-learn Cholesky. The audit checks eight separately fitted
configurations and all 512 predictions to ten decimal places. Maximum absolute prediction difference was `9.95e-14`.
[Reproduce](../research/model-reference-parity.md#live-audit)

## Sweep

Computes raw and centered training statistics once for the fixed training split. A change to alpha still requires solving for new coefficients.

The audit's nine-entry ordered plan completes eight distinct fits. Its
winner and validation MAE agree with the reference. The recorded winner's
MAE is 8.1439 MPa. Use a separate holdout for model
quality estimates. [Plan format](../sweeps.md#model-plan) ·
[Historical measurements](../research/README.md)
