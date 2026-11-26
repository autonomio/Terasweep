# TiDE

## Input

One observed series and a next-value target per row. Terasweep uses Google
Research's MLP encoder-decoder in a one-step configuration. The observed
series is the only input variable, with a constant placeholder feature passed
to the model. [TiDE paper](https://arxiv.org/abs/2304.08424)

## Output

One continuous prediction per validation or test row, in input order. The
reference case uses bike demand with 192 training rows and 64 validation rows,
each with 1 feature column.
[Fixture provenance](../../tests/fixtures/README.md)

## Reference and equivalence

Pinned Google Research TiDE on independently prepared windows. The audit checks eight separately fitted
configurations and all 512 predictions to ten decimal places. Predictions were bit-identical.
[Reproduce](../research/model-reference-parity.md#live-audit)

## Sweep

Runs one TensorFlow worker per effective seed. Each distinct configuration starts with deterministically initialized weights.

The audit's nine-entry ordered plan completes eight distinct fits. Its
winner and validation MAE agree with the reference. The recorded winner's
MAE is 1.8462 thousand rentals. Use a separate holdout for model
quality estimates. [Plan format](../sweeps.md#model-plan) ·
[Historical measurements](../research/README.md)
