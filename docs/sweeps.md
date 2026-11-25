# Run a parameter sweep

Terasweep fits each model entry on the training matrix and ranks it by mean
absolute error (MAE) on the labeled validation matrix. Both matrices use the
same numeric layout as the fit API and must have the same feature count.
Reserve a separate holdout to estimate the selected model's predictive quality.
[Matrix format](agents.md#matrix-format)

```sh
cargo build --release --locked
target/release/terasweep sweep \
  tests/fixtures/concrete/train.bin \
  tests/fixtures/concrete/validation.bin \
  docs/examples/models.txt
```

The command writes a JSON receipt to stdout. `best.trial` is the zero-based
index of the model entry, excluding blank lines and lines that begin with
`#`. `best.predictions` follows validation row order. The earliest entry
wins an exact MAE tie. Invalid settings, backend failures, and nonfinite
scores fail the sweep.

## Model plan

Enter one model per line. Fields are whitespace-delimited and ordered as
shown below. Boolean values are `true` or `false`. Ridge and LightGBM
accept tabular features. DLinear and TiDE require one observed series.

| Model | Fields after the model name |
| --- | --- |
| `ridge` | `alpha fit_intercept solver_eps` |
| `dlinear` | `alpha lookback kernel solver_eps` |
| `lightgbm` | `learning_rate num_trees num_leaves max_bins lambda_l2 min_data_in_leaf seed` |
| `tide` | `lookback hidden_width learning_rate epochs seed` |

Use `make run TRAIN=… VALIDATION=… PLAN=…` to override the example paths.
The Rust entry point is `terasweep::sweep(&models, &train, &validation)`.
The CLI and Rust API use the same fit settings and selection rule.

## Reuse and timing

The receipt reports the number of plan entries and distinct fits. Equivalent
training configurations reuse predictions and MAE. With fixed data, Ridge
reuses training statistics across regularization settings and DLinear
reuses its design for each lookback and kernel combination. LightGBM reuses
the loaded C API and converted labels. TiDE uses one worker per effective
seed, with fresh weights for each distinct configuration. Both backends use
the low 31 bits of the supplied seed.

The sweep timer includes planning model keys, fitting, and evaluating MAE.
It excludes file loading and JSON output. Every distinct configuration still
requires its own fit. Prediction-only runs can continue to use the single-model
CLI or `fit_predict`. [Backend setup](backends.md) · [Model cards](../README.md#model-cards)
