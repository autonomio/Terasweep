# Terasweep

Terasweep is a Rust package that tests how much faster established models can
fit and predict on the same inputs while matching their reference outputs.
It then uses classic precomputation and memoization to reduce repeated work
across parameter trials.

[Measured performance](#measured-performance) · [Key features](#key-features) · [Install](#install) · [Examples](#examples) · [Model cards](#model-cards) · [Research and learning](#research-and-learning) · [Support](#support) · [Cite](#citations) · [Download](https://github.com/autonomio/Terasweep/archive/refs/heads/main.zip)

Fit Ridge, DLinear, LightGBM regression, or TiDE on prepared numeric matrices.
Compare model settings on a labeled validation split using mean absolute error.

## Measured performance

### Single fit and predict

On Apple M1 Max, a single Ridge fit and prediction pass took **0.223 ms** in
Terasweep versus **1.987 ms** with scikit-learn's Cholesky Ridge through Limen.
That's **8.91×** on the same prepared matrices, with a maximum absolute
prediction difference of `5.77e-15`. The same benchmark recorded **26.79×** for
DLinear, **1.26×** for LightGBM, and **1.24×** for TiDE against their respective
references. Each timing is one observation of the combined fit and prediction
path. [Single-fit method and results](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/README.md)

### Parameter sweeps

A sweep compounds the gains. Faster fitting and prediction lower the cost of
each distinct model configuration. Sharing training statistics across settings
and reusing predictions and scores across trials reduces how often that work
runs.

The archived Apple M1 Max pilot matched the reference predictions, signals,
and sweep winners across linear models, boosted trees, and TiDE's MLP
encoder-decoder. The uncached path used each model's established reference,
including scikit-learn's Cholesky Ridge through Limen. It refit every trial
and applied the same scoring rules. The reference times below exclude backend
setup. [Benchmark method](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/README.md)

| Model | Matched trials | Uncached loop | Terasweep | Measured speedup | Terasweep at 1 billion trials |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 10,000 | 2.223 s | 2.524 ms | **881×** | 51.38 s |
| DLinear | 10,000 | 2.054 s | 1.755 ms | **1,170×** | 72.46 s |
| LightGBM | 10,000 | 31.547 s | 22.924 ms | **1,376×** | 77.48 s |
| TiDE | 1,000 | 347.227 s | 3.413 s | **102×** | 75.61 s |

Each billion-trial run reused 4 to 18 distinct fits and cached scoring work
across sampled trials. TiDE had the largest projected speedup at that scale.
The uncached estimate extrapolates its measured 1,000-trial loop. The current
package applies fit and training-statistic reuse to generic validation-MAE
sweeps. [Measurement receipt and projections](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/results-m1-max.json) · [Current reference audit](docs/research/model-reference-parity.md)

A separate Ridge ablation in the same Rust runner measured a **115×** speedup
from sharing training statistics, fitted predictions, and scores.
[Factorization ablation](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/reports/terasweep_reverse_funnel_report.md#outcome)

## Key features

- Ordered sweeps fit and score each distinct training configuration once, with deterministic tie handling.
- Ridge shares training statistics across model settings. DLinear shares window designs, LightGBM shares backend setup, and TiDE batches fits by seed.
- The single-model CLI and local agent interface use the same Rust fit and receipt. [Agent interface](docs/agents.md)
- The correctness suite checks native predictions to ten decimal places. An optional audit compares all four models with upstream and checks their sweep winners. [Reference audit](docs/research/model-reference-parity.md)

## Install

Cargo builds the `terasweep` library and CLI. Install the CLI from a checkout
with `cargo install --path . --locked`, or build both with
`cargo build --release --locked`. The library exposes `fit_predict` and `sweep`.
Use Python 3.10 or newer for the repository fit adapter and Make for the example
and correctness commands.

```sh
make build
make test
```

Ridge and DLinear run with Rust alone. LightGBM loads the official shared
library when selected. TiDE uses the pinned Google Research model with a local
TensorFlow environment. [Backend setup](docs/backends.md)

## Examples

### Run a model

The bundled example fits Ridge on concrete measurements. The receipt includes
model settings, matrix dimensions and SHA-256 hashes, predictions, and fit time.

```sh
python3 scripts/model_run.py fit docs/examples/ridge.json
```

Use `python3 scripts/model_run.py schema` to inspect the JSON contract. The
local stdio MCP server accepts the same request through `terasweep_fit`.
[Agent interface](docs/agents.md)

### Run a sweep

```sh
make run
```

The example evaluates four entries with three distinct fits. The JSON result
includes trial counts, elapsed time, and the best trial's validation MAE and
predictions. Supply your own matrices and ordered model settings to the CLI or
Rust library. [Sweep interface and plan format](docs/sweeps.md)

## Model cards

| Model | Input | Reference |
| --- | --- | --- |
| [Ridge](docs/models/ridge.md) | Prepared features and a continuous target | Limen Ridge with scikit-learn Cholesky |
| [DLinear](docs/models/dlinear.md) | One observed series and a next-value target | Limen DLinear with NumPy SVD |
| [LightGBM](docs/models/lightgbm.md) | Prepared features and a continuous target | Official Python LightGBM regression |
| [TiDE](docs/models/tide.md) | One observed series and a next-value target | Google Research TiDE |

## Research and learning

The interactive course covers fit reuse, validation, and reproducibility
and works offline from a checkout. [Open the course](https://terasweep.autonom.io/) ·
[Offline copy and sources](docs/masterclass/README.md)

The research record indexes reference checks and archived benchmark evidence.
[Research notes](docs/research/README.md)

## Support

Include the Git commit, model and backend versions, exact command, input shapes,
and expected result when filing a reproducible report. [GitHub issues](https://github.com/autonomio/Terasweep/issues)

## Citations

For published work, cite the software and record the full commit used. GitHub's
**Cite this repository** menu uses the citation metadata.
[CITATION.cff](CITATION.cff) · [BibTeX](CITATION.bib)

For a benchmark result, also cite its measurement receipt or report and retain
the hardware, backend versions, input provenance, and parameter settings.
[Research and reproducibility](docs/research/README.md#citing-an-experiment)
