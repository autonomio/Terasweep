# Terasweep

Factorized Rust parameter sweeps with reference-checked model predictions.

[Key features](#key-features) · [Install](#install) · [Examples](#examples) · [Model cards](#model-cards) · [Research and learning](#research-and-learning) · [Support](#support) · [Cite](#citations) · [Download](https://github.com/autonomio/Terasweep/archive/refs/heads/main.zip)

Fit a Ridge, DLinear, LightGBM regression, or TiDE model and compare its
predictions with an upstream numerical reference. Sweep model and scoring
settings on prepared data with the same runner.

## Key features

- Deterministic sampled sweeps reuse fits and base scores across trials that share their inputs.
- Scalar and compact scoring, with Rust AVX-512 intrinsics and handwritten assembly on supported x86-64 Linux hosts. [Execution levels](docs/research/optimization-pipeline.md)
- The single-model CLI and agent interface take the same settings and matrix files and return the same JSON receipt. [Agent interface](docs/agents.md)
- Reproducible model cards and sweep measurements include the inputs, reference checks, and timing boundaries. [Single-run comparison](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/README.md) · [Sweep scaling](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/README.md)

## Install

Cargo builds the `terasweep` library and CLI. Install the CLI from a checkout
with `cargo install --path . --locked`, or build both with
`cargo build --release --locked`. The library exposes `fit_predict` and
`sweep` for prepared matrices.

Use Make and Python 3.10 or newer for the repository examples and correctness
suite.

```sh
make build
make test
```

`make test` checks predictions to ten decimal places on the native reference
cases, bit equality between scoring paths, the seeded trial order, and the
CLI and MCP contracts. The quick live reference check covers all four models
using the optional backends. [Correctness checks](docs/research/model-reference-parity.md#fast-checks)

Ridge and DLinear run with Rust alone. LightGBM loads the official shared
library when selected. TiDE uses the pinned Google Research model with a local
TensorFlow environment. [Backend setup](docs/backends.md)

## Examples

### Run a model

Run an example using the bundled Ridge matrices. The receipt includes the model
settings, input dimensions and SHA-256 hashes, full prediction vector, and fit
and predict time.

```sh
python3 scripts/model_run.py fit docs/examples/ridge.json
```

Use `python3 scripts/model_run.py schema` to inspect valid model parameters and
the receipt schema. The same request can be sent to the `terasweep_fit` tool
through the local stdio MCP server. [Agent interface](docs/agents.md)

### Run a sweep

Sample 1,000 Ridge trials on the bundled data. The summary records trial and
failure counts, cache coverage, and the winning return, Sharpe, and MAE trials.

```sh
make run LEVEL=compact RUNS=1000 OUT=runs/ridge-example
```

## Model cards

| Model | Input | Reference |
| --- | --- | --- |
| [Ridge](docs/models/ridge.md) | Prepared features and a continuous target | Limen Ridge with scikit-learn Cholesky |
| [DLinear](docs/models/dlinear.md) | One observed return channel and a next-return target | Limen DLinear with NumPy SVD |
| [LightGBM](docs/models/lightgbm.md) | Prepared features and a continuous target | Official Python LightGBM regression |
| [TiDE](docs/models/tide.md) | One observed return channel and a next-return target | Google Research TiDE |

Each card gives the measured input, output, parity result, single-run speed
ratio, and sweep scaling through one billion sampled trials.
[Single-run comparison](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/README.md) · [Sweep scaling](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/README.md)

## Research and learning

The benchmark record preserves the original Ridge experiments, including the
optimization stages and large-scale AVX-512 measurements. The research notes
connect those experiments to the current reference checks and matched sweeps
for all four models. [Benchmark record](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/README.md) · [Research notes](docs/research/README.md)

The interactive performance course covers how to factor a sweep, validate its
outputs, and interpret the measured and projected results.
[Open the course](https://terasweep.autonom.io/) · [Offline copy and sources](docs/masterclass/README.md)

## Support

Include the Git commit, model and backend versions, exact command, input shapes,
and expected result when filing a reproducible report. [GitHub issues](https://github.com/autonomio/Terasweep/issues)

## Citations

For published work, cite the software and record the full commit used. GitHub's
**Cite this repository** menu uses the citation metadata. [CITATION.cff](CITATION.cff) · [BibTeX](CITATION.bib)

For a benchmark result, also cite its measurement receipt or report and retain
the hardware, backend versions, input provenance, seed, and parameter grid.
[Research and reproducibility](docs/research/README.md#citing-an-experiment)
