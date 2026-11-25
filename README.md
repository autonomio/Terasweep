# Terasweep

Factorized Rust parameter sweeps with reference-checked model predictions.

[Key features](#key-features) · [Install](#install) · [Examples](#examples) · [Model cards](#model-cards) · [Research and learning](#research-and-learning) · [Support](#support) · [Cite](#citations) · [Download](https://github.com/autonomio/Terasweep/archive/refs/heads/main.zip)

Fit Ridge, DLinear, LightGBM regression, or TiDE on prepared numeric matrices.
Compare model settings on a labeled validation split using mean absolute error.

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

The optional Qwen3 LoRA trainer hook has a separate CUDA research record.
[Pannu0 experiment](docs/models/qwen3-lora-pannu0.md)

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
