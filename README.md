# Terasweep

Fit one model and compare its predictions with an upstream numerical reference.
Terasweep supports Ridge, DLinear, LightGBM regression, and TiDE. The CLI and
agent interface take the same model settings and matrix files and return the
same JSON receipt.

[Install](#install) · [Run a model](#run-a-model) · [Model cards](#model-cards) ·
[Agent interface](docs/agents.md) · [Single-run comparison](benchmarks/single-model/README.md)

## Install

From a checkout, use `make`, Rust, and Python 3.10 or newer. Build and test the
local runner with the following commands.

```sh
make build
make test
```

Ridge and DLinear run with Rust alone. LightGBM loads the official shared
library when selected. TiDE uses the pinned Google Research model with a local
TensorFlow environment. [Backend setup](docs/backends.md)

## Run a model

Run an example using the bundled Ridge matrices. The receipt includes the model
settings, input dimensions and SHA-256 hashes, full prediction vector, and fit
and predict time.

```sh
python3 scripts/model_run.py fit docs/examples/ridge.json
```

Use `python3 scripts/model_run.py schema` to inspect valid model parameters and
the receipt schema. The same request can be sent to the `terasweep_fit` tool
through the local stdio MCP server. [Agent interface](docs/agents.md)

## Model cards

| Model | Input | Reference |
| --- | --- | --- |
| [Ridge](docs/models/ridge.md) | Prepared features and a continuous target | Limen Ridge with scikit-learn Cholesky |
| [DLinear](docs/models/dlinear.md) | One observed return channel and a next-return target | Limen DLinear with NumPy SVD |
| [LightGBM](docs/models/lightgbm.md) | Prepared features and a continuous target | Official Python LightGBM regression |
| [TiDE](docs/models/tide.md) | One observed return channel and a next-return target | Google Research TiDE |

Each card gives the exact measured input, output, parity result, and single-run
speed ratio. The comparison includes external alternatives on the same payload.
[Comparison table and reproduction](benchmarks/single-model/README.md)

## Research and teaching

The [benchmark record](benchmarks/README.md) and [research notes](docs/research/)
preserve the existing sweep experiments and reference checks. The interactive
[performance masterclass](docs/masterclass/) is an offline teaching artifact.
