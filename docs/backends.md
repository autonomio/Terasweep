# Backend setup

`cargo build --release --locked` builds the Rust library and executable on
macOS and Linux using Rust and Cargo 1.63 or newer. Ridge and DLinear require
only Rust. The repository Python fit adapter uses Python 3.10 or newer.

## LightGBM

On macOS, install the OpenMP runtime with `brew install libomp` if it is
absent. Install LightGBM 4.6.0 and point Terasweep to the official native
library. Python is used here to locate the library.

```sh
python3 -m venv runs/lightgbm-env
runs/lightgbm-env/bin/python -m pip install 'lightgbm==4.6.0'
export TERASWEEP_LIGHTGBM_LIB="$(runs/lightgbm-env/bin/python -c 'from lightgbm.basic import _LIB; print(_LIB._name)')"
```

## TiDE

Use Python 3.12 with the pinned TensorFlow environment and Google model
source. Terasweep checks the source digest before fitting.

```sh
python3.12 -m venv runs/tide-reference
runs/tide-reference/bin/python -m pip install 'tensorflow==2.16.2' \
  'keras==3.15.1' 'numpy==1.26.4' 'tqdm==4.67.1'
curl -fsSL https://raw.githubusercontent.com/google-research/google-research/5b09c22d73a9d35eb6c5d2a99b95677a45053466/tide/models.py \
  -o runs/tide-reference/models.py
export TERASWEEP_TIDE_PYTHON="$PWD/runs/tide-reference/bin/python"
export TERASWEEP_TIDE_SOURCE="$PWD/runs/tide-reference/models.py"
```

## Series inputs

DLinear and TiDE expect one observation per feature row and one next-value
target per row. Each split forms causal windows from its own preceding
observations, padding the start with its first value. See the
[bike demand fixtures](../tests/fixtures/README.md) for an example and the
[sweep plan format](sweeps.md#model-plan) for backend settings.
