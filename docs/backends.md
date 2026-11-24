# Backend setup

`cargo build --release --locked` builds the Rust library and executable.
`make build` also provides the executable path used by the repository's Python
CLI and MCP adapter. Ridge and DLinear have no external model backend. Use
Rust and Cargo 1.63 or newer on ARM64 macOS. The x86-64 Linux build requires
Rust and Cargo 1.89 or newer for its AVX-512 kernels.

## LightGBM

On macOS, install the OpenMP runtime with `brew install libomp` if it is not
already available. The LightGBM wheel loads `libomp.dylib` from the system.

Install LightGBM 4.6.0 in a local Python environment and point the executable at
its native library. Python is used here to locate the library.

```sh
python3 -m venv runs/lightgbm-env
runs/lightgbm-env/bin/python -m pip install 'lightgbm==4.6.0'
export TERASWEEP_LIGHTGBM_LIB="$(runs/lightgbm-env/bin/python -c 'from lightgbm.basic import _LIB; print(_LIB._name)')"
```

## TiDE

Use Python 3.12 with the pinned TensorFlow environment and the Google model
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

## Return inputs

DLinear and TiDE take a single observed return column and one next-return
target per row, both in percentage points. Use the bundled return matrices for
an example sweep.

```sh
make run CONFIG=configs/dlinear.json TRAIN=tests/fixtures/returns/train.bin \
  TEST=tests/fixtures/returns/test.bin PRICE=tests/fixtures/returns/price.bin \
  RUNS=1000 OUT=runs/dlinear-example
```

Use `CONFIG=configs/tide.json` with the same matrices after setting up TiDE.
[Fixture provenance](../tests/fixtures/README.md)
