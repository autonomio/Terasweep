# Single-model comparison

Each implementation ran one fit and one prediction pass on the same prepared
payload for its model. The recorded machine was an Apple M1 Max. All four
Terasweep runs matched their references at `atol=rtol=1e-10`, with zero
`prediction > 0` signal differences. LightGBM and TiDE matched bit for bit.

| Model | Reference | Alternative | Terasweep |
| --- | --- | --- | --- |
| [Ridge](../../docs/models/ridge.md) | Limen Ridge (scikit-learn Cholesky) · 1.987 ms | SmartCore Cholesky · 0.697 ms · 2.85× | 0.223 ms · **8.91×** |
| [DLinear](../../docs/models/dlinear.md) | Limen DLinear (NumPy SVD) · 6.050 ms | SciPy LAPACK Cholesky · 0.598 ms · 10.11× | 0.226 ms · **26.79×** |
| [LightGBM](../../docs/models/lightgbm.md) | Official LightGBM Python · 49.463 ms | LightGBM + lleaves · 141.755 ms · 0.35× | 39.359 ms · **1.26×** |
| [TiDE](../../docs/models/tide.md) | Google TiDE (TensorFlow eager) · 340.700 ms | Google TiDE (TensorFlow XLA) · 1843.515 ms · different output | 274.393 ms · **1.24×** |

Ratios are reference time divided by implementation time. The XLA TiDE fit had
a maximum absolute prediction difference of `2.98e-7` and zero sign differences.
Its timing has no parity-qualified ratio. SmartCore and SciPy use Cholesky on
the same Ridge and DLinear objectives. The lleaves run fits with official
LightGBM and includes exporting and compiling that model before prediction.
[SmartCore](https://docs.rs/smartcore/0.6.14/smartcore/linear/ridge_regression/) ·
[SciPy Cholesky](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_factor.html) ·
[lleaves](https://github.com/siboehm/lleaves) ·
[TensorFlow XLA](https://www.tensorflow.org/xla)

The timer starts with prepared matrices in memory and includes required
feature windows, fit, prediction, and first compilation. It excludes dataset
acquisition, input decoding, process startup, output serialization, and
backtest scoring. The single observation per implementation has no uncertainty
estimate. See the [measurement receipt](results-m1-max.json) for settings,
input and output digests, package versions, and numerical differences.

## Reproduce

Set up the TensorFlow environment and pinned model source from the
[backend guide](../../docs/backends.md). Build Terasweep and the standalone
SmartCore comparison program with the same Rust compiler. The comparison
program is built under `runs/`.

```sh
rustup toolchain install 1.100.0 --profile minimal
export TERASWEEP_BENCH_RUSTC="$(rustup which --toolchain 1.100.0 rustc)"
make -B build RUSTC="$TERASWEEP_BENCH_RUSTC"
sh benchmarks/single-model/build-smartcore.sh
```

```sh
python3.12 -m venv runs/single-model-reference
runs/single-model-reference/bin/python -m pip install 'vaquum-limen[data,stats]==6.0.1' \
  'numpy==2.2.6' 'scipy==1.15.3' 'scikit-learn==1.7.2' 'polars==1.42.1' \
  'pyarrow==23.0.1' 'lightgbm==4.6.0' 'lleaves==1.3.0' 'llvmlite==0.44.0'
runs/single-model-reference/bin/python -B benchmarks/single-model/compare.py \
  > runs/single-model-comparison.json
```

Pass one model name to the comparison script to run only that comparison. The
same script compares all prediction values before it reports a speed ratio.
