# Native Limen versus Terasweep — measured results

Run date: 22 September 2026. All measurements below came from execution on the connected Mac, not estimates or extrapolations.

## Scope

Two unchanged implementations were executed:

- Native Limen 5.11.0 at merged commit `44eff3b244c80ce8473845bd60b074b3682fe441`, through its actual `python -m limen run` CLI, YAML compiler, manifest preparation, UEL, Ridge SFD, inline metrics, and artifact writing.
- The existing `~/dev/Terasweep/target/terasweep` executable. No rebuild or source change was made.

The matched workload was the newly shipped Ridge SFD's six cases: alpha 0.1, 1.0, and 10.0, each with and without an intercept. This is NOT a one-million-round native-Limen-versus-Terasweep benchmark. The original Terasweep 1M workload was rerun separately and was not used to calculate a cross-engine speedup.

The original sweep contains solver-pivot tolerances and signal-rule/threshold variations that the native Ridge SFD does not implement. No substitute SFD, custom solver, custom backtest, or cache was introduced to simulate that original sweep inside Limen.

## Inputs and matching

The shipped native YAML template was changed only to use a frozen local source, deterministic native grid enumeration, and output paths. Model parameters, feature preparation, target, scaling, and native evaluation remained those of the shipped template.

- Raw data: 8,760 BTCUSDT hourly observations, 1 January–31 December 2025.
- Source: the already-cached `btcusdt_1h_kline_20200101_to_20260921.parquet` snapshot, read through native `HistoricalData.get_any_file`; the 2025 observations were retained as `raw.parquet`.
- Native prepared training shape: 6,521 × 22.
- Native prepared validation shape: 1,439 × 22.
- Native prepared test shape: 719 × 22; 719 matching open/close price pairs.
- Native training resolves `solver='auto'` to Cholesky in all six cases.
- Native signal: prediction > 0; native default costs: 5 bps fee and 5 bps slippage per fill.
- The exact native-prepared training/test arrays and prices were serialized into Terasweep's existing binary format. Exported matrices were read back and checked for exact equality.
- Terasweep configuration: the same three alphas and two intercept settings; `gt`, threshold 0, fee 5, slippage 5, `solver_eps=1e-12`. Native `tol` was not equated to Terasweep's pivot cutoff.
- Terasweep seed 20260698 makes the first six sampled rows cover all six combinations once. Its measured summary confirms six training configurations, six score-cache entries, six valid rows, and no failures.

Native validation data is retained by its pipeline but is not used to fit Ridge. Terasweep consumes only the training/test matrices and prices, as its normal interface specifies.

## Timings

Five separate-process repeats, with alternating engine order. No runs were discarded.

| Repeat | Native CLI wall time, s | Terasweep wall time, s | Sum of native logged round times, s | Terasweep internal timer, s |
|---|---:|---:|---:|---:|
| 1 | 2.84 | 0.02 | 0.32 | 0.003459208 |
| 2 | 2.34 | 0.01 | 0.29 | 0.003115333 |
| 3 | 2.46 | 0.01 | 0.35 | 0.003071750 |
| 4 | 2.33 | 0.01 | 0.37 | 0.003106458 |
| 5 | 2.40 | 0.01 | 0.24 | 0.003545583 |
| Median | 2.40 | 0.01 | 0.32 | 0.003115333 |

Median peak resident memory: native Limen 255.375 MiB; Terasweep 4.40625 MiB.

Wall timing used `/usr/bin/time -lp`, whose saved output resolves seconds to two decimals. The 0.01-second Terasweep readings do not justify a precise wall-time speedup factor.

These are different pipeline boundaries. Native CLI wall time includes Python startup/imports, YAML validation, raw loading, repeated manifest preparation, fitting, native metrics, and complete experiment artifacts. Terasweep reads already-prepared arrays, reuses training sufficient statistics, and writes its summary. Its internal timer includes preparation of ridge statistics and fits/predictions, scoring, and winner details, but excludes input loading. The native logged round timers include preparation and native metric computation and are also rounded. Neither timer isolates the ridge solver. No pure ridge-fit speedup is claimed.

This six-case run has no repeated score-cache hits: six rows produce six unique score keys. It therefore does not establish large-sweep scaling or steady-state cache throughput.

The machine was busy: the one-minute load average at the beginning of the measured sequence was 27.65. Other users' processes were not stopped. These are observed timings under contention, not an isolated-machine performance ceiling.

## Output agreement

All six cases have reported MAE 0.290 in both implementations, and all six agree on reported positive-signal rate.

| Alpha | Intercept | Positive predictions, both | Signal rate, both, % | Native compounded return rounded to 0.1%, % | Terasweep reported return, % |
|---|---|---:|---:|---:|---:|
| 0.1 | Yes | 310 | 43.115 | -26.5 | -26.5 |
| 1.0 | Yes | 313 | 43.533 | -24.5 | -24.4 |
| 10.0 | Yes | 335 | 46.592 | -22.3 | -22.3 |
| 0.1 | No | 314 | 43.672 | -23.4 | -23.4 |
| 1.0 | No | 318 | 44.228 | -24.4 | -24.4 |
| 10.0 | No | 335 | 46.592 | -22.3 | -22.3 |

Each Terasweep case was additionally executed as a singleton configuration for validation, outside the five timed comparisons. Terasweep signal counts in this table are recovered from its reported signal rate and the known 719-row denominator. They are not its differently defined `edge_per_signal_n` statistic.

Native compounded return is calculated from the per-bar net returns produced by Limen's existing `long_flat_strategy`, using the retained native predictions and the same prices and costs. It is a derived diagnostic, not a total-return field in native `results.csv`. Raw native values are retained in `comparison.json`.

The stock backtest implementations are not equivalent: Terasweep uses the entry bar's open-to-close return and its cached cost-factor formula; current Limen uses prior-close-to-close returns and its current fees-on-notional accounting. Their metric inventories and aggregation populations also differ. Five matching rounded total returns do not establish exact economic parity; one already differs at the shared display precision.

MAE/signal-rate agreement is aggregate, rounded-output agreement only. The unchanged Terasweep executable does not export the complete coefficient or prediction arrays, so full coefficient, prediction, or bar-by-bar signal parity was not established.

## Repeatability and native reconstruction

- All five native result tables are identical after excluding `execution_time`.
- All five native `round_data.jsonl` files, including predictions, are byte-identical.
- All five Terasweep summaries are identical after excluding the two timing fields.
- All six native models reconstruct through native `Trainer`, loading the frozen input from the retained YAML reference without a data-source monkeypatch.
- All six reconstructed Sensors return finite predictions. The stock sklearn feature-name warning is emitted during Sensor inference; it was not suppressed or patched.
- A further run using the saved replay command in a fresh output directory also reproduced native stable metrics, native prediction artifacts, and Terasweep stable output exactly. This replay is not included in the five-run timing table.

## Separate original Terasweep 1M regression run

Original, unchanged inputs and `configs/sweep.json`:

| Measurement | Result |
|---|---:|
| Sampled rows | 1,000,000 |
| Internal timer | 4.483524333 s |
| Process wall time | 4.50 s |
| Internal sampled-row throughput | 223,038.825203 rows/s |
| Valid rows | 728,102 |
| Failed rows | 271,898 |
| Unique training configurations | 2,304 |
| Failed training configurations | 627 |
| Golden comparison | Passed |

The failures match the saved canonical result. Sampled-row throughput includes failed rows and is not a count of independently fitted models per second. Native Limen was not run for this 1M workload, so there is no measured 1M cross-engine ratio.

## Environment

Apple M1 Max, arm64, 64 GiB RAM, macOS 26.6.2. Python 3.12.12; Limen 5.11.0; NumPy 2.5.1; SciPy 1.18.0; scikit-learn 1.9.0; Polars 1.42.1; pandas 2.3.3; PyArrow 22.0.0. These are the installed versions observed on the connected machine.

Both timed paths received `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`. Polars' thread setting was not overridden.

## Retained evidence and replay

Everything needed for the executed comparison is retained on the connected Mac under:

`/Users/mikkokotila/dev/_reports/ridge-side-by-side-20260922/`

Key files: `provenance.json`, `timings.json`, `comparison.json`, `repeatability.json`, `input-sha256.txt`, `native-repeatable.yaml`, `terasweep.json`, `raw.parquet`, `train.bin`, `test.bin`, `price.bin`, and `replay.sh`. Original measurements and experiment artifacts are under `native/`, `terasweep/`, `logs/`, `validation/`, and `original-1m/`.

Replay with an unused absolute output directory:

```sh
~/dev/_reports/ridge-side-by-side-20260922/replay.sh \
  "$HOME/dev/_reports/ridge-replay-$(date +%Y%m%d-%H%M%S)"
```

The script checks input/executable hashes before running. It defaults to the exact Python environment used here; `LIMEN_PYTHON` can select another interpreter, whose package versions must be matched to the recorded environment for a comparable replay. It refuses an already-existing output directory rather than deleting previous results.

Input SHA-256:

- `raw.parquet`: `405c7f9300b0e6f2e10fb11290e16588cf2eb615a994b46d309080680c67bfe5`
- `train.bin`: `18792e6e92b30e9f9fbd9c2c44b21332402f886b4fddecf5846672880c5d73a4`
- `test.bin`: `2beadb02a9fe98d188a28851043653c223f5ee1f68610b548001744992cc1d01`
- `price.bin`: `c099937f7723b571e21fa7931c033fb9a3a17a153c23b64db4c9e262f0003e73`
