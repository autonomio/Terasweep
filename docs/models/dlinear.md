# DLinear

## Input

A single observed return channel and a one-step return target in percentage points, derived from the same archived market series. Each split is windowed independently with replicated left-edge values. The recorded train and test matrices were **1,992 × 1** and **1,839 × 1**. Settings were `alpha=1.0`, `lookback=3`, `kernel=3`, `solver_eps=1e-14`.

## Output

The fit returned 1,839 continuous predictions in test-row order. The first three were `0.0019493493`, `0.00041738708`, `0.0035996999`.

## Reference and equivalence

Limen’s SVD-based DLinear with the same return windows and fitted intercept. Maximum absolute difference `6.70e-17` at `rtol=atol=1e-10`, with zero differences in the `prediction > 0` signal vector.

## Single run

On Apple M1 Max, Terasweep took **0.226 ms** and the reference took **6.050 ms**, a **26.79×** speed ratio for one fit and predict call. [Comparison and reproduction](../../benchmarks/single-model/README.md) · [Recorded inputs and results](../../benchmarks/single-model/results-m1-max.json)

## Sweep scaling

On M1 Max, the sweep used **256 × 1** training and **128 × 1** test matrices and 18 distinct fit settings. The uncached Limen’s SVD DLinear reference refit each trial and used the same scalar scorer on the same trial order.

| Sampled trials | Terasweep measured | Reference time | Reference basis | Ratio |
| ---: | ---: | ---: | --- | ---: |
| 10 | 0.701 ms | 1.217 s | Measured | 1,737× |
| 100 | 0.526 ms | 1.078 s | Measured | 2,052× |
| 1,000 | 0.651 ms | 1.512 s | Measured | 2,320× |
| 10,000 | 1.755 ms | 3.097 s | Measured | 1,764× |
| 100,000 | 10.593 ms | 21.587 s | Projected | 2,038× |
| 1,000,000 | 92.334 ms | 3.44 min | Projected | 2,236× |
| 10,000,000 | 805.010 ms | 34.26 min | Projected | 2,553× |
| 100,000,000 | 7.408 s | 5.71 h | Projected | 2,773× |
| 1,000,000,000 | 72.456 s | 2.38 days | Projected | 2,835× |

Reference measurements through 10,000 trials passed prediction, signal, and winner parity checks. Larger reference entries extend that calibration linearly. Terasweep completed each listed size. [Method and reproduction](../../benchmarks/sweep-scaling/README.md) · [Measurement receipt](../../benchmarks/sweep-scaling/results-m1-max.json)
