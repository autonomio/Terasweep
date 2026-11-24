# DLinear

## Input

A single observed return channel and a one-step return target in percentage points, derived from the same archived market series. Each split is windowed independently with replicated left-edge values. The recorded train and test matrices were **1,992 × 1** and **1,839 × 1**. Settings were `alpha=1.0`, `lookback=3`, `kernel=3`, `solver_eps=1e-14`.

## Output

The fit returned 1,839 continuous predictions in test-row order. The first three were `0.0019493493`, `0.00041738708`, `0.0035996999`.

## Reference and equivalence

Limen’s SVD-based DLinear with the same return windows and fitted intercept. Maximum absolute difference `6.70e-17` at `rtol=atol=1e-10`, with zero differences in the `prediction > 0` signal vector.

## Single run

On Apple M1 Max, Terasweep took **0.226 ms** and the reference took **6.050 ms**, a **26.79×** speed ratio for one fit and predict call. [Comparison and reproduction](../../benchmarks/single-model/README.md) · [Recorded inputs and results](../../benchmarks/single-model/results-m1-max.json)
