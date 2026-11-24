# TiDE

## Input

A single observed return channel and next-return target in percentage points. Uses the pinned Google Research model for one-step forecasting with no covariates. The recorded train and test matrices were **1,992 × 1** and **1,839 × 1**. Settings were `lookback=6`, `hidden_width=8`, `learning_rate=0.001`, `epochs=2`, `seed=20260514`.

## Output

The fit returned 1,839 continuous predictions in test-row order. The first three were `0.096030846`, `0.13600877`, `-0.032604571`.

## Reference and equivalence

The pinned Google Research TiDE in TensorFlow eager mode with independent window preparation. Bit-identical predictions, with zero differences in the `prediction > 0` signal vector.

## Single run

On Apple M1 Max, Terasweep took **274.393 ms** and the reference took **340.700 ms**, a **1.24×** speed ratio for one fit and predict call. Both runs used the same TensorFlow backend. [Comparison and reproduction](../../benchmarks/single-model/README.md) · [Recorded inputs and results](../../benchmarks/single-model/results-m1-max.json)
