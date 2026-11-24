# TiDE

## Input

A single observed return channel and next-return target in percentage points. Uses the pinned Google Research model for one-step forecasting with no covariates. The recorded train and test matrices were **1,992 × 1** and **1,839 × 1**. Settings were `lookback=6`, `hidden_width=8`, `learning_rate=0.001`, `epochs=2`, `seed=20260514`.

## Output

The fit returned 1,839 continuous predictions in test-row order. The first three were `0.096030846`, `0.13600877`, `-0.032604571`.

## Reference and equivalence

The pinned Google Research TiDE in TensorFlow eager mode with independent window preparation. Bit-identical predictions, with zero differences in the `prediction > 0` signal vector.

## Single run

On Apple M1 Max, Terasweep took **274.393 ms** and the reference took **340.700 ms**, a **1.24×** speed ratio for one fit and predict call. Both runs used the same TensorFlow backend. [Comparison and reproduction](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/README.md) · [Recorded inputs and results](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/results-m1-max.json)

## Sweep scaling

On M1 Max, the sweep used **256 × 1** training and **128 × 1** test matrices and 4 distinct fit settings. The uncached Google Research TiDE reference refit each trial and used the same scalar scorer on the same trial order.

| Sampled trials | Terasweep measured | Reference time | Reference basis | Ratio |
| ---: | ---: | ---: | --- | ---: |
| 10 | 6.064 s | 4.472 s | Measured | 0.74× |
| 100 | 3.650 s | 33.229 s | Measured | 9.10× |
| 1,000 | 3.413 s | 5.83 min | Measured | 103× |
| 10,000 | 3.395 s | 57.92 min | Projected | 1,024× |
| 100,000 | 3.449 s | 9.65 h | Projected | 10,068× |
| 1,000,000 | 3.508 s | 4.02 days | Projected | 98,983× |
| 10,000,000 | 4.038 s | 40.19 days | Projected | 859,835× |
| 100,000,000 | 11.056 s | 401.88 days | Projected | 3,140,695× |
| 1,000,000,000 | 75.606 s | 4,018.83 days | Projected | 4,592,608× |

Reference measurements through 1,000 trials passed prediction, signal, and winner parity checks. Larger reference entries extend that calibration linearly. Terasweep completed each listed size. [Method and reproduction](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/README.md) · [Measurement receipt](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/results-m1-max.json)
