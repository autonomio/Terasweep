# LightGBM

## Input

Bundled 55-column features and a continuous next-return target. Training uses the official LightGBM regression objective. The recorded train and test matrices were **4,136 × 55** and **1,136 × 55**. Settings were `learning_rate=0.1`, `num_trees=32`, `num_leaves=7`, `max_bins=32`, `lambda_l2=0.1`, `min_data_in_leaf=20`, `seed=20260514`.

## Output

The fit returned 1,136 continuous predictions in test-row order. The first three were `-0.032412537`, `0.012996523`, `0.0088985605`.

## Reference and equivalence

Official LightGBM 4.6.0 Python regression using the same settings and seed. Bit-identical predictions, with zero differences in the `prediction > 0` signal vector.

## Single run

On Apple M1 Max, Terasweep took **39.359 ms** and the reference took **49.463 ms**, a **1.26×** speed ratio for one fit and predict call. [Comparison and reproduction](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/README.md) · [Recorded inputs and results](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/results-m1-max.json)

## Sweep scaling

On M1 Max, the sweep used **256 × 55** training and **128 × 55** test matrices and 8 distinct fit settings. The uncached official Python LightGBM reference refit each trial and used the same scalar scorer on the same trial order.

| Sampled trials | Terasweep measured | Reference time | Reference basis | Ratio |
| ---: | ---: | ---: | --- | ---: |
| 10 | 34.475 ms | 850.905 ms | Measured | 24.68× |
| 100 | 21.010 ms | 1.081 s | Measured | 51.45× |
| 1,000 | 22.135 ms | 3.828 s | Measured | 173× |
| 10,000 | 22.924 ms | 32.353 s | Measured | 1,411× |
| 100,000 | 29.709 ms | 5.27 min | Projected | 10,646× |
| 1,000,000 | 254.097 ms | 52.59 min | Projected | 12,418× |
| 10,000,000 | 857.320 ms | 8.76 h | Projected | 36,798× |
| 100,000,000 | 7.690 s | 3.65 days | Projected | 41,024× |
| 1,000,000,000 | 77.479 s | 36.51 days | Projected | 40,717× |

Reference measurements through 10,000 trials passed prediction, signal, and winner parity checks. Larger reference entries extend that calibration linearly. Terasweep completed each listed size. [Method and reproduction](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/README.md) · [Measurement receipt](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/results-m1-max.json)
