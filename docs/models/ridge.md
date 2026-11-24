# Ridge

## Input

Prepared tabular features and a continuous next-return target from the archived Limen fixture. The recorded train and test matrices were **1,903 × 13** and **1,431 × 13**. Settings were `alpha=1.0`, `fit_intercept=false`, `solver_eps=1e-12`.

## Output

The fit returned 1,431 continuous predictions in test-row order. The first three were `0.045553036`, `0.044811896`, `0.027638978`.

## Reference and equivalence

Ridge with scikit-learn’s Cholesky solver. Maximum absolute difference `5.77e-15` at `rtol=atol=1e-10`, with zero differences in the `prediction > 0` signal vector.

## Single run

On Apple M1 Max, Terasweep took **0.223 ms** and the reference took **1.987 ms**, a **8.91×** speed ratio for one fit and predict call. [Comparison and reproduction](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/README.md) · [Recorded inputs and results](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/results-m1-max.json)

## Sweep scaling

On M1 Max, the sweep used **256 × 13** training and **128 × 13** test matrices and 6 distinct fit settings. The uncached Limen’s scikit-learn Ridge reference refit each trial and used the same scalar scorer on the same trial order.

| Sampled trials | Terasweep measured | Reference time | Reference basis | Ratio |
| ---: | ---: | ---: | --- | ---: |
| 10 | 0.600 ms | 1.361 s | Measured | 2,267× |
| 100 | 0.828 ms | 1.127 s | Measured | 1,361× |
| 1,000 | 0.913 ms | 1.780 s | Measured | 1,949× |
| 10,000 | 2.524 ms | 3.282 s | Measured | 1,300× |
| 100,000 | 6.979 ms | 23.286 s | Projected | 3,337× |
| 1,000,000 | 52.075 ms | 3.72 min | Projected | 4,289× |
| 10,000,000 | 487.795 ms | 37.06 min | Projected | 4,559× |
| 100,000,000 | 5.365 s | 6.17 h | Projected | 4,143× |
| 1,000,000,000 | 51.380 s | 2.57 days | Projected | 4,326× |

Reference measurements through 10,000 trials passed prediction, signal, and winner parity checks. Larger reference entries extend that calibration linearly. Terasweep completed each listed size. [Method and reproduction](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/README.md) · [Measurement receipt](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/results-m1-max.json)
