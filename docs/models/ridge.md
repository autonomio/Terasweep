# Ridge

## Input

Prepared tabular features and a continuous next-return target from the archived Limen fixture. The recorded train and test matrices were **1,903 × 13** and **1,431 × 13**. Settings were `alpha=1.0`, `fit_intercept=false`, `solver_eps=1e-12`.

## Output

The fit returned 1,431 continuous predictions in test-row order. The first three were `0.045553036`, `0.044811896`, `0.027638978`.

## Reference and equivalence

Ridge with scikit-learn’s Cholesky solver. Maximum absolute difference `5.77e-15` at `rtol=atol=1e-10`, with zero differences in the `prediction > 0` signal vector.

## Single run

On Apple M1 Max, Terasweep took **0.223 ms** and the reference took **1.987 ms**, a **8.91×** speed ratio for one fit and predict call. [Comparison and reproduction](../../benchmarks/single-model/README.md) · [Recorded inputs and results](../../benchmarks/single-model/results-m1-max.json)
