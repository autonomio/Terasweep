# LightGBM

## Input

Bundled 55-column features and a continuous next-return target. Training uses the official LightGBM regression objective. The recorded train and test matrices were **4,136 × 55** and **1,136 × 55**. Settings were `learning_rate=0.1`, `num_trees=32`, `num_leaves=7`, `max_bins=32`, `lambda_l2=0.1`, `min_data_in_leaf=20`, `seed=20260514`.

## Output

The fit returned 1,136 continuous predictions in test-row order. The first three were `-0.032412537`, `0.012996523`, `0.0088985605`.

## Reference and equivalence

Official LightGBM 4.6.0 Python regression using the same settings and seed. Bit-identical predictions, with zero differences in the `prediction > 0` signal vector.

## Single run

On Apple M1 Max, Terasweep took **39.359 ms** and the reference took **49.463 ms**, a **1.26×** speed ratio for one fit and predict call. [Comparison and reproduction](../../benchmarks/single-model/README.md) · [Recorded inputs and results](../../benchmarks/single-model/results-m1-max.json)
