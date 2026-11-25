# Reference fixtures

`train.bin` and `validation.bin` in each directory hold 192 and 64 rows,
respectively, encoded in the [Terasweep matrix format](../../docs/agents.md#matrix-format).
The datasets are CC BY 4.0.

| Directory | Dataset and preparation |
| --- | --- |
| `concrete` | Yeh's Concrete Compressive Strength. The 1,030 rows are permuted with NumPy 2.2.6 `default_rng(375)`. The first 648 rows form the training pool and the next 190 form the validation pool. The eight features are standardized on the training pool. Targets are in MPa. [UCI source](https://doi.org/10.24432/C5PK67) |
| `abalone` | Nash et al.'s Abalone. One-hot sex columns in F, I, M order followed by seven measurements, with ring count as the target. The permutation uses NumPy 2.2.6 `default_rng(375)` on 4,177 rows. The first 2,507 rows form the training pool and the next 835 the validation pool. [UCI source](https://doi.org/10.24432/C55C7W) |
| `bike` | Fanaee-T's Bike Sharing daily counts `day.csv`, ordered by date. The sole predictor is the observed daily count and the target is the next day's count, both in thousands. The first 510 pairs form the training pool and the next 110 form the validation pool. [UCI source](https://doi.org/10.24432/C5W894) |

The retained binary inputs take the first 192 rows of each training pool
and the first 64 rows of each validation pool. Verify their identity from
this directory with `shasum -a 256 -c SHA256SUMS`. Their purpose is deterministic
model parity and smoke coverage. The validation split would require a separate
holdout to estimate a tuned model's real-world accuracy.
