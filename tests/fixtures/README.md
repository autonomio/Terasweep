# Model regression fixtures

`ridge/` contains the archived Limen-prepared Ridge matrices, prices, settings,
and ordered trial plan. `returns/` contains the single-channel market-return
and price matrices used for DLinear and TiDE regression checks.

The files retain the exact bytes used by the
[four-model reference audit](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/docs/research/model-reference-parity.md).
The original raw fixture, its provenance, and preparation script are in the
[pinned benchmark checkout](https://github.com/autonomio/Terasweep/tree/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab).
`scripts/verify_model_parity.py` checks `SHA256SUMS` before testing predictions.
