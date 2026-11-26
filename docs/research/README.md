# Terasweep research

The [reference audit](model-reference-parity.md) checks the current package
on public regression and forecasting fixtures. It measures single fits and
predictions against direct model references and checks the sweep's exact fit
counts and winner. The [learning course](../masterclass/README.md) explains the
method.

## Archived pilot

The archived pilot runner, original learning material, and its benchmark
receipts are in the Vaquum Alpha Research bundle. Historical measurements
describe the pinned code and workload used for those experiments. The pilot
includes a same-runner Ridge ablation where sharing training statistics,
fitted predictions, and scores reduced elapsed time from 511.384 ms to 4.440 ms
on its 1,014-entry plan, a **115×** speedup. The scaling report records separate
timings and projections on the pilot workloads.
[Factorization ablation](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/reports/terasweep_reverse_funnel_report.md#outcome) · [Sweep scaling report](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/README.md)

[Research bundle](https://github.com/Vaquum/alpha-research/tree/main/terasweep-performance) ·
[Runnable pilot snapshot](https://github.com/Vaquum/alpha-research/tree/main/terasweep-performance/finance-pilot) ·
[Public historical measurement record](https://github.com/autonomio/Terasweep/tree/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks)

## Citing an experiment

Cite [Terasweep](../../CITATION.cff) and include a permalink to the full Git
commit used. Cite the report or measurement receipt and record the hardware,
toolchain, model versions, data and split, search space, and timing boundary.
For projected times, include the measured calibration and projection rule.
