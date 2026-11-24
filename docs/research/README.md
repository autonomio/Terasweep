# Terasweep research

Vaquum Alpha Research maintains the experiment bundle with raw results,
checksums, and replay instructions. The public records below are pinned to
the Terasweep revisions that preserve the evidence. [Research bundle](https://github.com/Vaquum/alpha-research/tree/main/terasweep-performance)

| Evidence | What it establishes |
| --- | --- |
| [Model reference parity](model-reference-parity.md) | Prediction and signal agreement for the retained model inputs and settings. |
| [Single-model comparison](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/single-model/README.md) | One fit and prediction on matched inputs, with the measured timing boundary for each backend. |
| [Matched sweep scaling](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/sweep-scaling/README.md) | Same-workload reference calibrations and measured Terasweep runs from 10 through one billion sampled trials for Ridge, DLinear, LightGBM, and TiDE. |
| [Original Ridge benchmark record](https://github.com/autonomio/Terasweep/blob/d5dd91675e647e38dd2f6f209e68eaa71e27c1ab/benchmarks/README.md) | Historical stage ablations, AVX-512 kernel studies, and large-scale Ridge runs with their own workloads, hosts, and reference projections. |

The [execution levels](optimization-pipeline.md) describe the scoring paths and
how to check them. The [workload notes](workload.md) document the retained inputs.
The [offline learning course](../masterclass/README.md) walks through the
factorization and shows how to read these experiment records.

## Citing an experiment

Cite [Terasweep](../../CITATION.cff) and include a permalink to the full Git
commit used. Cite the report or measurement receipt that supports the result,
along with its hardware, toolchain, model versions, data and split, seed, search
space, and timing boundary. For projected reference times, include the largest
measured calibration and the projection rule.
