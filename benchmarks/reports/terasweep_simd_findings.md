# Terasweep SIMD exploration — 22 September 2026

## Scope and evidence

This is an exploration of the existing original 1,000,000-row ridge workload, not a new Limen comparison. Work was performed on the connected Apple M1 Max. No AVX-512 instructions were executed. Limen and the production Terasweep sources/executable were not changed. All modified Rust sources and binaries are isolated under:

`/Users/mikkokotila/dev/_reports/terasweep-simd-20260922-u3uJGl/`

Machine-readable measurements accompany this report in `terasweep_simd_exploration.json`.

## Profiling

The original executable was sampled during one run. It completed in 4.17 seconds and passed the existing golden comparison. In the captured window, 3,115 of 3,300 top-of-stack samples were in `score_cached`, which includes inlined base-score work. Sampling started after process launch and missed early work; this is not a complete wall-time breakdown.

A diagnostic source copy added phase timers, named non-inlined scoring functions, and counters. Its output also passed the golden comparison. Phase times were:

| Phase | Seconds |
|---|---:|
| Training-statistics preparation | 0.010118333 |
| Fit/predict precomputation | 0.098406250 |
| Sampled sweep loop | 4.178596541 |
| Detailed winner reporting | 0.000777541 |

These times belong to an instrumented build, not an optimization ablation or an exact allocation of production wall time.

Original scoring disassembly shows scalar floating-point comparisons, indirect signal-rule dispatch within the scans, a serial equity product, and category-indexed moment updates. This identifies concrete control-flow and vectorization opportunities; it does not establish how the same source will compile on x86.

## Exact all-on/all-off shortcut

The 648,130 base-score cache misses divided into:

| Signal pattern | Count |
|---|---:|
| All off | 394,015 |
| All on | 209,555 |
| Mixed | 44,560 |

Thus 603,570 misses, or 93.1248%, correspond to one of two constant signal patterns.

An isolated prototype computes min/max prediction and absolute-prediction bounds once per encountered fitted configuration. For finite predictions, exact comparisons against those bounds detect all-off and all-on cases for all six signal rules. Non-finite prediction vectors use the original scorer.

The first actual occurrence of each constant pattern is evaluated with the original scorer. The resulting BaseScore is reused for later constant-pattern misses within the same fixed-data sweep. Mixed patterns still use the original scorer. All original score keys and cache-entry counts remain; candidate generation, fit failures, costs, winner selection, and reporting are unchanged.

No market data, targets, or predictions were fabricated. The shortcut is reuse of an already-computed result, not pruning or approximate scoring.

A validation run recomputed the original BaseScore for every shortcut and compared integer fields exactly and all floating-point fields by bit pattern. All 603,570 checks passed. The complete golden comparison passed as well. This proves parity for these encountered cases, not universal correctness for every possible future input.

## Unprofiled paired measurements

Same compiler/build options, same machine, same original data/config, separate processes. Execution order was original/prototype, prototype/original, original/prototype. Background activity was not stopped and CPU affinity was not controlled.

| Pair | Original wall seconds | Bounds prototype wall seconds |
|---|---:|---:|
| 1 | 3.81 | 0.77 |
| 2 | 4.07 | 0.74 |
| 3 | 4.01 | 0.82 |
| Median | 4.01 | 0.77 |

The ratio of medians is 5.2078×. Every output passed golden validation. The first exploratory prototype run took 0.67 seconds; the paired median above is the appropriate headline, not that best observation.

Median instructions retired fell from 64,980,727,442 to 7,478,546,203. Median peak RSS remained approximately 345 MB because the prototype intentionally retained the original cache structure.

This is not an AVX-512 speedup. It is evidence for work elimination before SIMD.

## Residual bottleneck

A short sample of the prototype captured 601 stacks. Of these, 274 were in `compute_base_score`; 174 were in the listed hashing, insertion, and rehashing functions. Startup and training may be missed. These roughly 46% and 29% shares describe the captured window only.

A 10× improvement over the new paired original baseline would require 0.401 seconds. From the prototype's 0.77-second median, another 1.9202× total-program improvement is required. Because substantial work now lies outside base scoring, vectorizing only that kernel is not a reliable route to the whole target.

## Recommended AVX-512 design to test

First specialize signal-rule dispatch outside the bar loop and avoid repeated comparison of the same prediction. Then test SIMD lanes representing independent score jobs, preferably different requested thresholds of one model and rule. Eight FP64 lanes can process eight thresholds while broadcasting one prediction and one price row.

Keep time sequential within each lane. Each lane maintains its own equity product and four category-count/moment accumulators. Masked updates preserve inactive accumulators. This avoids reassociating the time-series reductions, unlike vectorizing eight successive bars and horizontally reducing them.

This design is a proposal, not implemented or benchmarked AVX-512 code. It needs checks for lane occupancy, register spills, batch-scheduling overhead, tails, non-finite values, and exact threshold boundaries. Preserve logical run IDs and rounded tie-breaking despite physical batching. Do not eagerly compute the entire theoretical score grid just to fill lanes.

A complementary cache change is important: constant scores need not occupy separate 128-byte payloads for every logical key. Keep logical-key accounting but store compact references/tags to the two shared constant scores and an arena of mixed scores. Only 44,560 observed mixed scores require distinct payloads in this workload. Compact categorical keys and elimination of avoidable training-key hashing should also be measured. Do not allocate a full dense score grid without a memory-cost calculation.

Cholesky and prediction SIMD come after re-profiling this reduced engine. Their roughly 0.1-second diagnostic precomputation cost becomes more significant against a 0.401-second total target, but optimizing them first would have targeted a small part of the original sweep.

## Hardware, compiler, and acceptance

Only the ARM Mac was connected through Remote Desktop Commander. Historical May hosted-runner probes do not establish a currently available AVX-512 machine. A real x86 host with runtime-verified features is required for native measurement; emulation cannot establish throughput.

The local compiler is rustc 1.63.0. Stable AVX-512 intrinsics and target features needed for a modern Rust implementation require a newer pinned toolchain; many were stabilized in 1.89.0. Do not silently update the user's default toolchain.

Compare scalar/specialized, AVX2, and AVX-512 implementations on the same x86 CPU using the same pinned toolchain and workload. Separate compiler, algorithm, ISA, and threading gains. Start with intrinsics and inspect generated assembly before adding handwritten assembly. Preserve the original arithmetic ordering first; no global fast-math, automatic FMA substitution, or FP32 conversion as an unlabelled optimization.

Validate intermediate score fields, signals, failure classifications, per-row metrics, and tie-breaking, not only final winners. Use same-platform scalar output as the SIMD oracle and investigate any cross-platform golden drift separately. Optimizing Terasweep does not repair the previously documented economic-semantics differences from native Limen.

## Retained evidence on the Mac

- `sample.txt`, `time.txt`, `summary.json`: original sampled run.
- `profile-src/`, `profile-sweep`, `profile-time.txt`, `profile-sample.txt`: phase/counter diagnostics.
- `score-original.asm`: original scorer disassembly.
- `bounds-src/`, `bounds-sweep`: isolated shortcut prototype.
- `bounds-check.txt`, `bounds-checked-summary.json`: all shortcut comparisons and golden-compatible result.
- `original-pair-*.time`, `bounds-pair-*.time`, corresponding JSON summaries: paired runs.
- `bounds-sample.txt`: residual sample.
- `exploration.json`: collected measurements.

## Primary technical references

- Intel, Intel AVX-512 Instructions: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-avx-512-instructions.html
- Intel, Tuning SIMD vectorization: https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-simd-vectorization-when-targeting-intel-xeon-processor-scalable-family.html
- LLVM, Auto-Vectorization: https://llvm.org/docs/Vectorizers.html
- Rust, masked FP64 addition: https://doc.rust-lang.org/core/arch/x86_64/fn._mm512_mask_add_pd.html
- Rust, runtime x86 feature detection: https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html
- Rust 1.89 release announcement: https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/
