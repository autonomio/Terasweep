# Integrated Rust pipeline: reverse-funnel results

**Measured 23 September 2026.** All new x86 comparisons use one Intel Xeon
Platinum 8370C virtual machine, one pinned logical CPU, the same Rust 1.89.0
compiler/options, and frozen real inputs. No ratios below multiply results from
earlier machines or standalone C kernels.

## Outcome

The updated Rust runner includes all requested levels, including real Rust
intrinsics and an independently authored AVX-512 assembly function. Both are
linked into the same Rust executable and share data packing, compact caching,
job scheduling, cost adjustment and reporting. No C scorer or C kernel is used.

On the native 1,014-configuration grid, the original factorized Rust stage is
**1083× faster in whole-process throughput** than the
full native Limen workflow. This is an application-pipeline comparison: native
Limen does raw preparation and full artifacts; Rust consumes prepared matrices
and emits a smaller summary. It is not a 1083× pure solver or language gain.

Within Rust on that exact same grid, sharing statistics, fitted predictions and
scores reduces the internal timer from **511.384 ms to
4.440 ms**, a measured **115.18×**
factorization gain. Every stage returns the same scoring/winner results; cache
cardinality naturally differs in the cache-disabled ablations.

On the **unchanged original one-million-row sweep**, the complete handwritten
pipeline is **9.16× faster** than original Rust.
In a separate 16-pair full-process A/B, switching only the kernel gives an
observed **4.06%** increase
in median throughput. That small increment is noisy on this shared VM; it is
not a universal hardware guarantee. Kernel-only claims are no longer substituted
for whole-pipeline measurements.

## 1. Existing controls were run before modification

On the connected M1 Max Mac, the saved six-case native replay completed in
3.58 seconds and the unchanged Terasweep executable in 0.02 seconds (coarse
hundredth-second wall timer). The original million-row sweep completed in
3.91 seconds and passed its existing golden comparison. Input and executable
hashes were checked. These Mac controls establish the requested before-state;
**they are not denominators in the x86 reverse funnel**.

The original source was then compiled unchanged on x86 with the same compiler
and flags used for the updated runner. Its first cold run took 9.08 seconds
and passed golden validation. The tables use interleaved repeated measurements,
not that cold observation.

## 2. Reverse funnel against native Limen — exactly 1,014 requested rows

The native path is the actual `python -m limen run --no-progress-bar` command,
using the merged Ridge SFD and existing `HistoricalData.get_any_file`,
manifest preparation, GridStrategy, sklearn Ridge, native metrics and artifacts.
The ordered GridStrategy plan is replayed exactly by every Rust level.

The frozen source is Limen's retained real-market parquet fixture, with 4,585
observations. Date splits are training 1–23 January 2026, validation 24–30
January, and test 31 January–6 February. Prepared shapes are training
**1,903 × 13**, validation **554 × 13**, and test/prices **1,431 rows**.

The grid is 3 alphas × 2 intercept choices × 13 fees × 13 slippages = 1,014.
It has only **six distinct trained models and six distinct signal paths**.
The signal rule is the native Ridge SFD's existing `prediction > 0`; sklearn
`tol` is not relabeled as Terasweep's Cholesky pivot cutoff.

| Level | Whole-process median | Range | Native-relative throughput | Repeats |
|---|---:|---:|---:|---:|
| Native Limen CLI / UEL | 31.279629 s | 30.848185–31.489824 s | 1.00× | 3 |
| Rust, no cross-trial reuse | 0.544151 s | 0.481443–0.599411 s | 57.48× | 9 |
| + shared Gram statistics | 0.061128 s | 0.057178–0.068708 s | 511.71× | 9 |
| + fitted-prediction reuse | 0.037433 s | 0.035075–0.047790 s | 835.61× | 9 |
| + original score cache | 0.028887 s | 0.027117–0.038030 s | 1082.81× | 9 |
| + constant-signal bounds | 0.031684 s | 0.026999–0.038098 s | 987.23× | 9 |
| + compact cache / categorical IDs | 0.031391 s | 0.027855–0.034461 s | 996.47× | 9 |
| + AVX-512 intrinsics written in Rust | 0.030904 s | 0.028130–0.043627 s | 1012.15× | 9 |
| + handwritten AVX-512 assembly | 0.032837 s | 0.029076–0.038901 s | 952.57× | 9 |

The later rows do not form a monotonically decreasing timing sequence, and no
such sequence was manufactured. Once 1,014 trials collapse to six score paths,
there is very little scoring left for SIMD to improve. Each model has only one
requested mixed score job, so its eight-lane batch is mostly unused. Millisecond
startup/scheduling variation dominates the differences among the final levels.

The data is also unrepresentative of rich signal variation: each native fitted
model produces 1,430 positive predictions out of 1,431 test rows. This is a
measured property of the retained workload, not a selection criterion imposed
to create fast results. It explains why this grid is useful for factorization
but insufficient on its own to characterize SIMD.

### Narrower reference calls, measured rather than projected

Calling unmodified sklearn Ridge fit/predict/MAE 1,014 times on already-prepared
matrices took **0.981746 seconds inside the loop**.
Calling the native Limen `ridge_regressor` wrapper 1,014 times on prepared
matrices, with native metrics and backtest, took **5.022037 seconds inside the loop**.
The latter's whole process took 7.459218 seconds.
These controls remove preparation but intentionally do not cache fits. The
sklearn-only control computes fewer outputs and is not another full-pipeline
funnel level. Its process includes Python/import startup, so its loop time is
the useful reported number.

## 3. Full-pipeline reverse funnel — original one-million-row workload

This preserves the original training/test binaries (4,136 × 55 and 1,136 × 55),
configuration, seed, six signal rules, thresholds, fees, slippages, failure
behavior, rounding, and original-order winner selection. All five levels passed
full stable-JSON golden comparison in every repetition. Each cell is based on
five complete process executions; order rotates and reverses between repeats.

| Level | Whole-process median | Range | Speedup over original Rust | Median peak RSS |
|---|---:|---:|---:|---:|
| Original factorized Terasweep | 6.941829 s | 6.801860–7.774548 s | 1.000× | 250.59 MiB |
| + constant-signal bounds | 2.474729 s | 2.367482–2.664892 s | 2.805× | 250.67 MiB |
| + compact cache / scheduling | 1.003812 s | 0.957339–1.139549 s | 6.915× | 53.89 MiB |
| + Rust AVX-512 intrinsics | 0.773382 s | 0.700420–0.854136 s | 8.976× | 53.99 MiB |
| + handwritten AVX-512 assembly | 0.758201 s | 0.682398–0.780800 s | 9.156× | 53.99 MiB |

The final measured throughput is approximately
**1,318,911 sampled rows/second**.
This counts all requested rows, including cached failed fits. It is not a claim
of a million independently fitted models. Peak RSS falls from about 250.6 MiB
to 54 MiB. The observed full-program improvement is about 9×, **not a certified
10×**; previous experimental source versions are retained but not mixed into
these final-version medians.

There is no measured native-Limen execution of this original million-row
configuration. Its extra signal rules and pivot-cutoff parameter have no direct
native Ridge SFD equivalent. Consequently, there is **no fabricated million-row
Limen-relative number**, and this table must not be multiplied by the first.

## 4. Handwritten assembly's incremental full-pipeline gain

Both choices are functions in the **same Rust executable**, selected at runtime.
Both use the identical compact scheduler, mixed-rule grouping by fitted model,
pack/unpack wrapper, cost calculation and final reduction. The authored
`src/optimized/score8_gpr.S` is embedded with Rust `global_asm!`; the comparison
function uses `std::arch::x86_64` intrinsics written in Rust. The C ABI notation
is only a calling convention and does not introduce a C implementation.

The separate final experiment uses 16 adjacent full-run pairs, alternating which
kernel executes first. No outliers were removed.

| Included work | Rust intrinsics median | Handwritten median | Ratio of medians | Assembly wins |
|---|---:|---:|---:|---:|
| Complete one-million-row process | 770.530 ms | 740.476 ms | 1.0406× | 12/16 |
| Mixed-score subphase inside that process | 75.848 ms | 61.768 ms | 1.2280× | 14/16 |

For the whole process, the median of the 16 individual paired throughput ratios
is 1.0428×. A descriptive paired bootstrap
95% interval is **1.0002–1.0834×**;
its lower edge is effectively no gain. The score subphase has a clearer signal:
its paired-ratio interval is **1.1725–1.2466×**.
The subphase includes grouping, packing, kernel calls and result materialization,
not merely arithmetic instructions. These bootstrap intervals do not account
for every shared-VM noise source or guarantee independent samples.

A defensible statement is: **after factorization, compact scoring caches and
Rust AVX-512 intrinsics, handwritten assembly produced about 4% higher median
whole-pipeline throughput in this paired test, with a clearer improvement in
the mixed-score phase.** The previously reported standalone C/assembly timings
are not used as evidence for this result.

## 5. Correctness and repeatability

- The original x86 build matches the retained golden result.
- Each integrated SIMD path separately checked **648,130 complete BaseScores**
  against the original scalar scorer: all eight integer fields and all nine
  floating-point bit patterns. This includes 603,570 all-off/all-on shortcuts
  and 44,560 mixed jobs. All checks passed; final summaries passed too.
- Rust boundary tests passed **138,240 complete BaseScore comparisons**, using
  real predictions, adjacent representable thresholds, signed zero, all six
  comparison rules, prefix lengths, tail lanes and controlled tradability masks.
- A **1,048,583-row** run crosses the bounded scheduler's 1,048,576-row boundary
  and matches the original control, preserving cached state and row order.
- All native grid result tables match after excluding execution time; retained
  prediction-bearing round artifacts are byte-identical across the three runs.
- Native vs Rust predictions differ by at most **4.156e-13**;
  all six full sign vectors match. Native Trainer reconstructs all six models.

No reduced precision, FMA substitution, global fast-math, temporal reassociation,
changed thresholds, dropped trials, or relaxed BaseScore comparisons were used.
This is empirical coverage of the tested data/boundaries, not a proof for every
possible input or workload.

**Economic equivalence remains a separate limitation.** Current Limen's entry
and fee-on-notional accounting differ from the original Terasweep convention,
and their output metric sets/populations differ. The update preserves original
Terasweep economics rather than silently rewriting Limen or its reference.
The native prediction comparison is strong; it does not establish complete
native/Terasweep backtest parity.

## 6. Implementation and memory behavior

The source has eight selectable levels: `naive`, `stats`, `train`, `score`,
`bounds`, `compact`, `intrinsics`, and `assembly`. `make` builds a separate
`target/terasweep-funnel` and defaults to `compact`, preserving a retained
`target/terasweep` control. Verification creates a fresh directory and never
runs the previous destructive clean target.

The compact path replaces duplicated 128-byte constant payloads with shared
references, uses categorical model/rule/threshold identities, and stores full
payloads only for mixed cases. Dense indexing is limited to 64 MiB; larger
configured address spaces use a sparse map with bounded initial reservation.
Row scheduling is processed in blocks of at most 1,048,576, while caches survive
across blocks. Ordinary generated sweeps do not allocate one row reference per
requested billion/trillion rows. Distinct cached state can still grow; explicit
CSV plans are intentionally loaded in memory.

The final source also contains a visibility-only fix required by Rust 1.63.
Compiling before and after that fix with the benchmark toolchain produced
**byte-identical x86 executables**, so it does not invalidate the recorded times.

## 7. Measurement controls and limits

Rust: 1.89.0, LLVM 20.1.7. Flags for original and updated source:
`--edition=2021 -O -C panic=abort -C target-cpu=native -C lto=fat -C codegen-units=1`.
Host: Intel Xeon Platinum 8370C, Linux x86-64 KVM. Five logical CPUs are exposed,
with a four-CPU quota and 4 GiB memory limit. Every timed child is pinned to CPU0.
OMP, OpenBLAS, MKL and Polars worker limits are set to one. Python is 3.13.5;
Limen 5.11.0 at commit `44eff3b244c80ce8473845bd60b074b3682fe441`; NumPy 2.3.5,
SciPy 1.17.0, sklearn 1.8.0, Polars 1.42.1, pandas 2.3.3, PyArrow 23.0.1.

Whole-process timing uses a monotonic parent timer around blocking child wait,
including launch, imports/input reading, computation, outputs, and exit.
`/usr/bin/time` records CPU time and peak RSS. Kernel/subphase timers are a
separate diagnostic; unlike timers are not divided to create whole-run gains.
No downloads/builds or repeated oracle checks are included in timed commands.

An early pilot used Python's timeout wait, which biased short process durations.
That entire pilot was excluded consistently, for all levels, and is retained
with an exclusion note. Final native controls were completed in this same
session before the last Rust scheduler refinements and retained because native
code, input, and commands did not change; all Rust levels were rerun after the
final behavioral refinement. Final-version results contain **138 process
measurements**. Earlier prototype results are not pooled with them.

The VM is shared and not a dedicated frequency-controlled benchmark server.
The results support this workload and execution boundary, not universal
speedup, unconstrained scaling, or predictive/trading quality.

## 8. Delivery and deployment status

The attached bundle contains `pipeline/`, unchanged `Terasweep/`, real frozen
inputs, native templates and plan, Rust tests, measured native artifacts,
all raw final timings, compiler/source/binary hashes, disassembly and replay
commands. `evidence/recorded/` is the immutable reported experiment; replay
writes to a separate empty `evidence/final/`.

The Linux pipeline was compiled and tested. Transfer to the Mac staging area
succeeded, but its Rust 1.63 compilation found a crate-visibility error. The
provided source includes the correction; the subsequent remote staging
write/build action was blocked by the tool. **The original Mac Terasweep source
and historical executable were therefore not overwritten.** Final corrected
ARM compilation is not claimed. The final package, rather than that staged
pre-correction patch, is the authoritative update.

Build-only GitHub artifact branches and the private Drive transfer file are
recorded in the bundle README. No Limen package change or merge was made as part
of this optimization. See the bundle's replay instructions for exact execution.
