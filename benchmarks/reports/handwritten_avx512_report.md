# Handwritten AVX-512 after intrinsics: incremental experiment

## The claim this experiment supports

After improving the AVX-512 intrinsics implementation, replacing the **mixed-signal base-scoring kernel** with hand-authored x86-64 assembly increased median scoring throughput by **6.96%** in the principal matched-block experiment. Median accumulated scoring time for 44,560 requested jobs was **54.547 ms with intrinsics** and **50.997 ms with handwritten assembly**: **6.51% less time**, or **1.070× throughput**.

This is a **kernel-level result in a shared VM**, not a Rust Terasweep end-to-end speedup, not an additional 10×, and not a new Limen benchmark. The effect is modest and environment-dependent. We did not multiply it by earlier optimizations measured on another machine.

## What is genuinely handwritten

`src/score8_gpr.S` is a **143-line, manually authored assembly source file**. It was not obtained by compiling C to assembly and relabeling the output. Its calling convention, vector-register assignment, integer-register assignment, masks, branches, numerical instructions, and stores were written explicitly. It is assembled into a separate object and called as an external function.

The function uses eight AVX-512 FP64 lanes for eight independent scoring jobs. The jobs share a fitted model and can have different thresholds and signal rules. Time advances sequentially within every lane.

The final schedule keeps position-state masks in 32-bit general-purpose registers, uses Boolean instructions such as `andn`, and transfers final masks into AVX-512 mask registers for the arithmetic. The accumulators stay in explicitly assigned vector registers. A ternary bitwise instruction applies the same per-lane sign/absolute-value transformation as the tuned intrinsics.

Example from the authored source:

```asm
andn edx, ebx, esi
kmovw k2, edx
vpaddq zmm3{k2}, zmm3, zmm18
vaddpd zmm7{k2}, zmm7, zmm21
vaddpd zmm11{k2}, zmm11, zmm22
```

The file's SHA-256 is:

`7114c244ff89646de1a260a8fcb8654c77f794f5f3a1aa84aeadad37ccc2bfcb`

The code uses separate multiplication and addition. No FMA substitution, FP32 conversion, fast-math, horizontal reassociation, or relaxed correctness tolerance was introduced.

## Why the baseline matters

The objective was an increment **beyond good intrinsics**, not an impressive ratio against scalar code.

The first handwritten version used mask registers for essentially all mask-state arithmetic. It passed validation, but its apparent advantage diminished after improving the intrinsics. That version was not used as evidence of a successful final optimization.

Four prepared-input intrinsics variants were tested: 8-bit ordinary mask expressions and explicit 16-bit mask intrinsics, each compiled with GCC and Clang. The earlier mixed-rule intrinsics implementation was also retained with both compilers. The strongest candidate in discovery was the prepared-input, 8-bit-mask version compiled with Clang 17. It remained faster than the corresponding GCC version in the full-set confirmation.

The final handwritten implementation uses a different hand-scheduled GPR/mask arrangement. The strongest tested intrinsics is not proof that every possible intrinsic implementation has been exhausted. The experiment demonstrates an improvement over the versions actually tested, not the impossibility of expressing equivalent performance in intrinsics.

## Same-wrapper A/B boundary

Both final candidates receive the same `Packed8` input and write the same `Raw8` output. The **same compiled C wrapper** performs grouping, rule/threshold packing, the indirect call, unpacking, and complete output stores. Only the kernel function pointer changes.

Included in the timing:

- grouping the already sorted requested jobs into eight-lane batches;
- packing thresholds, sign/absolute-value transformations, and inclusive-rule masks;
- calling and executing the selected scoring kernel;
- converting its full results into the original 128-byte `Base` representation and storing them.

Excluded:

- file loading, training-statistics preparation, fitting, and prediction;
- sampling, deduplication, constant-signal classification, initial sorting, and chunk planning;
- validation and logging;
- cache management, fee/slippage application, winner selection, and full experiment artifacts.

No trial pruning, new score-cache shortcut, reduced precision, or smaller output contract accounts for the measured assembly increment. The compared variants evaluate identical requested jobs.

## Hardware and toolchain

The execution environment changed from the previously reported AMD host. The current guest reports **Intel Xeon Platinum 8573C**, x86-64 under KVM. Both AVX2 and AVX-512 execute successfully. All results here were remeasured on this guest; the earlier AMD timings were not used as its baseline.

The process is single-threaded and pinned to guest logical CPU 0. The VM exposes five logical CPUs, four CPU-equivalents of quota, and a 4 GiB memory limit. It is shared, not an isolated hardware benchmark.

The common driver is compiled with GCC 14.2.0. The strongest intrinsics object is compiled with Clang 17.0.0. Both use:

```text
-O3 -g -march=x86-64-v3 -ffp-contract=off -fno-fast-math -Wall -Wextra
```

The explicit kernels target AVX-512. The handwritten GPR version also uses BMI1 `andn`; the executable checks the required CPU capabilities. Disassembly and the environment record are retained in the bundle.

## Workload

The uploaded frozen matrices are unchanged: **4,136 training rows, 55 features, and 1,136 test/price rows**. The prior experimental preparation and sampling code is retained. It reconstructs the million-row workload's 2,304 model configurations, 627 failed model configurations, and 648,130 unique valid scoring keys. Of these, **44,560 have mixed signals** and are the timed kernel workload. Eight-lane, same-model/mixed-rule batching has approximately **88.27% useful-lane utilization**.

This is still the inherited C experimental preparation/reference, not a native-x86 build of Rust. No Rust toolchain archive was uploaded in this turn. No production source or executable on the user's Mac was modified.

## Measurements

### Whole-job-set checks

The initial confirmation used 32 rotated repetitions, each timing an uninterrupted evaluation of all 44,560 jobs. The best intrinsics median was **53.928 ms**, versus **50.408 ms** for the final handwritten kernel, a ratio of **1.070×**. Assembly was faster in **22 of 32** same-repetition comparisons.

A subsequent confirmation used the exact shared wrapper, 24 balanced blocks, and four full evaluations per timed block. The strongest intrinsics median per evaluation was **56.301 ms**, versus **52.918 ms** for assembly: **1.064×**. Assembly won **15 of 24** blocks. Large host-related timing excursions remained; no samples were discarded.

### Principal matched-block confirmation

To reduce the exposure of each A/B pair to long timing swings, the sorted jobs were partitioned at **model boundaries**, with up to 16 model groups in a chunk. There were **105 chunks**, and no model was split across chunks. Each chunk was evaluated by both kernels, alternating which ran first by chunk and repetition.

Across 32 complete passes of this protocol:

| Implementation | Median sum of timed chunks per pass | Minimum–maximum |
|---|---:|---:|
| Strongest tested AVX-512 intrinsics | 54.547 ms | 45.527–70.280 ms |
| Handwritten AVX-512 assembly | 50.997 ms | 44.262–66.761 ms |

Ratio of medians: **1.069614×**. Median of per-pass paired ratios: **1.063691×**. Assembly was faster in **27/32 passes**. The reported time is the sum of the timed chunks for one implementation, not the wall duration of the full alternating two-implementation experiment.

The repeated median advantage across these protocols supports a small measured kernel improvement. It is not a general hardware speed guarantee or a claim of tightly bounded statistical uncertainty. Shared-VM noise remains visible, and all outliers are retained.

## Correctness

The final handwritten kernel passed:

| Validation | Result |
|---|---:|
| All unique valid scoring keys from the original million-row sample | 648,130 / 648,130 |
| Complete fields compared for every score | 17, exact integers and FP64 bit patterns |
| Additional thresholds at real predictions, neighboring representable values, signed zero, and observed extrema | 19,440 / 19,440 |
| Prefix lengths, missing-bar control-flow masks, and requested tail-lane counts | 30,240 / 30,240 |
| All timed mixed-signal jobs | 44,560 / 44,560 |

The missing-bar tests change only tradability flags for control-flow coverage, retaining the actual prediction/price arrays. They do not create synthetic market observations and are not included in performance timings. Prefixes include lengths 0, 1, 2, values around vector widths, and the full input; tail tests also detect writes beyond the requested lane count.

The local oracle is the unchanged C reference scorer from the supplied experiment. The prior report separately records that reference matching the original Rust scorer on the Mac for all 648,130 keys. That prior cross-language check is inherited evidence, not a new Rust execution in this session.

## The supported statement

> After the workload reductions and AVX-512 intrinsics optimization, we replaced the remaining mixed-signal scoring kernel with handwritten x86-64 assembly. On the tested Intel Xeon VM, this added approximately 7% median scoring throughput, with bit-for-bit identical complete base-score outputs.

For a **whole-Terasweep** statement, the same two kernels still need to be integrated and timed in the same native Rust runner, with all scheduling, cache, cost, and reporting overhead included. That measurement has not been performed.

## Reproduction

The accompanying archive includes the hand-authored assembly, intrinsics variants, unchanged experimental reference/preparation, frozen inputs, replay script, build commands, raw validation and timing logs, source/input hashes, and disassembly. It excludes generated benchmark binaries and duplicate large output arrays.

Run `./replay.sh /absolute/path/to/a/new/directory` on compatible x86-64 Linux with GCC, Clang, and Python installed. The script refuses to overwrite an existing output directory and verifies the retained source/input checksums before compiling. No network service or additional package download is required.
