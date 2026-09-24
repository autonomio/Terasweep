# AVX-512 scoring-kernel experiment — 22 September 2026

## Result and scope

The best tested AVX-512 implementation took **60.285 ms** to evaluate the original workload's **44,560 distinct mixed-signal base-score jobs**. Across 12 interleaved repetitions, this was **3.974× faster than the C reference scorer**, **3.396× faster than a specialized scalar C scorer**, and **1.299× faster than the best tested AVX2 implementation**.

These are **C kernel measurements, not end-to-end Rust Terasweep measurements**. No new native-Limen comparison or whole-sweep speedup is claimed.

The uploaded archive's SHA-256 matched the archive prepared on the Mac:

`6c1e549ff86b80446e2231e3f219760ae0fb746b38a91e1316bb4d0d3e1e4bc4`

## Why this is a C experiment

The x86 execution environment exposes AVX2 and AVX-512F and successfully executes the kernels. GCC 14.2.0 is installed. Rust is not installed, and attempts to download its toolchain into this environment failed. The unchanged Rust runner was therefore **not built or benchmarked here**.

Rather than substitute a new full-sweep runner and call it Terasweep, the experiment isolates the identified scoring bottleneck. Its C reference is a direct translation of the supplied `results.rs::compute_base_score`; preparation translates the supplied ridge normal-equation/Cholesky loops and sampling sequence to obtain workloads from the retained input matrices. This preparation is not a new production model and is excluded from kernel timings.

## Hardware and compilation

- Reported processor: AMD EPYC 9V74 80-Core Processor, x86-64, KVM guest.
- Guest exposes five logical CPUs and a four-CPU-equivalent quota, with a 4 GiB memory limit.
- Benchmark process pinned itself to logical CPU 0 and ran single-threaded.
- Compiler: GCC 14.2.0.
- Common options: `-O3 -g -march=x86-64-v3 -ffp-contract=off -fno-fast-math -Wall -Wextra`.
- Explicit SIMD functions have `target("avx2")` or `target("avx512f")` attributes.
- FP64 retained throughout; no fast-math, fused multiply-add substitution, or horizontal reassociation of the per-job temporal reductions.
- This is a shared virtual environment, not an isolated hardware-limit benchmark.

The compiled disassembly includes executed AVX-512 masked FP64 additions and vector multiplications, for example:

```asm
vaddpd zmm4{k2}, zmm6, QWORD BCST [r13+r10]
vaddpd zmm19{k1}, zmm19, zmm1
vmulpd zmm3, zmm3, zmm1
```

The experiment uses intrinsics. It does not use handwritten assembly.

## Workload identity

The supplied train/test/price files were used without alteration. They describe 4,136 training rows, 55 features, and 1,136 test/price rows. The experiment reconstructs the original configuration's seed 20260514, 384 log-spaced alphas, two intercept choices, three solver cutoffs, six signal rules, and signed/absolute threshold grids.

Observed counts match the previously recorded workload:

| Quantity | Count |
|---|---:|
| Training configurations | 2,304 |
| Failed training configurations | 627 |
| Failed sampled rows | 271,898 |
| Unique successful base-score keys | 648,130 |
| All-off keys | 394,015 |
| All-on keys | 209,555 |
| Mixed-signal keys | 44,560 |

Only the mixed-signal jobs are timed in this kernel experiment. Constant-pattern shortcut performance, hashing, cache representation, fee/slippage application, and full-sweep orchestration are not measured here. Matching these counts alone is not proof of all cross-platform prediction bits; numerical validation is described separately below.

## Timings

Each value below is for all 44,560 mixed-signal jobs. There were 12 repetitions, rotating the order of six implementations. Each implementation occupies every timing-order position twice. Validation executes each implementation before the timed sequence.

| Implementation | Median | Observed minimum–maximum | Speedup vs C reference |
|---|---:|---:|---:|
| C reference scorer | 239.565 ms | 219.412–255.650 ms | 1.000× |
| Specialized scalar C | 204.712 ms | 190.290–222.676 ms | 1.170× |
| AVX2, one model/rule per batch | 83.592 ms | 78.050–91.492 ms | 2.866× |
| AVX-512, one model/rule per batch | 83.368 ms | 78.254–89.490 ms | 2.874× |
| AVX2, one model with mixed rules per batch | 78.282 ms | 73.825–94.894 ms | 3.060× |
| AVX-512, one model with mixed rules per batch | 60.285 ms | 56.838–70.688 ms | 3.974× |

The source reference is compiled with the same optimization options as the other implementations. There is no forced no-vectorization flag to artificially slow it down. “Specialized scalar” describes its programming structure, not a promise that GCC emits no SIMD anywhere.

### Timing boundary

Included: grouping the already-sorted job sequence into batches, packing thresholds and lane-rule masks, evaluating base scores, and writing the complete outputs.

Excluded: loading files; forming training statistics; fitting and predicting; generating the one-million-row sample; classifying constant patterns; deduplicating and sorting jobs; correctness validation; fingerprinting outputs; cost application; winner selection; and experiment-artifact serialization.

Consequently these numbers cannot be substituted into the prior full-process Limen/Terasweep table.

## What changed between the two batching designs

### First design: eight thresholds for one model and one rule

Each lane holds a different threshold. All lanes share the current prediction and price values. The time axis remains sequential within each lane. Many model/rule groups are small, however.

Useful-lane utilization was **73.917% for AVX2** and **52.696% for AVX-512**. The wider implementation was effectively tied with AVX2.

### Improved design: eight requested score jobs for one model, allowing different rules

The batch can now cross rule boundaries while retaining the same fitted model and shared prediction loads. Lane masks encode whether a lane needs a strict/inclusive comparison, a sign reversal for less-than rules, or an absolute value.

Useful-lane utilization rose to **94.511% for AVX2** and **88.273% for AVX-512**. The best AVX-512 version is approximately 30% faster than the corresponding AVX2 version.

Every lane still accumulates its equity product and category moments in original chronological order. Entry/exit counts are reconstructed from category counts; their equality is included in validation. Tail lanes duplicate a valid threshold/rule internally, and only requested outputs are stored.

This result supports testing cross-rule, same-model batching in the Rust implementation. It does not establish that grouping/scheduling and cache costs will be negligible in an integrated runner.

## Correctness evidence

### Independent C-reference/Rust comparison on the Mac

A diagnostic copy of the **original Rust source** exported original predictions, precomputed prices, and the complete `compute_base_score` result for all **648,130 unique base-score keys** in the one-million-row sample. The same C reference-scoring function was compiled on the Mac with Clang using strict FP options and run against this fixture.

Observed result:

```text
Original Rust scorer exported 648130 unique base scores
C reference equals original Rust bit-for-bit: 648130 complete BaseScores, all 17 fields each
```

All eight integer fields and all nine floating-point bit patterns were compared, rather than just rounded final scores.

The retained validation is at:

`~/dev/_reports/terasweep-c-oracle-fU6UxO/`

The diagnostic copy and its 113 MB fixture are separate from the original project. This check establishes reference parity on the Mac. It is not a native-x86 Rust execution or an integrated-runner golden check.

### SIMD validation in the x86 environment

Each of the five optimized variants matched the C reference on every field of all **44,560 mixed-signal jobs**.

The same-rule AVX2 and AVX-512 kernels additionally passed **19,440 boundary jobs each**, using thresholds at selected real predictions, one representable value on either side, signed zero, and observed extrema.

The mixed-rule versions additionally passed **3,240 mixed-rule boundary jobs each**, using actual predictions and adjacent representable thresholds. The raw log's `boundary_validation.jobs_per_simd_mode` field refers to the 19,440 same-rule tests; the extra mixed-rule assertion loop is present in the source but does not print a separate counter.

All measured outputs had fingerprint `b4bba056c98c06c0`. Fingerprinting prevents unused results and checks repeat consistency; the full pre-timing bitwise comparison, not this fingerprint, is the substantive correctness test.

## Interpretation

The useful measured result is a **3.974× scoring-kernel speedup over the validated C reference**, with **1.299× specifically over the best AVX2 implementation tested**. Improved batching was necessary to make 512-bit width pay off.

There is no supported 10× whole-program claim. The earlier bounds shortcut was measured on an M1 Max, while these kernel measurements are on an x86 guest and cover a different timing boundary. Multiplying the two factors would not be a valid measured end-to-end result.

The required next measurement is the original Rust runner, the bounds/compact-cache scalar path, and integrated AVX2/AVX-512 paths built with one pinned x86 compiler and run on the same x86 machine, including all grouping, cache, cost, and reporting overhead.

## Offline Rust toolchain prepared on the Mac

The official Rust 1.89.0 x86-64 Linux compiler and standard-library archives were downloaded and checked against their published SHA-256 files. They have **not** been installed on the Mac and no default toolchain was changed.

Bundle:

`~/Downloads/terasweep-linux-toolchain-DUwjgC/rust-linux-offline.tar`

Reported size: 106 MB.

SHA-256:

`de9f039ba6db992f4e007c7cccc60833cc399d6847eb55eb65d6a07f60db78b8`

Uploading that bundle into the conversation enables the missing native-Rust build step without paying for another machine.

## Reproduction and evidence

The accompanying experiment archive includes the exact C source, original supplied inputs and source snapshots, input checksums, replay script, raw timing logs, compiler output, disassembly, environment record, and the recorded Mac oracle-validation result. The Mac's large oracle fixture and the compiler archives are not duplicated in the download.

Run `./replay.sh /absolute/path/to/a/new/output-directory` on compatible x86 Linux with GCC, Python, and AVX-512F. The script verifies the original input/configuration hashes and refuses to overwrite an existing output directory. It reproduces the kernel experiment, not a full Terasweep benchmark.

Primary instruction/toolchain references consulted:

- https://doc.rust-lang.org/core/arch/x86_64/fn._mm512_mask_add_pd.html
- https://doc.rust-lang.org/core/arch/x86_64/fn._mm512_cmp_pd_mask.html
- https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/

All performance numbers in this report come from the retained experiment logs, not these external references.
