# Git handoff — 24 September 2026

Repository: `https://github.com/mikkokotila/Terasweep` (private).
Mac working copy: `~/dev/Terasweep`.

## Source identity

- Original source baseline: commit `02d3faa75def4064bb90ae1f12ad7ee3f31f88e2`,
  annotated tag `baseline-2026-05`.
- Integrated runtime: commit `70525ba2673ffe6d0cff7f94df82bb786446f925`.
- `pipeline-source-sha256.json` pins all 22 files in the delivered final pipeline.
  The final visibility correction is present. The handoff adds documentation,
  archives, and verification only; it does not alter the measured runtime.
- Native Ridge remains the uncached sklearn-backed Limen SFD merged in
  `Vaquum/Limen` at `44eff3b244c80ce8473845bd60b074b3682fe441` (PR #826).
  No new Limen runtime changes are part of this handoff.

## Handoff checks

The x86 handoff checks in this directory build the exact delivered source whose
hashes match the Git runtime. They are correctness/build checks, not new timings
pooled into the reported measurements. With the offline official Rust 1.89.0
compiler, the boundary suite passed 138,240 full-BaseScore comparisons, and both
intrinsics and assembly million-row runs passed the existing golden comparison.
See `x86-environment.txt`, `x86-tests.txt`, `x86-intrinsics-golden.txt`, and
`x86-assembly-golden.txt`. `x86-validated-source.json` records the tested files.
Fresh-GitHub-clone validation is recorded separately after that clone completes.

## What is preserved

`../archives/` contains the six original delivered archives. `../reports/` makes
the readable reports, plots, and machine-readable summaries directly browsable.
`../mac-history/` preserves the earlier Mac-only experiments, including the
binary C-oracle fixture and the original Mac executable/results. Its inventory
lists file identities and excluded rebuildable files. Archived prototypes can
contain older source versions and earlier measurements; they are not the
current implementation.

## What is deliberately not vendored

Rust installers, Python wheels/environments, OS/compiler caches, and duplicate
clean Git worktrees are rebuildable dependencies rather than unique project
work. Toolchain versions, dependencies, and available installer checksums remain
in the experiment records. No account credentials or Git authentication state
are required in this repository.

The two temporary toolchain-only branches were already pushed:

- `mikkokotila/avx512-test`, `bench/terasweep-rust-offline-20260923`,
  commit `4baa361bc52bf7e860a0ec2faf07005ccd5b086c`.
- `Vaquum/Limen`, `bench/offline-rust-20260923`,
  commit `de839c7f0282fa24dce4b7757783b9cb7e0dd07e`.

Their patches and identities are also preserved in `../mac-history/`. These
branches are not the Terasweep runtime and were not merged into Limen. Historical
Mac scratch directories and those branches are retained rather than deleted;
unique code and evidence are now in this handoff, not solely in those locations.
Unrelated pre-existing files in the user's Limen checkout are untouched.

The original `target/terasweep` and `runs/` stay ignored and are preserved. The
new build target is `target/terasweep-funnel`. A clean Git status means no
uncommitted project material, not that ignored rebuildable files cannot exist.

## Fresh clone result

A new HTTPS clone from GitHub at `1b1cd072ede293b2744bf00c6f0317a2e97b0b34`
passed all archive/source checks and `git fsck --full`. The Mac built the
final source with its existing Rust 1.63.0 compiler, and the compact million-row
run matched the golden output. The working copy at `~/dev/Terasweep` also builds;
its historical executable is unchanged. See `fresh-clone-validation.json` and
the `fresh-clone-integrity.txt` / `mac-*.txt` logs. The following commit records
only handoff checks, not changes to the runtime or historical benchmark results.
