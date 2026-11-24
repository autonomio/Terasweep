# Ridge execution levels

The runner now includes scalar factorization controls, compact scoring caches,
Rust AVX-512 intrinsics, and an authored assembly kernel. The ridge mathematics,
original deterministic sampler, cost formulas, rounding, and original-order
winner selection are retained. Other model modules are unchanged.

## Build and execute

On ARM64 macOS, the scalar levels work with the existing Rust 1.63 toolchain.
On x86-64 Linux, use Rust 1.89 or newer. AVX-512 kernels require AVX-512F;
the handwritten kernel also requires BMI1. Unsupported selections fail rather
than silently falling back. The handwritten source uses the Linux System V ABI;
the AVX-512 build has not been ported to x86 macOS or Windows.

```sh
make build
make run LEVEL=compact RUNS=1000000 OUT=runs/compact-1m
# On an AVX-512 x86-64 Linux host:
make run LEVEL=intrinsics RUNS=1000000 OUT=runs/intrinsics-1m
make run LEVEL=assembly RUNS=1000000 OUT=runs/assembly-1m
```

`make` builds `target/terasweep-funnel` and defaults to `LEVEL=compact`.
It does not overwrite the retained historical `target/terasweep` executable.
The bare executable defaults to `score` when `TERASWEEP_LEVEL` is absent.
`make verify LEVEL=...` creates a new verification directory; it no longer
runs `clean` or deletes past measurements. `make run OUT=...` still writes to
its explicitly selected directory; use distinct directories to retain runs.

## Factorization controls

| Level | Work performed |
|---|---|
| `naive` | Rebuild Gram statistics, fit, predict, and score each requested row. |
| `stats` | Reuse Gram statistics; refit, predict, and score each row. |
| `train` | Reuse fitted predictions; recompute the base score per row. |
| `score` | Existing full-payload score cache keyed by model/rule/threshold. |
| `bounds` | Same cache, with shared all-off/all-on results detected from bounds. |
| `compact` | Categorical identities, shared constant payloads, mixed-score storage, scalar scorer. |
| `intrinsics` | Same compact pipeline, using eight-lane AVX-512 intrinsics written in Rust. |
| `assembly` | Same compact pipeline, replacing only that kernel with handwritten assembly. |

`src/optimized/simd.rs` packs and unpacks the same ABI for both kernels.
`src/optimized/score8_gpr.S` is embedded by Rust `global_asm!`; there is no C
implementation, external C library, or separate assembly compiler in this build.
Runtime dispatch checks the required instruction-set features before execution.

## Cache and ordering

Prediction min/max bounds prove whether a requested rule/threshold yields a
constant signal vector. The first occurrence of each constant vector is scored
with the original scalar scorer; later logical keys share its exact payload.
Non-finite prediction vectors retain the scalar path. Mixed results remain
separate. The returned score-cache count remains the number of logical keys.

Requested rows are processed in blocks of at most 1,048,576. Cache state survives
across blocks. Only pending mixed jobs in the current block are grouped by model,
with different rules permitted within a SIMD batch. Summary reduction still runs
in original row order, including rounded ties. No full Cartesian score grid is
precomputed. Dense key indexing is capped at 64 MiB; larger configured address
spaces use a sparse map with bounded initial reservation. Unique cached state
can still grow with the number of distinct keys encountered.

## Validation and instrumentation

```sh
make verify LEVEL=compact
# On the supported x86 host:
make test
TERASWEEP_CHECK_BASE=1 make verify LEVEL=intrinsics
TERASWEEP_CHECK_BASE=1 make verify LEVEL=assembly
```

With `TERASWEEP_CHECK_BASE=1`, every newly computed shortcut or SIMD BaseScore is
compared against the original scalar scorer, including floating-point bit
patterns. This deliberately doubles work and must not be enabled for timing.
Tests use the retained real input matrices and exercise threshold boundaries,
strict/inclusive rules, tails, prefixes, and tradability masks. They require a
supported x86 host to execute the AVX-512 cases.

The original summary timing fields are unchanged. Additional `PHASE` and
`PIPELINE` diagnostics go to stderr (therefore `time.txt` under make).
Whole-process timings, not a sum of mismatched phase medians, define end-to-end
speedup. Plan, mixed-score, and cost/reduction subphase timers exclude model prep.

## Replaying an exact native parameter plan

The optional `TERASWEEP_PLAN` environment variable names a CSV with this header:

```text
alpha,fit_intercept,solver_eps,threshold,fee_bps,slippage_bps,signal_rule
```

The CSV must have exactly the requested row count, and its training/threshold
values must be represented in the supplied configuration. The row order is
preserved. An explicit plan is loaded in memory; ordinary generated sweeps do
not retain all requested rows. Without this variable the original sampler is
used. Do not map sklearn `tol` to `solver_eps`: they are different contracts.

`TERASWEEP_EXPORT_PREDICTIONS` optionally writes trained prediction arrays for
untimed numerical validation. Neither plan replay nor prediction export changes
Limen or replaces its native Ridge implementation.

## Comparison limits

The full native Limen workflow re-prepares raw data, fits sklearn Ridge, computes
its current native metrics, and writes complete round artifacts. Terasweep starts
from frozen prepared matrices and writes its existing smaller summary. Their
backtest mechanics differ. Native-relative ratios therefore describe these
application pipelines, not an isolated language/solver gain or proven economic
parity. Only the within-Rust stages and the original-million-row controls preserve
the same Rust calculation/output contract (cache cardinality differs in ablations).
