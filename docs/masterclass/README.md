# Terasweep performance learning

An offline, interactive course on factorization, compact score caches, and
benchmark design. It follows the original Ridge investigation through Rust
AVX-512 intrinsics and handwritten assembly, then examines matched sweeps for
Ridge, DLinear, LightGBM, and TiDE. [Research notes](../research/README.md)

## Open it

Open [`index.html`](index.html) from a checkout in a modern browser. The lessons
and labs work offline. GitHub links open the public source and benchmark records
when connected.

The single HTML file is self-contained. It can be copied or shared independently
of this directory. Enable JavaScript for the labs; the prose remains readable
without it. On narrow screens, use the menu button and horizontally scroll dense
charts/tables. Arrow keys navigate lessons when focus is not inside a control.

## The experience

Thirteen chapters cover the inherited design, the uncached sklearn-backed Limen
SFD, ridge sufficient statistics, sweep dependencies, factored cost/Sharpe
arithmetic, profiling, constant-signal bounds, cache memory layout, independent
SIMD lanes, hand-authored instruction scheduling, validation, the reverse funnel,
and measured 100k–1b Ridge scaling runs. The scaling chapter compares that
historical experiment with the four matched scalar sweeps on Apple M1 Max.
The capstone asks the learner to write a claim using the correct reference and
timing boundary.

Interactive elements include regularization and dependency labs, a 12-bar cost
lab, a strict/inclusive threshold lab, bounded-buffer calculations, an animated
eight-lane timeline, an annotated assembly source walk, claim classification,
measured-result explorers, and an Amdahl calculator. Checkpoints, a glossary,
search, theme switching, field notes and learning-record export are included.

## Evidence contract

- The original Ridge study is `benchmark-handoff-2026-09-24` at commit
  `2e549d4b782e569a47888e804d2c5fc00f79906b`.
- The measured integrated runtime is commit
  `70525ba2673ffe6d0cff7f94df82bb786446f925`.
- Twenty-two reports/code snapshots are embedded with SHA-256 identities and
  pinned Git links. The evidence room works without fetching those links.
- All 25 historical Ridge scaling measurements come directly from the preserved
  data. The historical native-reference estimates are projections.
- The initial six-case study, 1,014-row grid, standalone kernel studies, integrated
  Intel runs and AMD scaling runs remain separate.
- Full native CLI projections use a different calibration workload. The UI
  labels this explicitly. Native and Terasweep backtest conventions differ.
- The four-model table comes from the [matched sweep receipt](../../benchmarks/sweep-scaling/results-m1-max.json).
  It shows measured compact-scalar times, fit and score-cache sizes, and the
  largest measured reference for each model. It uses the timing boundary
  documented in the [matched sweep method](../../benchmarks/sweep-scaling/README.md).
- Browser labs mark their illustrative fixtures.
- Historical reports retain their original deployment status and limitations;
  the handoff source records the later completion.

Progress and notes are saved only in browser local storage when available. In
restricted/opaque-origin contexts, notes remain in memory and can be exported.
The capstone checklist looks for the required concepts.

## Rebuild and verify

```sh
cd docs/masterclass
python3 source/build.py
python3 verify.py
```

`source/` contains the application, styles, template, original lesson payload,
and the current lesson updates. The build reads the matched sweep receipt and
embeds its billion-trial rows.
[Source manifest](source-manifest.json) records the evidence identities.
[Browser QA](qa-report.json) covers every chapter at desktop and mobile widths,
the matched sweep table, the original charts, labs, glossary, notes, and exports.
The checks render the offline page in Chromium and exercise in-session notes.

The repository's `scripts/verify_evidence.py` verifies the preserved research
archives and benchmark files.

Optional browser QA (requires Playwright and Chromium):

```sh
CHROMIUM_BIN=/path/to/chromium python3 test_browser.py
```

The QA script writes ignored screenshots and test exports into `qa-output/`.
Its JSON summary is retained in `qa-report.json`. Browser binaries and Python
dependencies are not bundled with the learning page.
