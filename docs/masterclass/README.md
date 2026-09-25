# Terasweep: make the work disappear

An offline, interactive masterclass based on the preserved ridge performance
investigation, from the native Limen reference to factorization, compact caches,
Rust AVX-512 intrinsics, handwritten assembly, scaling and conclusive Git handoff.

## Open it

Open `index.html` in a modern browser. No installation, server, account, external
font, CDN, analytics, API key or network service is required. Private GitHub source
links are optional and require your existing repository access.

The single HTML file is self-contained. It can be copied or shared independently
of this directory. Enable JavaScript for the labs; the prose remains readable
without it. On narrow screens, use the menu button and horizontally scroll dense
charts/tables. Arrow keys navigate lessons when focus is not inside a control.

## The experience

Thirteen chapters cover the inherited design, the uncached sklearn-backed Limen
SFD, ridge sufficient statistics, sweep dependencies, factored cost/Sharpe
arithmetic, profiling, constant-signal bounds, cache memory layout, independent
SIMD lanes, hand-authored instruction scheduling, validation, the reverse funnel,
and actual 100k–1b scaling runs. A final capstone asks the learner to state a
claim with its correct denominator and limitations.

Interactive elements include regularization and dependency labs, a 12-bar cost
lab, a strict/inclusive threshold lab, bounded-buffer calculations, an animated
eight-lane timeline, an annotated assembly source walk, claim classification,
measured-result explorers, and an Amdahl calculator. Checkpoints, a glossary,
search, theme switching, field notes and learning-record export are included.

## Evidence contract

- This is a teaching layer, not a new performance experiment.
- The basis is `benchmark-handoff-2026-09-24` at commit
  `2e549d4b782e569a47888e804d2c5fc00f79906b`.
- The measured integrated runtime is commit
  `70525ba2673ffe6d0cff7f94df82bb786446f925`.
- Twenty-two reports/code snapshots are embedded with SHA-256 identities and
  pinned Git links. The evidence room works without fetching those links.
- All 25 recorded scaling cells are loaded from the preserved data, not refitted
  or smoothed. Only the original native-reference projections are modeled.
- The initial six-case study, 1,014-row grid, standalone kernel studies, integrated
  Intel runs and AMD scaling runs remain separate.
- Full native CLI projections use a different calibration workload. The UI
  labels this explicitly. Native and Terasweep backtest conventions differ.
- Teaching fixtures are explicitly marked and are not benchmark observations.
- Historical reports retain their original deployment status and limitations;
  the handoff source records the later completion.

Progress and notes are saved only in browser local storage when available. In
restricted/opaque-origin contexts, notes remain in memory and can be exported.
The capstone review is a keyword checklist, not an automated factual grader.

## Rebuild and verify

```sh
python3 source/build.py
python3 verify.py
```

`source/` contains the readable application, styles, template and frozen lesson
payload. `source-manifest.json` identifies its evidence. `qa-report.json` records
76 browser checks covering initialization, every chapter at desktop/mobile
widths, all labs, chart controls, evidence, glossary, notes and exports. QA used
Chromium via Playwright and in-memory document rendering because local URL
navigation was restricted in the execution environment. It did not rerun any
benchmark or verify persistence across browser restarts.

The surrounding repository's `scripts/verify_evidence.py` verifies the original
research archives separately. Do not replace those archived measurements when
editing this teaching application.

Optional browser QA (requires Playwright and Chromium):

```sh
CHROMIUM_BIN=/path/to/chromium python3 test_browser.py
```

The QA script writes ignored screenshots and test exports into `qa-output/`.
Its JSON summary is retained in `qa-report.json`. Browser binaries and Python
dependencies are not bundled with the learning page.
