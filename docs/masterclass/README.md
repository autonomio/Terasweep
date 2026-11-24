# Terasweep performance learning

Terasweep reuses calculations across parameter trials that share inputs. The
course presents the original Ridge measurements of that reuse and the subsequent
changes to scoring, cache storage and SIMD execution. A separate matched sweep
compares Ridge, DLinear, LightGBM and TiDE against their uncached references.
[Research notes](../research/README.md)

## Open it

Open [`index.html`](index.html) from a checkout in a modern browser. The lessons
and labs work offline. GitHub links open the public source and benchmark records
when connected.

The HTML file can be copied on its own. Enable JavaScript for the labs. On narrow
screens, use **Chapters** to navigate and scroll wide charts and tables
horizontally. The arrow keys move between lessons when no control has focus.

## Sources and measurements

The original Ridge study is retained in `benchmark-handoff-2026-09-24` at
`2e549d4b782e569a47888e804d2c5fc00f79906b`, and the integrated runtime it
measures is `70525ba2673ffe6d0cff7f94df82bb786446f925`. The course embeds 22
reports and source snapshots, each with its SHA-256 identity and pinned Git
link. The scaling explorer uses the 25 retained Ridge measurements and the
native-reference projections recorded with them.

The four-model table gives measured compact-scalar times, fit counts, score-cache
sizes and the largest measured uncached reference for each model. Its timings
begin after input loading. [Matched sweep method](../../benchmarks/sweep-scaling/README.md) ·
[Measurement receipt](../../benchmarks/sweep-scaling/results-m1-max.json)

Progress and notes are saved only in browser local storage when available. In
restricted/opaque-origin contexts, notes remain in memory and can be exported.
The results exercise checks for the required concepts.

## Rebuild and verify

```sh
cd docs/masterclass
python3 source/build.py
python3 verify.py
```

The current copy is in `source/lessons/`. The build retains the original lesson
payload and reads the billion-trial rows from the matched sweep receipt.
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
