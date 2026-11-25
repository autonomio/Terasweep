# Terasweep performance learning

The course explores the current package through the concrete strength examples,
a Ridge regularization lab, and a sweep reuse lab. It covers the validation
split, reference parity, and how to interpret speedups from shared work.
[Research record](../research/README.md)

Open [index.html](index.html) from a checkout in a modern browser or use the
[online course](https://terasweep.autonom.io/). The lessons, labs, and evidence
room work offline. Source links open the repository when connected. Notes are
kept in browser storage when available and can be exported.

## Build and verify

From the repository root, rebuild the self-contained HTML after editing the
lesson sources or the package documents embedded in the evidence room.

```sh
python3 docs/masterclass/source/build.py
python3 docs/masterclass/verify.py
```

The evidence room identifies the embedded source by SHA-256. The
[browser QA record](qa-report.json) covers desktop and mobile layouts, labs,
navigation, evidence, notes, and offline behavior. With Playwright and Chromium
installed, rerun it with `python3 docs/masterclass/test_browser.py` and set
`CHROMIUM_BIN` when using a system browser.
