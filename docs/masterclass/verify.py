#!/usr/bin/env python3
"""Verify the offline course payload, citations, and recorded scaling data."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def read_payload(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="course-data">(.*?)</script>', text, re.S)
    if not match:
        raise ValueError("Missing embedded course data")
    return text, json.loads(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", nargs="?", type=Path, default=Path(__file__).with_name("index.html"))
    args = parser.parse_args()
    text, data = read_payload(args.html)
    assert len(data["chapters"]) == 13, "Unexpected chapter count"
    assert len({c["id"] for c in data["chapters"]}) == 13, "Duplicate chapter id"
    for key, src in data["sources"].items():
        digest = hashlib.sha256(src["text"].encode("utf-8")).hexdigest()
        assert digest == src["sha256"], f"Embedded source changed: {key}"
    for ref in re.findall(r'data-source="([^"]+)"', text.split('<script type="application/json"')[0]):
        assert ref in data["sources"], f"Unresolved source button: {ref}"
    course_root = Path(__file__).resolve().parent
    manifest = json.loads((course_root / "source-manifest.json").read_text())
    receipt = manifest["matched_sweep_receipt"]
    receipt_bytes = (course_root.parents[1] / receipt["path"]).read_bytes()
    assert hashlib.sha256(receipt_bytes).hexdigest() == receipt["sha256"], "Matched receipt changed"
    models = json.loads(receipt_bytes)["models"]
    scale = next(chapter["body"] for chapter in data["chapters"] if chapter["id"] == "scale")
    assert set(re.findall(r'<tr data-model="([^"]+)">', scale)) == models.keys()
    for model, record in models.items():
        measured = next(
            row["terasweep"] for row in record["scaling"] if row["trials"] == 1_000_000_000
        )
        cells = (
            f'<td>{measured["seconds"]:.2f} s</td>'
            f'<td>{measured["unique_training_configs"]:,}</td>'
            f'<td>{measured["score_cache_entries"]:,}</td>'
            f'<td>{record["projection_anchor_trials"]:,}</td>'
        )
        row = re.search(rf'<tr data-model="{model}">.*?</tr>', scale)
        assert row and cells in row.group(), f"Wrong matched sweep data: {model}"
    rows = data["scaling"]["rows"]
    assert len(rows) == 25
    assert len({(r["n"], r["level"]) for r in rows}) == 25
    assert sum(r["n"] for r in rows) == 5_555_500_000
    for row in rows:
        assert row["rows"] == row["valid_rows"] + row["failed_rows"] == row["n"]
        assert row["wall_seconds"] > 0 and row["stable_output_match"]
        for key, model in data["scaling"]["native_models"].items():
            estimate = model["startup_seconds"] + model["seconds_per_row"] * row["n"]
            assert abs(estimate-row["native_estimates_seconds"][key]) < max(1,estimate)*1e-10
            ratio = estimate / row["wall_seconds"]
            assert abs(ratio-row["multipliers_vs_projected_native"][key]) < max(1,ratio)*1e-10
    assert not re.search(r'<script[^>]+src=', text), "External JavaScript dependency"
    assert not re.search(r'<link[^>]+rel=["\']stylesheet', text), "External CSS dependency"
    assert not re.search(r'\bfetch\s*\(', text.split('</script><script>')[-1]), "Unexpected runtime fetch"
    print(json.dumps({"status":"PASS", "chapters":13,"embedded_sources":len(data["sources"]),
                      "matched_model_sweeps":len(models),
                      "recorded_scaling_rows":25,"recorded_sample_evaluations":5_555_500_000,
                      "new_benchmarks_run":False,"html_sha256":hashlib.sha256(args.html.read_bytes()).hexdigest()},indent=2))


if __name__ == "__main__":
    main()
