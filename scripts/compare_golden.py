#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

VOLATILE = {"hot_loop_seconds", "hot_loop_iterations_per_second"}


def stable(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    for key in VOLATILE:
        data.pop(key, None)
    return data


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_golden.py <expected.json> <actual.json>", file=sys.stderr)
        return 2

    expected = stable(sys.argv[1])
    actual = stable(sys.argv[2])
    if actual != expected:
        print("golden validation failed", file=sys.stderr)
        print(json.dumps({"expected": expected, "actual": actual}, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print("golden validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
