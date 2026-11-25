#!/usr/bin/env python3
"""Verify that the offline learning page matches its source."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "source"))
from build import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", nargs="?", type=Path, default=ROOT / "index.html")
    page = parser.parse_args().html.read_text(encoding="utf-8")
    if page != render():
        raise SystemExit("Learning page differs from its source. Rebuild with source/build.py.")
    markup, payload = page.split('<script type="application/json" id="course-data">')
    data = json.loads(payload.split("</script>", 1)[0])
    chapters = {chapter["id"] for chapter in data["chapters"]}
    for attribute, valid in (("data-go", chapters), ("data-source", data["sources"])):
        missing = set(re.findall(rf'{attribute}="([^"]+)"', markup)) - set(valid)
        if missing:
            raise SystemExit(f"Unresolved {attribute}: {', '.join(sorted(missing))}")
    if re.search(r'<script[^>]+src=|<link[^>]+rel=["\']stylesheet', markup):
        raise SystemExit("Learning page depends on external JavaScript or CSS.")
    print(f'Learning page verified ({len(chapters)} chapters, {len(data["sources"])} embedded sources)')


if __name__ == "__main__":
    main()
