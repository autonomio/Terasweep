#!/usr/bin/env python3
"""Build the self-contained learning page from the current package sources."""
from base64 import b64encode
import hashlib
from html import escape
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
GITHUB = "https://github.com/autonomio/Terasweep/blob/main/"
EVIDENCE = {
    "ridge": ("Ridge model", "Code", "src/ridge.rs"),
    "sweep": ("Sweep selection and reuse", "Code", "src/sweep.rs"),
    "plan": ("Sweep plan format", "Guide", "docs/sweeps.md"),
    "audit": ("Model reference audit", "Guide", "docs/research/model-reference-parity.md"),
    "backends": ("Backend setup", "Guide", "docs/backends.md"),
    "research": ("Research record", "Index", "docs/research/README.md"),
}


def render():
    lessons = SOURCE / "lessons"
    index = json.loads((lessons / "index.json").read_text(encoding="utf-8"))
    sources = {}
    for key, (title, kind, path) in EVIDENCE.items():
        content = (REPO / path).read_text(encoding="utf-8")
        sources[key] = {
            "title": title, "kind": kind, "path": path, "url": GITHUB + path,
            "text": content, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    payload = {"chapters": [{"id": chapter} for chapter in index], "sources": sources}
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")

    mark = (SOURCE / "load-terasweep.svg").read_text(encoding="utf-8").strip()
    load_line = f'<p class="course-load"><span class="sr-only">Load Terasweep, 8, 1</span>{mark}</p>'
    nav, sections = [], []
    group = None
    for ordinal, (chapter, lesson) in enumerate(index.items()):
        title, lead = escape(lesson["title"]), escape(lesson["lead"])
        body = (lessons / f"{chapter}.html").read_text(encoding="utf-8")
        if group != lesson["group"]:
            group = lesson["group"]
            nav.append(f'<div class="nav-group">{escape(group)}</div>')
        nav.append(
            f'<a class="nav-item" href="#{chapter}" data-go="{chapter}">'
            f'<span>{title}</span><span class="nav-done" aria-hidden="true"></span></a>'
        )
        sections.append(
            f'<section class="lesson {"active" if ordinal == 0 else ""}" id="{chapter}" '
            f'aria-labelledby="{chapter}-title"><header class="chapter-header">'
            f'{load_line if ordinal == 0 else ""}<h1 id="{chapter}-title" tabindex="-1">{title}</h1>'
            f'<p class="lead">{lead}</p></header><div class="lesson-body">{body}</div></section>'
        )

    font = b64encode((SOURCE / "fonts/finlandica-latin-400-normal.woff2").read_bytes()).decode("ascii")
    page = (SOURCE / "template.html").read_text(encoding="utf-8")
    replacements = {
        "@@FONT_LICENSE@@": (SOURCE / "fonts/finlandica-OFL.txt").read_text(encoding="utf-8"),
        "@@CSS@@": (SOURCE / "theme.css").read_text(encoding="utf-8").replace("@@FONT@@", font),
        "@@DATA@@": data, "@@JS@@": (SOURCE / "app.js").read_text(encoding="utf-8"),
        "@@NAV@@": "".join(nav), "@@CHAPTERS@@": "".join(sections),
        "@@COUNT@@": str(len(index)),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page


if __name__ == "__main__":
    output = SOURCE.parent / "index.html"
    output.write_text(render(), encoding="utf-8")
    print(f"Built {output.name}: {output.stat().st_size:,} bytes")
