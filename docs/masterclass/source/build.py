#!/usr/bin/env python3
"""Build the self-contained performance learning page from repository evidence."""
from base64 import b64encode
from html import escape
import json
from pathlib import Path
S=Path(__file__).resolve().parent
D=json.loads((S/'course-data.json').read_text(encoding='utf-8'))
# Keep the archived payload intact. Render the current lesson copy separately.
data=json.dumps(D,separators=(',',':'),ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
lessons = S / 'lessons'
lesson_index = json.loads((lessons / 'index.json').read_text(encoding='utf-8'))
receipt = S.parents[2] / 'benchmarks/sweep-scaling/results-m1-max.json'
models = json.loads(receipt.read_text(encoding='utf-8'))['models']
cards = {'ridge': 'Ridge', 'dlinear': 'DLinear', 'lightgbm': 'LightGBM', 'tide': 'TiDE'}
rows = []
for name, label in cards.items():
    result = models[name]
    billion = next(row for row in result['scaling'] if row['trials'] == 1_000_000_000)
    measured = billion['terasweep']
    rows.append(
        f'<tr data-model="{name}"><th scope="row"><a href="https://github.com/autonomio/Terasweep/blob/main/docs/models/{name}.md#sweep-scaling">{label}</a></th>'
        f'<td>{measured["seconds"]:.2f} s</td><td>{measured["unique_training_configs"]:,}</td>'
        f'<td>{measured["score_cache_entries"]:,}</td><td>{result["projection_anchor_trials"]:,}</td></tr>'
    )
load_mark = (S / 'load-terasweep.svg').read_text(encoding='utf-8').strip()
load_line = f'<p class="course-load"><span class="sr-only">Load Terasweep, 8, 1</span>{load_mark}</p>'
nav = []
sections = []
group = None
for i, chapter in enumerate(D['chapters']):
    chapter_id = chapter['id']
    lesson = lesson_index[chapter_id]
    title = escape(lesson['title'])
    lead = escape(lesson['lead'])
    body = (lessons / f'{chapter_id}.html').read_text(encoding='utf-8')
    if chapter_id == 'scale':
        body = body.replace('@@MATCHED_SWEEP_ROWS@@', ''.join(rows))
    if group != lesson['group']:
        group = lesson['group']
        nav.append(f'<div class="nav-group">{escape(group)}</div>')
    nav.append(
        f'<a class="nav-item" href="#{chapter_id}" data-go="{chapter_id}">'
        f'<span>{title}</span><span class="nav-done" aria-hidden="true"></span></a>'
    )
    sections.append(
        f'<section class="lesson {"active" if i == 0 else ""}" id="{chapter_id}" aria-labelledby="{chapter_id}-title">'
        f'<header class="chapter-header">{load_line if i == 0 else ""}'
        f'<h1 id="{chapter_id}-title" tabindex="-1">{title}</h1>'
        f'<p class="lead">{lead}</p></header><div class="lesson-body">{body}</div></section>'
    )

font = b64encode((S / 'fonts' / 'finlandica-latin-400-normal.woff2').read_bytes()).decode('ascii')
css = (S / 'theme.css').read_text(encoding='utf-8').replace('@@FONT@@', font)
replacements = {
    '@@FONT_LICENSE@@': (S / 'fonts' / 'finlandica-OFL.txt').read_text(encoding='utf-8'),
    '@@CSS@@': css,
    '@@DATA@@': data,
    '@@JS@@': (S / 'app.js').read_text(),
    '@@NAV@@': ''.join(nav),
    '@@CHAPTERS@@': ''.join(sections),
}
page=(S/'template.html').read_text()
for key,val in replacements.items():page=page.replace(key,val)
out=S.parent/'index.html';out.write_text(page,encoding='utf-8');print(f'Built {out.name}: {out.stat().st_size:,} bytes')
