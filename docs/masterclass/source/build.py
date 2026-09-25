#!/usr/bin/env python3
"""Build the single-file masterclass. Standard library only; no network calls."""
import json
from pathlib import Path
S=Path(__file__).resolve().parent
D=json.loads((S/'course-data.json').read_text(encoding='utf-8'))
labels=['The investigation','Protect the reference','Factor the model','Factor the sweep',
        'Factor the scorer','Follow the profile','Redesign the cache','Choose the SIMD axis',
        'Write the instructions','Prove the claim','The reverse funnel','Scale to one billion','Reproduce & teach']
nav=[];group=None
for i,c in enumerate(D['chapters']):
    if group!=c['group']:
        group=c['group'];nav.append(f'<div class="nav-group">{group}</div>')
    nav.append(f'<button class="nav-item" data-go="{c["id"]}"><span class="nav-no">{i:02d}</span><span>{labels[i]}</span><span class="nav-done" aria-hidden="true"></span></button>')
sections=''.join(f'<section class="lesson {"active" if i==0 else ""}" id="{c["id"]}" aria-labelledby="{c["id"]}-title"><header class="chapter-header"><div class="eyebrow">{c["kicker"]}</div><h1 id="{c["id"]}-title" tabindex="-1">{c["title"]}</h1><p class="lead">{c["lead"]}</p></header><div class="lesson-body">{c["body"]}</div></section>' for i,c in enumerate(D['chapters']))
data=json.dumps(D,separators=(',',':'),ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
replacements={'@@CSS@@':(S/'theme.css').read_text(),'@@DATA@@':data,'@@JS@@':(S/'app.js').read_text(),'@@NAV@@':''.join(nav),'@@CHAPTERS@@':sections}
page=(S/'template.html').read_text()
for key,val in replacements.items():page=page.replace(key,val)
out=S.parent/'index.html';out.write_text(page,encoding='utf-8');print(f'Built {out.name}: {out.stat().st_size:,} bytes')
