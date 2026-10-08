#!/usr/bin/env python3
"""生成只含「图内复用」卡片的报告，用于快速目视校验查看器/标注层。"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'bench'))
from _console import fix_console_encoding  # noqa: E402
from dedup import report as R  # noqa: E402

fix_console_encoding()

payload = json.loads((REPO / 'bench' / 'results' / 'demo.json').read_text(encoding='utf-8'))
items = []
for entry in payload['intra'][:4]:
    for f in entry['findings']:
        it = R.build_intra_item(entry['figure'], f, None)
        if it:
            items.append(it)

out = REPO / 'bench' / 'results' / 'intra_only.html'
R.generate_report(items, [], str(out), {'directory': 'intra-only demo',
                                        'n_images': len(items), 'elapsed': 0})
print('->', out, len(items), 'cards')
