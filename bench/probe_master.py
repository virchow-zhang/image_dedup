#!/usr/bin/env python3
"""直接探查 master 线 v3.0 的行为：CMFD 是否给出区域坐标、误报长什么样。"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO.parent / 'image_dedup_master'
sys.path.insert(0, str(REPO / 'bench'))

from _console import fix_console_encoding  # noqa: E402

fix_console_encoding()

import yaml  # noqa: E402

sys.path.insert(0, str(MASTER))
os.chdir(MASTER)
from core.engine import run_detection  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else 'intra'

cfg = yaml.safe_load((MASTER / 'config.yaml').read_text(encoding='utf-8'))
cfg['report']['clean_output'] = False
cfg['ai']['enabled'] = False

if mode == 'intra':
    cfg['scan']['directory'] = str(REPO / 'bench' / 'data' / 'intra')
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        infos, matches, cmfd = run_detection(cfg['scan']['directory'], cfg)
    print(f'images={len(infos)}  matches={len(matches)}  cmfd={len(cmfd)}')
    print('\n--- CMFD 结果 ---')
    for m in cmfd[:10]:
        print(f'  {Path(m.image1).name}  type={m.match_type}  sim={m.similarity:.3f}  '
              f'region1={m.region1}  region2={m.region2}  inliers={m.inlier_count}')
    if not cmfd:
        print('  （空）')
else:
    d = json.loads((REPO / 'bench' / 'results' / 'v3m.json').read_text(encoding='utf-8'))
    fps = d['cross']['false_positives']
    print(f'--- 跨 case 误报样例（共 {len(fps)}） ---')
    shown = 0
    for pair in fps:
        a, b = pair
        ca = a.split('/')[1]
        cb = b.split('/')[1]
        if ca != cb and shown < 12:
            print(f'  [{ca}] {a.split("/")[-1]}   <->   [{cb}] {b.split("/")[-1]}')
            shown += 1
