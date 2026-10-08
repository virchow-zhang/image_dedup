#!/usr/bin/env python3
"""核对 v3m 适配器：master 的 matches 数量 vs 适配后得到的唯一对数量。"""
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

data_dir = (REPO / 'bench' / 'data_master').resolve()
cfg = yaml.safe_load((MASTER / 'config.yaml').read_text(encoding='utf-8'))
cfg['scan']['directory'] = str(data_dir)
cfg['report']['clean_output'] = False
cfg['ai']['enabled'] = False

import contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    infos, matches, cmfd = run_detection(str(data_dir), cfg)

print(f'infos={len(infos)}  matches={len(matches)}  cmfd={len(cmfd)}')
print('matches 里前 3 条的路径形态:')
for m in matches[:3]:
    print('   ', repr(m.image1), '|', repr(m.image2))

dres = data_dir.resolve()


def rel(p):
    try:
        return Path(p).resolve().relative_to(dres).as_posix()
    except ValueError:
        return None


ok, bad, selfpair, uniq = 0, 0, 0, set()
for m in matches:
    a, b = rel(m.image1), rel(m.image2)
    if a is None or b is None:
        bad += 1
        continue
    if a == b:
        selfpair += 1
        continue
    ok += 1
    uniq.add((min(a, b), max(a, b)))
print(f'\n适配结果: 可用 {ok}  路径不可映射 {bad}  同图自对 {selfpair}  唯一对 {len(uniq)}')

# 用 basename 统计（对照 master 自己的 eval）
fn = set()
for m in matches:
    if m.image1 != m.image2:
        fn.add((os.path.basename(m.image1), os.path.basename(m.image2)))
print(f'按 basename 有序对（master eval 口径）: {len(fn)}')

# 看看 uniq 里有多少是真正的同组对
import json
gt = json.loads((REPO / 'bench' / 'data_master' / 'ground_truth.json').read_text(encoding='utf-8'))
pos = {frozenset((p['a'], p['b'])) for p in gt['cross']['positives']}
ign = {frozenset(p) for p in gt['cross'].get('ignore', [])}
pred = {frozenset(p) for p in uniq}
print(f'\n唯一对 {len(pred)}: 正样本 {len(pred & pos)}  ignore {len(pred & ign)}  '
      f'其余(算误报) {len(pred - pos - ign)}')
