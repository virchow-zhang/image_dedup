#!/usr/bin/env python3
"""验证 panel 切分：对 intra 数据集跑切分，统计正确率并输出可视化。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'bench'))
from _console import fix_console_encoding  # noqa: E402
from dedup.panels import draw_panels, segment_panels  # noqa: E402

fix_console_encoding()

data = REPO / 'bench' / 'data'
gt = json.loads((data / 'ground_truth.json').read_text(encoding='utf-8'))

outdir = REPO / 'bench' / 'results' / 'panel_vis'
outdir.mkdir(parents=True, exist_ok=True)

ok = 0
tiles = []
for fig in gt['intra']['figures']:
    path = data / fig['file']
    img = cv2.imread(str(path))
    panels = segment_panels(img)
    expect = fig['n_panels']
    hit = len(panels) == expect
    ok += hit
    print(f"{fig['file']:20s} expect={expect} got={len(panels)} "
          f"{'OK ' if hit else 'MISS'} boxes={[p.bbox for p in panels]}")
    if len(tiles) < 4:
        vis = draw_panels(img, panels)
        tiles.append(cv2.resize(vis, (498, 249)))

print(f'\n切分正确 {ok}/{len(gt["intra"]["figures"])}')

if tiles:
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]) if len(tiles) >= 4 else tiles[0]
    cv2.imwrite(str(outdir / 'panel_sheet.png'), sheet)
    print('可视化 ->', outdir / 'panel_sheet.png')
