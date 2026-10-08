#!/usr/bin/env python3
"""隔离诊断：直接对两个已知 panel 做 SIFT 互匹配，看理论上应有多少配对。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'bench'))
from _console import fix_console_encoding  # noqa: E402

fix_console_encoding()

data = REPO / 'bench' / 'data'
gt = json.loads((data / 'ground_truth.json').read_text(encoding='utf-8'))
by_file = {f['file']: f for f in gt['intra']['figures']}

sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.008, edgeThreshold=16, sigma=1.2)
bf = cv2.BFMatcher()

for name in ['intra/fig_000.png', 'intra/fig_002.png', 'intra/fig_007.png']:
    fig = by_file[name]
    img = cv2.imread(str(data / name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f'\n=== {name}  relation={fig["relation"]} ===')

    # 每个 panel 的 SIFT 特征数
    for p in fig['panels']:
        x, y, w, h = p['bbox']
        sub = gray[y:y + h, x:x + w]
        kp, de = sift.detectAndCompute(sub, None)
        print(f"  panel {p['id']}  {w}x{h}  kp={0 if kp is None else len(kp)}")

    # 真值里的两个区域直接互匹配
    for g in fig['reuse']:
        ax, ay, aw, ah = g['a_bbox']
        bx, by, bw, bh = g['b_bbox']
        ca = gray[ay:ay + ah, ax:ax + aw]
        cb = gray[by:by + bh, bx:bx + bw]
        ka, da = sift.detectAndCompute(ca, None)
        kb, db = sift.detectAndCompute(cb, None)
        if da is None or db is None:
            print(f'  [{g["relation"]}] 特征为空')
            continue
        ms = bf.knnMatch(da, db, k=2)
        good = [m for m, n in ms if m.distance < 0.8 * n.distance]
        dists = sorted(m.distance for m in good)
        print(f'  [{g["relation"]}] A kp={len(ka)} B kp={len(kb)} '
              f'比值检验通过={len(good)}  最近距离={dists[:5]}')
