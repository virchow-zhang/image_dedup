#!/usr/bin/env python3
"""定位跨图配对失败在哪一环：匹配 / RANSAC / 退化 / NCC。"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'bench'))
from _console import fix_console_encoding  # noqa: E402
from dedup import features as F  # noqa: E402

fix_console_encoding()
D = REPO / 'bench' / 'data_master' / 'cross' / 'all'
G = lambda n: cv2.imread(str(D / n), cv2.IMREAD_GRAYSCALE)

CASES = [('base_0000.png', 'rotate90_0000.png'),
         ('base_0000.png', 'rot_free_0000.png'),
         ('base_0000.png', 'crop_60_0000.png'),
         ('crop_60_0000.png', 'rotate90_0000.png'),
         ('base_0000.png', 'exact_copy_0000.png')]

for a, b in CASES:
    ga, gb = G(a), G(b)
    (pa, da, _), (pb, db, _) = F.build_pair_sets(ga, gb, 2000)
    raw = F.ratio_match(pa, da, pb, db)
    print(f'\n=== {a} vs {b} ===')
    print(f'  特征 A={len(pa)} B={len(pb)}   比值检验后配对={len(raw)}')
    if len(raw) < 12:
        print('  → 卡在描述子匹配')
        continue
    off = len(pa)
    coords = np.vstack([pa, pb])
    pairs = [(i, off + j, 0.0) for i, j in raw]
    models = F.iterative_ransac(coords, pairs, 12, max_models=3)
    print(f'  RANSAC 模型数={len(models)}')
    tex = F._texture_mask(ga)
    for k, (M, idx) in enumerate(models):
        sp = np.array([coords[pairs[i][0]] for i in idx], np.float32)
        dp = np.array([coords[pairs[i][1]] for i in idx], np.float32)
        ab = cv2.boundingRect(sp.reshape(-1, 1, 2))
        bb = cv2.boundingRect(dp.reshape(-1, 1, 2))
        dg = F._degenerate_line(sp) or F._degenerate_line(dp)
        ncc = F._pair_ncc(ga, gb, M, ab, tex)
        lab = F.classify_transform(M)[0]
        print(f'    模型{k}: 内点={len(idx):5d} abox={ab} bbox={bb} '
              f'退化={dg} ncc={ncc:.3f} {lab}')
