#!/usr/bin/env python3
"""对比「真同源」与「误报」在特征级判定各指标上的分布，找可分的判据。"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'bench'))
from _console import fix_console_encoding  # noqa: E402
from dedup.features import extract_pair_features, match_features  # noqa: E402

fix_console_encoding()

MA = REPO / 'bench' / 'data_master' / 'cross' / 'all'
MY = REPO / 'bench' / 'data'


def run(pa, pb, tag):
    ga = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
    gb = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
    fa, fb = extract_pair_features(ga), extract_pair_features(gb)
    ms = match_features(fa, fb, min_inliers=12, ncc_threshold=0.0)
    if not ms:
        print(f'  {tag:28s} 无匹配')
        return
    m = max(ms, key=lambda r: r.score)
    ax, ay, aw, ah = m.a_bbox
    area = max(1, aw * ah)
    dens = m.n_inliers / area * 100
    print(f'  {tag:28s} inl={m.n_inliers:5d} ncc={m.ncc:.3f} '
          f'cov={m.meta.get("coverage"):.3f} 区域={aw}x{ah} 内点密度={dens:.2f}%  {m.transform}')


print('=== 真同源（master 基准） ===')
for a, b in [('base_0000.png', 'rotate90_0000.png'),
             ('base_0000.png', 'rot_free_0000.png'),
             ('base_0000.png', 'crop_60_0000.png'),
             ('crop_60_0000.png', 'rotate90_0000.png'),
             ('flip_h_0000.png', 'rot_free_0000.png'),
             ('base_0000.png', 'splice_0000.png')]:
    run(MA / a, MA / b, f'{a[:16]} vs {b[:16]}')

print('\n=== 误报（我的基准，同风格不同图） ===')
for a, b in [('cross/case_022/barplot_cond2_3.jpg', 'cross/case_022/barplot_cond2_5.jpg'),
             ('cross/case_010/barplot_cond0_4.jpg', 'cross/case_022/barplot_cond2_5.jpg')]:
    run(MY / a, MY / b, f'{Path(a).name[:16]} vs {Path(b).name[:16]}')

print('\n=== 真同源（我的基准） ===')
for a, b in [('cross/case_000/blot_cond0_1.jpg', 'cross/case_000/blot_cond0_1_rep.jpg'),
             ('cross/case_009/gel_cond4_1.jpg', 'cross/case_009/gel_cond4_2.jpg')]:
    run(MY / a, MY / b, f'{Path(a).name[:16]} vs {Path(b).name[:16]}')
