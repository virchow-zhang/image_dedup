#!/usr/bin/env python3
"""验证跨图特征级配对：叠加变换能不能找到、难负样本会不会误报。"""
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'bench'))
from _console import fix_console_encoding  # noqa: E402
from dedup.features import find_pair_matches  # noqa: E402

fix_console_encoding()

D = REPO / 'bench' / 'data_master' / 'cross' / 'all'


def G(name):
    return cv2.imread(str(D / name), cv2.IMREAD_GRAYSCALE)


CASES = [
    # (A, B, 期望能找到同源)
    ('base_0000.png', 'exact_copy_0000.png', True),
    ('base_0000.png', 'rot_free_0000.png', True),
    ('base_0000.png', 'crop_60_0000.png', True),
    ('base_0000.png', 'rotate90_0000.png', True),
    ('base_0000.png', 'splice_0000.png', True),
    ('crop_60_0000.png', 'rotate90_0000.png', True),
    ('rotate90_0000.png', 'flip_h_0000.png', True),
    ('flip_h_0000.png', 'rot_free_0000.png', True),
    ('brightness_70_0000.png', 'copy_move_0000.png', True),
    # 负样本：不同基底，长得像但不是同源
    ('base_0000.png', 'base_0001.png', False),
    ('base_0000.png', 'base_0002.png', False),
    ('rotate90_0000.png', 'rotate90_0001.png', False),
    ('crop_60_0000.png', 'crop_60_0003.png', False),
]

tp = fp = fn = tn = 0
for a, b, expect in CASES:
    ga, gb = G(a), G(b)
    t0 = time.perf_counter()
    ms = find_pair_matches(ga, gb)
    dt = (time.perf_counter() - t0) * 1000
    got = len(ms) > 0
    ok = '✓' if got == expect else '✗'
    if expect and got:
        tp += 1
    elif expect and not got:
        fn += 1
    elif not expect and got:
        fp += 1
    else:
        tn += 1
    desc = ''
    if ms:
        m = ms[0]
        desc = (f"{m.transform} ncc={m.ncc:.3f} inl={m.n_inliers} "
                f"覆盖={m.meta.get('coverage')} A={m.a_bbox} B={m.b_bbox}")
    print(f"{ok} {'应找到' if expect else '应无  '} {a[:22]:24s} vs {b[:22]:24s} "
          f"{dt:6.0f}ms  {desc}")

print(f"\nTP={tp} FN={fn} FP={fp} TN={tn}")
