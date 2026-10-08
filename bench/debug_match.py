#!/usr/bin/env python3
"""调试图内匹配：打印关键点/配对/RANSAC 中间量。"""
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

for name in sys.argv[1:] or ['intra/fig_000.png', 'intra/fig_002.png']:
    path = REPO / 'bench' / 'data' / name
    gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    coords, desc, origin = F.build_match_set(gray, 4000)
    print(f'\n=== {name}  {w}x{h} ===')
    print(f'  关键点总数={len(coords)}  原图={int((origin==0).sum())} 镜像={int((origin==1).sum())}')

    min_sep = 0.06 * min(h, w)
    pairs = F.self_match(coords, desc, min_sep)
    print(f'  min_sep={min_sep:.1f}  比值检验后配对={len(pairs)}')
    if pairs:
        d = [np.hypot(coords[j][0]-coords[i][0], coords[j][1]-coords[i][1]) for i, j, _ in pairs]
        print(f'  配对空间距离: min={min(d):.1f} med={np.median(d):.1f} max={max(d):.1f}')
        both = sum(1 for i, j, _ in pairs if origin[i] == 0 and origin[j] == 0)
        cross = sum(1 for i, j, _ in pairs if origin[i] != origin[j])
        print(f'  同源配对(都在原图)={both}  跨集(原图<->镜像)={cross}')

    models = F.iterative_ransac(coords, pairs, 14)
    print(f'  RANSAC 模型数={len(models)}')
    for k, (M, idx) in enumerate(models):
        sp = np.array([coords[pairs[i][0]] for i in idx])
        dp = np.array([coords[pairs[i][1]] for i in idx])
        print(f'    模型{k}: 内点={len(idx)}  M={np.round(M,3).tolist()}')
        print(f'      src bbox={cv2.boundingRect(sp.reshape(-1,1,2))}  '
              f'dst bbox={cv2.boundingRect(dp.reshape(-1,1,2))}')
        print(f'      平均位移={np.mean(np.linalg.norm(dp-sp,axis=1)):.1f}px')

    found = F.find_region_matches(gray)
    print(f'  最终输出={len(found)}')
    for f in found:
        print(f'    {f.transform} ncc={f.ncc} A={f.a_bbox} B={f.b_bbox}')
