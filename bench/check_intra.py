#!/usr/bin/env python3
"""验证图内区域匹配：对 intra 数据集逐图跑检测并与真值比对。"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'bench'))
from _console import fix_console_encoding  # noqa: E402
from dedup.features import find_region_matches  # noqa: E402

fix_console_encoding()

data = REPO / 'bench' / 'data'
gt = json.loads((data / 'ground_truth.json').read_text(encoding='utf-8'))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


T = 0.5
tp = fp = fn = 0
ious = []
t_all = 0.0

for fig in gt['intra']['figures']:
    img = cv2.imread(str(data / fig['file']))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    found = find_region_matches(gray)
    dt = time.perf_counter() - t0
    t_all += dt

    gts = fig['reuse']
    used = [False] * len(found)
    hits = 0
    for g in gts:
        for i, f in enumerate(found):
            if used[i]:
                continue
            same = iou(f.a_bbox, g['a_bbox']) >= T and iou(f.b_bbox, g['b_bbox']) >= T
            swap = iou(f.a_bbox, g['b_bbox']) >= T and iou(f.b_bbox, g['a_bbox']) >= T
            if same or swap:
                used[i] = True
                hits += 1
                ious.append((max(iou(f.a_bbox, g['a_bbox']), iou(f.a_bbox, g['b_bbox'])) +
                             max(iou(f.b_bbox, g['b_bbox']), iou(f.b_bbox, g['a_bbox']))) / 2)
                break
    tp += hits
    fn += len(gts) - hits
    fp += sum(1 for u in used if not u)

    mark = 'OK ' if hits == len(gts) and sum(used) == len(gts) else 'MISS'
    print(f"{fig['file']:20s} {fig['relation']:24s} gt={len(gts)} found={len(found)} "
          f"hit={hits} {mark} {dt*1000:.0f}ms")
    for f in found:
        print(f"      -> {f.transform:16s} ncc={f.ncc:.3f} inl={f.n_inliers:3d} "
              f"A={f.a_bbox} B={f.b_bbox}")

p = tp / (tp + fp) if tp + fp else 0
r = tp / (tp + fn) if tp + fn else 0
f1 = 2 * p * r / (p + r) if p + r else 0
print(f'\nregion P={p:.3f} R={r:.3f} F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})')
print(f'mean IoU={np.mean(ious):.3f}' if ious else 'mean IoU=n/a')
print(f'total {t_all:.2f}s  ({t_all/len(gt["intra"]["figures"])*1000:.0f} ms/图)')
