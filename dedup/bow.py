"""
ORB 词袋候选层
==============
来源：整合自 master 线 v3.0 的 `core/bow.py` + `core/loader.py::_compute_bow_keys`。

为什么需要它：哈希类候选对「**变换的组合**」无能为力。实测（见 COMPARISON.md）
一张 17° 旋转副本的 pHash 距离是 36/256，而 `裁剪60% + 旋转90°` 这种组合
更是任何单一哈希变体都覆盖不到 —— 把候选阈值从 24 放宽到 64 也只多召回 1.3%。

ORB 描述子天生对旋转与缩放不敏感，于是：
  每张图取 ORB 描述子的前 4 字节当「词」，建倒排索引；
  两张图共享的词桶数 >= min_shared 即成为候选对。
代价是候选更宽（要靠下游投票/NCC/退化检测把关），换来的是组合变换的召回。
"""

from collections import defaultdict
from itertools import combinations
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np


def compute_bow_keys(gray: np.ndarray, max_dim: int = 512,
                     nfeatures: int = 1000) -> np.ndarray:
    """ORB 描述子前 4 字节构成的词集合（去重后的 uint32 数组）""" 
    h, w = gray.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        small = cv2.resize(gray, (int(w * s), int(h * s)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray
    try:
        orb = cv2.ORB_create(nfeatures=nfeatures)
        _, des = orb.detectAndCompute(small, None)
    except Exception:
        return np.zeros(0, np.uint32)
    if des is None or len(des) == 0:
        return np.zeros(0, np.uint32)
    return np.unique(des.view(np.uint32).reshape(-1, 8)[:, 0])


def bow_candidates(keys: Sequence[np.ndarray], min_shared: int = 8,
                   max_bucket: int = 3000) -> Dict[Tuple[int, int], int]:
    """倒排索引 → 共享词数 >= min_shared 的图对。返回 {(i,j): 共享词数}"""
    inv = defaultdict(list)
    for i, ks in enumerate(keys):
        if ks is None:
            continue
        for k in ks.tolist():
            inv[k].append(i)

    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for ids in inv.values():
        m = len(ids)
        # 桶太大说明是"烂大街"的描述子，对区分度没有贡献，直接丢
        if m < 2 or m > max_bucket:
            continue
        for a, b in combinations(ids, 2):
            counts[(a, b) if a < b else (b, a)] += 1

    return {p: c for p, c in counts.items() if c >= min_shared}
