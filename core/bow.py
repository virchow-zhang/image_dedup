"""ORB 词袋 (Bag-of-Words) 候选层。

在加载时为每张图提取 ORB 特征, 用描述子前 4 字节做倒排索引;
两张图共享的桶键数量 >= min_shared 即成为候选对。
ORB 对旋转/缩放/局部裁剪鲁棒 -> 覆盖哈希层抓不到的
自由角度旋转、裁剪、跨图拼接 (局部区域复用)。
"""
from collections import defaultdict
from itertools import combinations


def candidate_pairs(infos, min_shared: int = 8, max_group: int = 3000) -> set:
    inv = defaultdict(list)
    for i, info in enumerate(infos):
        keys = info.bow_keys
        if not keys:
            continue
        for k in keys:
            inv[k].append(i)

    counts = defaultdict(int)
    for ids in inv.values():
        m = len(ids)
        if m < 2 or m > max_group:
            continue
        for a, b in combinations(ids, 2):
            counts[(a, b) if a < b else (b, a)] += 1

    return {p for p, c in counts.items() if c >= min_shared}
