"""
子图 / 裁剪检测
================
来源：整合自 master 线 v3.0 的 `detectors/subimage.py`（同一作者的另一条开发线）。
main 线的 v3 此前没有这一层，导致「整图哈希」的天然盲区 —— 裁剪副本召回为 0。

原理：面积比先筛出「一大一小」的候选对，再做三级金字塔（256→512→1024）
由粗到精的 `cv2.matchTemplate`。用 TM_CCOEFF_NORMED（零均值归一化）所以
对亮度/对比度调整不敏感；每级只在前一级最优点附近开窗搜索，把 O(W·H·S)
的全图滑窗压到毫秒级。
"""

import cv2
import numpy as np

LEVELS = (256, 512, 1024)
SCALES = (1.0, 0.85, 0.7, 0.55)


def read_gray_capped(path: str, max_dim: int = 1024):
    """读灰度并把最长边压到 max_dim 以内"""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.draft('L', (max_dim, max_dim))
            g = np.asarray(im.convert('L'))
    except Exception:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return None
    if max(g.shape) > max_dim:
        f = max_dim / max(g.shape)
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(g)


def _coarse_to_fine(big: np.ndarray, small: np.ndarray, early_floor: float = 0.0):
    """
    由粗到精金字塔模板匹配，返回 (最佳得分, 最佳位置, 缩放比)。

    early_floor > 0 时，第一级（最粗）跑完若最佳得分仍低于该下限，
    直接放弃 —— 真裁剪在最粗一级就已经能拿到较高相关，非裁剪对则
    在 4 次小图 matchTemplate 后就被淘汰，省掉后面两级 8 次大图匹配。
    """
    best_score, best_pos, best_scale = -1.0, None, 1.0
    prev_center = prev_radius = prev_shape = None

    for level in LEVELS:
        f = level / max(big.shape)
        if f >= 1.0:
            big_l, f = big, 1.0
        else:
            big_l = cv2.resize(big, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        bh_l, bw_l = big_l.shape[:2]

        for sc in SCALES:
            tw, th = int(small.shape[1] * f * sc), int(small.shape[0] * f * sc)
            if tw < 24 or th < 24 or tw >= bw_l or th >= bh_l:
                continue
            tmpl = cv2.resize(small, (tw, th), interpolation=cv2.INTER_AREA)
            tf = tmpl.astype(np.float32)
            tn = (tf - tf.mean()) / (tf.std() + 1e-8)

            if prev_center is None:
                res = cv2.matchTemplate(big_l.astype(np.float32), tn,
                                        cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
            else:
                cx = prev_center[0] * bw_l / prev_shape[1]
                cy = prev_center[1] * bh_l / prev_shape[0]
                r = int(max(16, prev_radius * bw_l / prev_shape[1]))
                x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
                x1, y1 = min(bw_l, int(cx + r) + tw), min(bh_l, int(cy + r) + th)
                if x1 - x0 < tw or y1 - y0 < th:
                    x0, y0, x1, y1 = 0, 0, bw_l, bh_l
                win = big_l[y0:y1, x0:x1].astype(np.float32)
                res = cv2.matchTemplate(win, tn, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                max_loc = (max_loc[0] + x0, max_loc[1] + y0)

            if max_val > best_score:
                best_score = float(max_val)
                best_pos = (int(max_loc[0] / f), int(max_loc[1] / f))
                best_scale = sc
                prev_center = (max_loc[0], max_loc[1])
                prev_radius = max(bw_l, bh_l) // 4
                prev_shape = (bh_l, bw_l)

        if early_floor > 0 and level == LEVELS[0] and best_score < early_floor:
            break

    return best_score, best_pos, best_scale


def check_subimage_arrays(big: np.ndarray, small: np.ndarray,
                          threshold: float = 0.85):
    """已加载灰度图版本的裁剪检测（避免每个候选对都重复读盘）"""
    if big is None or small is None:
        return None
    bh, bw = big.shape[:2]
    sh, sw = small.shape[:2]

    # 面积太接近 → 不是裁剪关系；小图反而更大 → 尺寸异常
    if sh * sw > bh * bw * 0.85:
        return None
    if min(sh, sw) < 24 or min(bh, bw) < 64:
        return None
    if sw >= bw or sh >= bh:
        return None

    try:
        score, pos, scale = _coarse_to_fine(big, small,
                                            early_floor=max(0.0, threshold - 0.25))
    except Exception:
        return None
    if score < threshold or pos is None:
        return None

    tw = int(sw * scale)
    th = int(sh * scale)
    return {'score': float(score), 'pos': (int(pos[0]), int(pos[1])),
            'scale': float(scale), 'box': (int(pos[0]), int(pos[1]), tw, th)}


def check_subimage(big_path: str, small_path: str, threshold: float = 0.85,
                   max_dim: int = 1024):
    """small 是否为 big 的裁剪副本。返回 None 或 dict(score, pos, scale, box)。"""
    return check_subimage_arrays(read_gray_capped(big_path, max_dim),
                                 read_gray_capped(small_path, max_dim), threshold)


class SubimageCache:
    """
    按路径缓存 subimage 用的灰度图。

    不做缓存时，每个候选对都要重新读两张盘 + 缩放，140 张图的基准上
    检测耗时从 0.04s 飙到 22.5s —— 瓶颈根本不在模板匹配，而在重复 IO。
    """

    def __init__(self, max_dim: int = 1024):
        self.max_dim = max_dim
        self._cache = {}

    def get(self, path: str):
        if path not in self._cache:
            self._cache[path] = read_gray_capped(path, self.max_dim)
        return self._cache[path]


def subimage_candidate_pairs(sizes, min_ratio: float = 1.5, max_ratio: float = 4.0,
                             max_pairs: int = 2000):
    """
    从 (w, h) 尺寸数组里筛出可能的「大图包含小图」候选对。
    返回 [(i, j, big_idx, small_idx), ...]，i<j，按面积比从大到小排序。

    min_ratio=1.5 对应「裁剪掉约 18% 边长」；再小的面积差更可能是
    不同倍率拍摄而非裁剪，把它们放进模板匹配既费时又徒增误报
    （实测 1.2 时 120 张图会产生 2169 个候选，1.5 时降到 687）。
    """
    n = len(sizes)
    if n < 2:
        return []
    w = np.array([s[0] for s in sizes], np.float64)
    h = np.array([s[1] for s in sizes], np.float64)
    area = w * h
    ar = w / np.maximum(h, 1)

    big = area[:, None] >= area[None, :]          # big[a,b] = a 比 b 大
    ratio = np.maximum(area[:, None], area[None, :]) / \
        np.maximum(np.minimum(area[:, None], area[None, :]), 1)
    ar_ratio = np.maximum(ar[:, None], ar[None, :]) / \
        np.maximum(np.minimum(ar[:, None], ar[None, :]), 1e-6)

    iu = np.triu(np.ones((n, n), bool), k=1)
    ok = iu & (ratio <= max_ratio) & (ratio >= min_ratio) & (ar_ratio <= 2.0)

    rows, cols = np.where(ok)
    order = np.argsort(-ratio[rows, cols])        # 面积比大的优先
    out = []
    for k in order:
        a, b = int(rows[k]), int(cols[k])
        big_i, small_i = (a, b) if big[a, b] else (b, a)
        out.append((min(a, b), max(a, b), big_i, small_i))
        if len(out) >= max_pairs:
            break
    return out
