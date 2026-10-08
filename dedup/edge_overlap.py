"""
边缘重叠 / 拼接检测
====================
来源：整合自 master 线 v3.0 的 `detectors/edge_overlap.py`。
main 线的 v3 此前没有这一层 —— v2 有过 `check_edge_overlap`，重写时漏掉了。

原理：两张图若是从一张大图上「切开后并排拼」出来的，那么 A 的右边缘带与
B 的左边缘带应当在内容上接得上（同理 A 下 / B 上）。判据取「边缘掩膜的
IoU × 0.6 + 边缘图 NCC × 0.4」，避免只看重合率被大片空白蒙混过去。
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def _edge_overlap_ratio(ea: np.ndarray, eb: np.ndarray) -> float:
    overlap = np.logical_and(ea > 0, eb > 0).sum()
    total = np.logical_or(ea > 0, eb > 0).sum()
    return float(overlap) / float(total) if total else 0.0


def _ncc_edge(ea: np.ndarray, eb: np.ndarray) -> float:
    a = ea.astype(np.float64).ravel()
    b = eb.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def canny_small(gray: np.ndarray, size: int = 256) -> np.ndarray:
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.Canny(g, 50, 150)


def check_edge_overlap(gray1: np.ndarray, gray2: np.ndarray,
                       threshold: float = 0.55, strip: int = 24
                       ) -> Optional[Tuple[str, float, str]]:
    """
    返回 (方向, 得分, 说明) 或 None。
    两张图会被统一缩放到同样的 256×256 边缘图再比，因此要求两者宽高比接近，
    否则「右边缘 vs 左边缘」这种比较没有意义。
    """
    h1, w1 = gray1.shape[:2]
    h2, w2 = gray2.shape[:2]
    ar1, ar2 = w1 / max(h1, 1), w2 / max(h2, 1)
    if max(ar1, ar2) / max(min(ar1, ar2), 1e-6) > 1.25:
        return None

    e1 = canny_small(gray1)
    e2 = canny_small(gray2)

    rs, ls = e1[:, -strip:], e2[:, :strip]
    bs, ts = e1[-strip:, :], e2[:strip, :]

    h_ratio = _edge_overlap_ratio(rs, ls)
    h_score = h_ratio * 0.6 + max(0.0, _ncc_edge(rs, ls)) * 0.4
    v_ratio = _edge_overlap_ratio(bs, ts)
    v_score = v_ratio * 0.6 + max(0.0, _ncc_edge(bs, ts)) * 0.4

    cands = []
    if h_score >= threshold:
        cands.append((h_score, '水平', h_ratio))
    if v_score >= threshold:
        cands.append((v_score, '垂直', v_ratio))
    if not cands:
        return None

    score, direction, ratio = max(cands, key=lambda x: x[0])
    return direction, float(score), f'边缘重合率={ratio:.2f}'
