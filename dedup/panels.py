"""
组图 panel 切分
================
把一张拼好的 SCI 组图（多 panel 网格）切成若干子图区域。

算法：递归 XY-cut + 「走廊」检测
  走廊（gutter）判据 = 该行/列同时满足
    (a) 边缘密度≈0     —— Sobel 梯度超过阈值的像素占比极低
    (b) 灰度标准差≈0   —— 整行/列近似单色
    (c) 与两侧内容有对比 —— 走廊色必须明显区别于它要分开的内容
  条件 (c) 是关键：否则一块纯色 panel 内部的空白行会被误判为走廊，
  导致把单个 panel 越切越碎。

早期版本用「全局直方图众数」当背景色，实测失败：荧光图里暗色 panel
面积远大于留白，众数取到 panel 底色而非留白；blot 图底色(≈240)与
留白(255)差 15，超过容差也被当成内容。改用局部均匀性后与背景色无关。
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Panel:
    x: int
    y: int
    w: int
    h: int
    index: int = -1

    @property
    def bbox(self) -> List[int]:
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y:self.y + self.h, self.x:self.x + self.w]


def _gutter_flags(gray: np.ndarray, edge_tol: float = 18.0,
                  std_tol: float = 4.0, edge_frac: float = 0.012):
    """逐行/逐列的走廊标记"""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge = ((np.abs(gx) + np.abs(gy)) > edge_tol).astype(np.float32)

    row_edge, col_edge = edge.mean(axis=1), edge.mean(axis=0)
    row_std, col_std = g.std(axis=1), g.std(axis=0)
    return ((row_edge < edge_frac) & (row_std < std_tol),
            (col_edge < edge_frac) & (col_std < std_tol))


def _runs(flags: np.ndarray) -> List[Tuple[int, int]]:
    """True 的连续区间 [(start, end_exclusive), ...]"""
    if not flags.any():
        return []
    d = np.diff(flags.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if flags[0]:
        starts.insert(0, 0)
    if flags[-1]:
        ends.append(len(flags))
    return list(zip(starts, ends))


def _has_contrast(gray: np.ndarray, s: int, e: int, axis: int,
                  min_contrast: float) -> bool:
    """走廊 [s,e) 的颜色是否明显区别于它两侧的内容"""
    prof = gray.mean(axis=1) if axis == 0 else gray.mean(axis=0)
    band = prof[s:e]
    gutter = float(band.mean())
    above = prof[max(0, s - max(8, e - s)):s]
    below = prof[e:e + max(8, e - s)]
    deltas = []
    if above.size:
        deltas.append(abs(float(above.mean()) - gutter))
    if below.size:
        deltas.append(abs(float(below.mean()) - gutter))
    return bool(deltas) and max(deltas) >= min_contrast


def _split(gray: np.ndarray, mask: np.ndarray, box: Tuple[int, int, int, int],
           min_panel: int, min_gutter: int, min_contrast: float,
           max_depth: int, depth: int, out: List[Panel]):
    x, y, w, h = box
    if depth >= max_depth or (w < min_panel * 2 and h < min_panel * 2):
        out.append(Panel(x, y, w, h))
        return

    sub = gray[y:y + h, x:x + w]
    row_flags, col_flags = _gutter_flags(sub)
    row_runs = [(s, e) for s, e in _runs(row_flags) if e - s >= min_gutter]
    col_runs = [(s, e) for s, e in _runs(col_flags) if e - s >= min_gutter]

    # 只保留「与两侧内容有对比」的走廊
    row_runs = [r for r in row_runs if _has_contrast(sub, r[0], r[1], 0, min_contrast)]
    col_runs = [r for r in col_runs if _has_contrast(sub, r[0], r[1], 1, min_contrast)]

    row_best = max((e - s for s, e in row_runs), default=0)
    col_best = max((e - s for s, e in col_runs), default=0)

    if row_best == 0 and col_best == 0:
        out.append(Panel(x, y, w, h))
        return

    if row_best >= col_best:
        s, e = max(row_runs, key=lambda r: r[1] - r[0])
        cut = y + (s + e) // 2
        parts = [(x, y, w, cut - y), (x, cut, w, y + h - cut)]
    else:
        s, e = max(col_runs, key=lambda r: r[1] - r[0])
        cut = x + (s + e) // 2
        parts = [(x, y, cut - x, h), (cut, y, x + w - cut, h)]

    big = [(px, py, pw, ph) for px, py, pw, ph in parts
           if pw >= min_panel and ph >= min_panel]
    if not big:
        out.append(Panel(x, y, w, h))
        return
    for px, py, pw, ph in big:
        _split(gray, mask, (px, py, pw, ph), min_panel, min_gutter,
               min_contrast, max_depth, depth + 1, out)


def segment_panels(img: np.ndarray, min_panel: int = 40, min_gutter: int = 5,
                   min_contrast: float = 7.0, max_depth: int = 8,
                   min_content: float = 0.004) -> List[Panel]:
    """
    切分组图。img 为 BGR 或灰度。
    返回按阅读顺序（先行后列）排序的 Panel 列表。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge = ((np.abs(gx) + np.abs(gy)) > 18.0).astype(np.float32)

    panels: List[Panel] = []
    _split(gray, edge, (0, 0, gray.shape[1], gray.shape[0]),
           min_panel, min_gutter, min_contrast, max_depth, 0, panels)

    # 丢弃近乎空白的块
    kept = []
    for p in panels:
        if p.w < min_panel or p.h < min_panel:
            continue
        sub_edge = edge[p.y:p.y + p.h, p.x:p.x + p.w]
        if float(sub_edge.mean()) < min_content and \
           float(gray[p.y:p.y + p.h, p.x:p.x + p.w].std()) < 3.0:
            continue
        kept.append(p)

    if not kept:
        kept = [Panel(0, 0, gray.shape[1], gray.shape[0])]

    kept.sort(key=lambda p: (p.y, p.x))
    bands: List[List[Panel]] = []
    for p in kept:
        for band in bands:
            ref = band[0]
            if abs((p.y + p.h / 2) - (ref.y + ref.h / 2)) < max(p.h, ref.h) * 0.5:
                band.append(p)
                break
        else:
            bands.append([p])
    ordered: List[Panel] = []
    for band in bands:
        band.sort(key=lambda p: p.x)
        ordered.extend(band)
    for i, p in enumerate(ordered):
        p.index = i
    return ordered


def draw_panels(img: np.ndarray, panels: List[Panel],
                color: Tuple[int, int, int] = (0, 140, 255)) -> np.ndarray:
    """调试用：把切分结果画出来"""
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for p in panels:
        cv2.rectangle(vis, (p.x, p.y), (p.x + p.w, p.y + p.h), color, 2)
        cv2.putText(vis, f'P{p.index}', (p.x + 4, p.y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis
