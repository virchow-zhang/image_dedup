"""
图内区域匹配核心
================
在一张图内部寻找「成对出现的相似区域」——panel 复用、局部重叠、图内复制粘贴。

与跨文件查重的本质区别：
  跨文件是「两张图整体是否相同」；图内是「同一张图里两个局部区域是否同源」，
  区域可能只占图的一小块，且可能经过旋转/翻转/缩放/裁剪/调色。

方法（SIFT + 迭代 RANSAC，Amerini 等人的经典 copy-move 思路）：
  1. 在原图与「水平镜像图」上分别提 SIFT，镜像特征坐标映射回原图坐标系
     —— 镜像后坐标系的仿射拟合会得到负行列式，从而统一处理翻转副本
  2. 全部特征自匹配（FLANN kNN），剔除自身与空间近邻，做 Lowe 比值检验
  3. 迭代 RANSAC 拟合仿射：一次找出一个主导变换，剔除内点后再找下一个
     —— 一张图里可能有多组复用，单次 RANSAC 只能找到一个
  4. 内点包围盒定位两个区域；再用「把 B 按 M 变换回 A 的坐标系」做
     零均值归一化互相关(NCC)验证 —— 这一步专门杀掉「重复纹理」
     （如一片长得差不多的细胞核）造成的伪匹配
  5. 由仿射矩阵反解旋转角/缩放/是否翻转，给出人类可读的变换类型
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RegionMatch:
    """一对同源区域"""
    a_bbox: List[int]           # [x, y, w, h] 区域 A
    b_bbox: List[int]           # [x, y, w, h] 区域 B
    transform: str              # 人类可读变换类型
    rotation_deg: float
    scale: float
    flip: bool
    n_inliers: int
    ncc: float
    score: float
    a_panel: Optional[int] = None
    b_panel: Optional[int] = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 特征提取


def _detect(gray: np.ndarray, max_features: int):
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.008,
                           edgeThreshold=16, sigma=1.2)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts = np.array([k.pt for k in kps], np.float32)
    return pts, desc.astype(np.float32)


def build_match_set(gray: np.ndarray, max_features: int = 4000
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (coords Nx2, desc Nx128, origin N)，坐标统一在原图坐标系。
    origin=0 原图特征，origin=1 镜像图特征（描述的是「镜像后的样子」）。
    """
    h, w = gray.shape[:2]
    p1, d1 = _detect(gray, max_features)
    p2, d2 = _detect(cv2.flip(gray, 1), max_features)
    if len(p2):
        # 镜像图坐标 -> 原图坐标
        p2 = p2.copy()
        p2[:, 0] = (w - 1) - p2[:, 0]
    coords = np.vstack([p1, p2]) if len(p1) and len(p2) else (p1 if len(p1) else p2)
    desc = np.vstack([d1, d2]) if len(p1) and len(p2) else (d1 if len(d1) else d2)
    origin = np.concatenate([np.zeros(len(p1), np.int8), np.ones(len(p2), np.int8)])
    return coords, desc, origin


# ---------------------------------------------------------------- 自匹配


def self_match(coords: np.ndarray, desc: np.ndarray, min_sep: float,
               ratio: float = 0.80, knn: int = 10) -> List[Tuple[int, int, float]]:
    """
    描述子自匹配。剔除自身与空间距离 < min_sep 的近邻（否则每个特征都会
    匹配到自己或紧邻的同一个结构），再做 Lowe 比值检验。
    """
    n = len(desc)
    if n < 4:
        return []
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
    knn_matches = flann.knnMatch(desc, desc, k=min(knn + 1, n))

    pairs = []
    for i, ms in enumerate(knn_matches):
        good = []
        for m in ms:
            j = m.trainIdx
            if j == i:
                continue
            if np.hypot(coords[j, 0] - coords[i, 0], coords[j, 1] - coords[i, 1]) < min_sep:
                continue
            good.append(m)
            if len(good) == 2:
                break
        if len(good) == 2 and good[0].distance < ratio * good[1].distance:
            pairs.append((i, good[0].trainIdx, float(good[0].distance)))
    return pairs


# ---------------------------------------------------------------- RANSAC


def iterative_ransac(coords: np.ndarray, pairs: List[Tuple[int, int, float]],
                     min_inliers: int, reproj: float = 3.5,
                     max_models: int = 5, max_iters: int = 4000
                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    迭代 RANSAC：一次拟合一个主导仿射变换，剔除内点后继续找下一个。
    返回 [(M 2x3, 内点索引数组), ...]
    """
    if len(pairs) < min_inliers:
        return []
    src = np.array([coords[i] for i, _, _ in pairs], np.float32)
    dst = np.array([coords[j] for _, j, _ in pairs], np.float32)
    remaining = np.arange(len(pairs))
    models = []
    for _ in range(max_models):
        if len(remaining) < min_inliers:
            break
        M, inl = cv2.estimateAffine2D(
            src[remaining], dst[remaining], method=cv2.RANSAC,
            ransacReprojThreshold=reproj, maxIters=max_iters, confidence=0.995)
        if M is None or inl is None:
            break
        inl = inl.ravel().astype(bool)
        if int(inl.sum()) < min_inliers:
            break
        idx = remaining[inl]
        models.append((M, idx))
        remaining = remaining[~inl]
    return models


def classify_transform(M: np.ndarray) -> Tuple[str, float, float, bool]:
    """由仿射矩阵反解 旋转角/缩放/翻转，给出可读标签"""
    a, b = float(M[0, 0]), float(M[0, 1])
    c, d = float(M[1, 0]), float(M[1, 1])
    scale = float(np.sqrt(abs(a * d - b * c)))
    det = a * d - b * c
    flip = det < 0
    ang = float(np.degrees(np.arctan2(c, a)))
    while ang > 180:
        ang -= 360
    while ang < -180:
        ang += 360

    parts = []
    if abs(scale - 1.0) > 0.08:
        parts.append(f'缩放{scale:.2f}x')
    if flip:
        parts.append('镜像')
    if abs(ang) > 12:
        if abs(abs(ang) - 180) < 14:
            parts.append('旋转180°')
        elif abs(abs(ang) - 90) < 14:
            parts.append('旋转90°')
        else:
            parts.append(f'旋转{ang:.0f}°')
    if not parts:
        parts.append('平移')
    return '+'.join(parts), ang, scale, flip


# ---------------------------------------------------------------- 验证


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _texture_mask(gray: np.ndarray, win: int = 9, std_thresh: float = 6.0
                  ) -> np.ndarray:
    """
    局部方差掩膜：只保留有结构的像素。

    纯白留白、纯黑背景这类平坦区域「看起来当然一样」，但它们不含信息：
    既会让一致掩膜顺着留白把相邻两个 panel 连成一片（区域被撑大），
    也会把 NCC 抬虚高（两处留白互相加分）。所以一致性判定和 NCC 都只在
    有纹理的像素上做。
    """
    g = gray.astype(np.float32)
    mu = cv2.blur(g, (win, win))
    mu2 = cv2.blur(g * g, (win, win))
    var = np.maximum(mu2 - mu * mu, 0.0)
    return var > (std_thresh ** 2)


def _warp(gray: np.ndarray, M: np.ndarray):
    """按 M 变换整图，同时返回有效像素掩膜。"""
    h, w = gray.shape[:2]
    warped = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), M, (w, h),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    return warped, valid


def _ncc_in(gray: np.ndarray, warped: np.ndarray, valid: np.ndarray,
            box: Tuple[int, int, int, int], tex: Optional[np.ndarray] = None,
            min_overlap: int = 400) -> float:
    h, w = gray.shape[:2]
    x, y, bw, bh = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    a = gray[y0:y1, x0:x1].astype(np.float32)
    bimg = warped[y0:y1, x0:x1].astype(np.float32)
    m = valid[y0:y1, x0:x1] > 0
    if tex is not None:
        m &= tex[y0:y1, x0:x1]
    if int(m.sum()) < min_overlap:
        return 0.0
    av, bv = a[m] - a[m].mean(), bimg[m] - bimg[m].mean()
    denom = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    if denom < 1e-6:
        return 0.0
    return float((av * bv).sum() / denom)


def verify_ncc(gray: np.ndarray, M: np.ndarray, box: Tuple[int, int, int, int],
               tex: Optional[np.ndarray] = None, min_overlap: int = 400) -> float:
    """
    验证 M 是否真的把 B 处的图像搬到了 A 处。

    warpAffine(gray, M) 满足 warped(p) = gray(M⁻¹p)，所以对区域 B 内的 p，
    warped(p) = gray(M⁻¹p) ∈ A —— 即 warped 在 B 区域的内容正是 A 的内容。
    因此在 **B 区域** 比较 warped 与原图，等价于比较 A 与 B。
    （早期版本在 A 区域比较，等于拿 A 和 A 自己比，恒为高分，
      导致所有自映射都通过验证。）
    零均值归一化互相关对亮度/对比度调整不敏感。
    """
    warped, valid = _warp(gray, M)
    return _ncc_in(gray, warped, valid, box, tex, min_overlap)


def _agreement_mask(gray: np.ndarray, warped: np.ndarray, valid: np.ndarray,
                    seed: Tuple[int, int, int, int], tex: np.ndarray,
                    tol: float = 0.20):
    """
    线性归一化后的逐像素一致掩膜 + 连通域标注。

    关键点只落在有纹理的地方：blot 这类稀疏特征图，内点包围盒只覆盖
    条带附近，远小于真实复用范围，定位 IoU 会很低。一致掩膜的连通域
    能给出完整的同源区域边界。
    返回 (labels, stats, num)；失败返回 None。
    """
    h, w = gray.shape[:2]
    x, y, bw, bh = seed
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    m = valid[y0:y1, x0:x1] > 0
    if int(m.sum()) < 100:
        return None

    a = gray[y0:y1, x0:x1].astype(np.float32)
    b = warped[y0:y1, x0:x1].astype(np.float32)
    am, asd = float(a[m].mean()), float(a[m].std()) + 1e-6
    bm, bsd = float(b[m].mean()), float(b[m].std()) + 1e-6

    gn = (gray.astype(np.float32) - am) / asd
    wn = (warped.astype(np.float32) - bm) / bsd
    agree = ((np.abs(gn - wn) < tol) & (valid > 0) & tex).astype(np.uint8)
    agree = cv2.morphologyEx(agree, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(agree, connectivity=8)
    return (labels, stats, num) if num > 1 else None


def _component_bbox(mask, seed: Tuple[int, int, int, int], fallback
                    ) -> Tuple[int, int, int, int]:
    """取包含 seed 中心的连通域包围盒；域太小则退回 fallback。"""
    labels, stats, num = mask
    x, y, bw, bh = seed
    h, w = labels.shape
    cx, cy = min(w - 1, x + bw // 2), min(h - 1, y + bh // 2)
    lab = int(labels[cy, cx])
    if lab == 0:
        region = labels[y:y + bh, x:x + bw]
        vals, counts = np.unique(region[region > 0], return_counts=True)
        if len(vals) == 0:
            return fallback
        lab = int(vals[int(np.argmax(counts))])
    bx, by = int(stats[lab, cv2.CC_STAT_LEFT]), int(stats[lab, cv2.CC_STAT_TOP])
    bwd, bht = int(stats[lab, cv2.CC_STAT_WIDTH]), int(stats[lab, cv2.CC_STAT_HEIGHT])
    if bwd * bht < bw * bh:
        return fallback
    return (bx, by, bwd, bht)


def dedupe_matches(matches: List[RegionMatch], iou_thresh: float = 0.40
                   ) -> List[RegionMatch]:
    """
    迭代 RANSAC 会把同一处复用同时解出正向和反向两个变换（A→B 与 B→A），
    导致同一条线索报两次、拉高误报。这里按区域对去重，保留得分高的。
    """
    kept: List[RegionMatch] = []
    for m in sorted(matches, key=lambda r: (-r.score, -r.n_inliers)):
        dup = False
        for k in kept:
            direct = (_iou(m.a_bbox, k.a_bbox) > iou_thresh and
                      _iou(m.b_bbox, k.b_bbox) > iou_thresh)
            swapped = (_iou(m.a_bbox, k.b_bbox) > iou_thresh and
                       _iou(m.b_bbox, k.a_bbox) > iou_thresh)
            if direct or swapped:
                dup = True
                break
        if not dup:
            kept.append(m)
    return kept


def _cluster_inliers(coords: np.ndarray, pairs: List[Tuple[int, int, float]],
                     idx: np.ndarray, min_inliers: int, sep: float
                     ) -> List[np.ndarray]:
    """
    把一个 RANSAC 模型的内点按空间聚簇。

    同一组复用往往只是整图的一个局部，而拟合出的变换可能同时「顺带」
    解释别处的巧合配对（例如整图 180° 旋转既覆盖真实的 p0↔p4 复用，
    也覆盖一堆零散巧合），于是内点包围盒被撑成整张图。
    按「源点相近 且 目标点相近」聚簇后，每个簇才是真正的一处复用。
    """
    if len(idx) < min_inliers:
        return []
    src = np.array([coords[pairs[k][0]] for k in idx], np.float32)
    dst = np.array([coords[pairs[k][1]] for k in idx], np.float32)
    n = len(src)
    if n == 1:
        return [np.arange(1)]

    if n > 4000:
        # 退化路径：网格分桶，避免 n² 邻接矩阵爆内存
        keys = np.stack([(src[:, 0] // sep).astype(np.int64),
                         (src[:, 1] // sep).astype(np.int64),
                         (dst[:, 0] // sep).astype(np.int64),
                         (dst[:, 1] // sep).astype(np.int64)], axis=1)
        _, inv_idx = np.unique(keys, axis=0, return_inverse=True)
        comp = inv_idx.ravel()
    else:
        close = lambda a: np.abs(a[:, None] - a[None, :]) < sep
        adj = (close(src[:, 0]) & close(src[:, 1]) &
               close(dst[:, 0]) & close(dst[:, 1])).astype(np.uint8)
        labels = cv2.connectedComponents(adj, connectivity=4)[1]
        # 邻接矩阵是对称的：点 i 的连通域号就在对角线上
        comp = labels[np.arange(n), np.arange(n)]

    out = []
    for lab in np.unique(comp):
        sel = np.where(comp == lab)[0]
        if len(sel) >= min_inliers:
            out.append(sel)
    return out


def _dense_seed(src_pts: np.ndarray, dst_pts: np.ndarray, radius: float,
                shape: Tuple[int, int]):
    """
    取「局部最密」的内点子集作为种子。

    自映射模型（整图 180° 旋转等）的内点会撒满整张图，直接用全部内点的
    包围盒会得到整图。先找局部最密的一点，用它邻域内的内点做种子，
    种子就落在真实复用的那一块里，之后靠区域生长扩到完整范围。
    """
    n = len(src_pts)
    if n == 0:
        return None, None
    dx = np.abs(src_pts[:, 0][:, None] - src_pts[:, 0][None, :]) < radius
    dy = np.abs(src_pts[:, 1][:, None] - src_pts[:, 1][None, :]) < radius
    cnt = (dx & dy).sum(axis=1)
    i = int(np.argmax(cnt))
    sel = np.where(dx[i] & dy[i])[0]
    return (cv2.boundingRect(src_pts[sel].reshape(-1, 1, 2)),
            cv2.boundingRect(dst_pts[sel].reshape(-1, 1, 2)))


def _grow_region(gray: np.ndarray, warped: np.ndarray, valid: np.ndarray,
                 tex: np.ndarray, seed: Tuple[int, int, int, int], M: np.ndarray,
                 step: int = 10, max_quality_drop: float = 0.12,
                 tol: float = 0.20, max_iters: int = 400
                 ) -> Tuple[Tuple[int, int, int, int], float]:
    """
    以种子盒为起点向四边生长，直到「一致率」明显下降为止。

    为什么不用 NCC 当生长判据：区域长到留白/相邻 panel 上时，新增面积里
    几乎没有纹理像素（被纹理掩膜排除），原本吻合的纹理像素仍占主导，
    NCC 几乎不掉 —— 区域会一路长到整张图。改用「盒内纹理像素的一致比例」
    作判据，新增区域一旦不再逐像素吻合，一致率立刻下降，生长自然停止。

    这是定位精度的关键：关键点只落在有纹理处，blot/gel 这类稀疏特征图的
    内点包围盒往往只覆盖条带附近，生长才能扩到内容真正不再吻合的边界。
    """
    h, w = gray.shape[:2]
    x, y, bw, bh = seed
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return tuple(seed), 0.0
    m0 = (valid[y0:y1, x0:x1] > 0) & tex[y0:y1, x0:x1]
    if int(m0.sum()) < 100:
        return tuple(seed), 0.0

    g = gray.astype(np.float32)
    wimg = warped.astype(np.float32)
    gm, gs = float(g[y0:y1, x0:x1][m0].mean()), float(g[y0:y1, x0:x1][m0].std()) + 1e-6
    wm, ws = float(wimg[y0:y1, x0:x1][m0].mean()), float(wimg[y0:y1, x0:x1][m0].std()) + 1e-6
    agree = (np.abs((g - gm) / gs - (wimg - wm) / ws) < tol) & (valid > 0) & tex

    def quality(box):
        a, b = max(0, box[0]), max(0, box[1])
        c, d = min(w, box[0] + box[2]), min(h, box[1] + box[3])
        if c - a < 4 or d - b < 4:
            return 0.0
        m = (valid[b:d, a:c] > 0) & tex[b:d, a:c]
        if int(m.sum()) < 100:
            return 0.0
        return float(agree[b:d, a:c][m].mean())

    base = quality(tuple(seed))
    if base <= 0:
        return tuple(seed), 0.0

    cur = [int(v) for v in seed]
    it = 0
    improved = True
    while improved and it < max_iters:
        improved = False
        it += 1
        for side in range(4):
            cand = cur[:]
            if side == 0 and cand[0] >= step:
                cand[0] -= step
                cand[2] += step
            elif side == 1 and cand[1] >= step:
                cand[1] -= step
                cand[3] += step
            elif side == 2 and cand[0] + cand[2] + step <= w:
                cand[2] += step
            elif side == 3 and cand[1] + cand[3] + step <= h:
                cand[3] += step
            else:
                continue
            if quality(cand) >= base - max_quality_drop:
                cur = cand
                improved = True

    ncc = _ncc_in(gray, warped, valid,
                  _apply_affine_bbox(M, tuple(cur), (h, w)), tex)
    return tuple(cur), ncc


def _degenerate_line(pts: np.ndarray, ratio: float = 0.03) -> bool:
    """
    内点是否退化成一条线。

    单张散点/折线图里，长直线（趋势线、坐标轴）本身没有判别力：RANSAC 可以
    沿这条线凑出一组自洽的对应，NCC 还不低。真实的两处同源复用，内点应当
    在二维上铺开，因此用协方差最小/最大特征值之比做退化检测。
    """
    if len(pts) < 6:
        return True
    c = pts - pts.mean(axis=0)
    cov = (c.T @ c) / len(pts)
    ev = np.linalg.eigvalsh(cov)
    return float(ev[0]) <= max(float(ev[1]), 1e-6) * ratio


# ---------------------------------------------------------------- 跨图配对
#
# 下面这套是「判定层」的关键补充。哈希/SSIM 投票要求**整图结构一致**，
# 因此对「变换的叠加」（裁剪∘旋转、镜像∘任意角旋转）无能为力：
# 实测 17° 旋转副本的 pHash 距离是 36/256，裁剪60% 是 118/256。
# SIFT 描述子天生抗旋转与缩放，可以在**只有局部证据**时判定同源。
#
# 但纯 SIFT+RANSAC 会大量误报（实测 master 那条线在难负样本上 114 个误报，
# 其中 88 个跨实验），根因是长直线（散点图趋势线、坐标轴）能让 RANSAC
# 凑出自洽变换。所以这里叠加三道约束，缺一不可：
#   1. 退化检测     —— 内点不能退化成一条线
#   2. 纹理掩膜 NCC —— 只在有结构的像素上比对，留白不算数
#   3. 区域占比标注 —— 匹配区域远小于整图时归为"局部复用/拼接"


def build_pair_sets(gray_a: np.ndarray, gray_b: np.ndarray, max_features: int):
    """分别构建 A、B 的特征集（含镜像），坐标各自在自己的图像坐标系里。"""
    return (build_match_set(gray_a, max_features),
            build_match_set(gray_b, max_features))


def ratio_match(pa, da, pb, db, ratio: float = 0.80, knn: int = 2):
    """A→B 的描述子匹配 + Lowe 比值检验。返回 [(i, j), ...]"""
    if len(da) < 2 or len(db) < 2:
        return []
    try:
        ms = cv2.BFMatcher().knnMatch(da, db, k=min(knn, len(db)))
    except cv2.error:
        return []
    out = []
    for m in ms:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            out.append((m[0].queryIdx, m[0].trainIdx))
    return out


def _pair_ncc(gray_a: np.ndarray, gray_b: np.ndarray, M: np.ndarray,
              box: Tuple[int, int, int, int], tex: np.ndarray,
              min_overlap: int = 300) -> float:
    """
    把 B 按 M 映射到 A 的坐标系后，在 box 内与 A 做纹理掩膜 NCC。

    坑：cv2.warpAffine 默认把传入矩阵当**目标→源**，即 dst(p) = src(M⁻¹p)；
    要得到 dst(p) = src(M·p) 必须显式加 WARP_INVERSE_MAP。
    这里需要后者 —— M 是 A→B 的变换，只有 warped(p) = gray_b(M·p)
    才是「A 坐标系里 B 应该长什么样」。
    漏掉该 flag 时恒等变换恰好也对（所以 exact_copy 能过），
    而旋转/镜像类会全部 NCC≈0，表面上看非常像"匹配失败"。
    """
    ha, wa = gray_a.shape[:2]
    hb, wb = gray_b.shape[:2]
    warped = cv2.warpAffine(gray_b, M, (wa, ha),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.ones((hb, wb), np.uint8), M, (wa, ha),
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, bw, bh = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(wa, x + bw), min(ha, y + bh)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    a = gray_a[y0:y1, x0:x1].astype(np.float32)
    b = warped[y0:y1, x0:x1].astype(np.float32)
    m = (valid[y0:y1, x0:x1] > 0) & tex[y0:y1, x0:x1]
    if int(m.sum()) < min_overlap:
        return 0.0
    av = a[m] - a[m].mean()
    bv = b[m] - b[m].mean()
    denom = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    return float((av * bv).sum() / denom) if denom > 1e-6 else 0.0


def find_pair_matches(gray_a: np.ndarray, gray_b: np.ndarray,
                      min_inliers: int = 12,
                      ncc_threshold: float = 0.60,
                      min_region: int = 32,
                      max_features: int = 2000,
                      detect_max_dim: int = 900,
                      pad_frac: float = 0.05,
                      max_models: int = 3,
                      max_self_overlap: float = 0.30) -> List[RegionMatch]:
    """
    判断两张图之间是否存在同源区域（支持裁剪/旋转/缩放/镜像/局部拼接）。

    返回的 RegionMatch：a_bbox 在 gray_a 坐标系，b_bbox 在 gray_b 坐标系。
    """
    ha, wa = gray_a.shape[:2]
    hb, wb = gray_b.shape[:2]
    sa = min(1.0, detect_max_dim / max(ha, wa))
    sb = min(1.0, detect_max_dim / max(hb, wb))
    ga = cv2.resize(gray_a, (max(1, int(wa * sa)), max(1, int(ha * sa))),
                    interpolation=cv2.INTER_AREA) if sa < 1 else gray_a
    gb = cv2.resize(gray_b, (max(1, int(wb * sb)), max(1, int(hb * sb))),
                    interpolation=cv2.INTER_AREA) if sb < 1 else gray_b
    return match_features(extract_pair_features(ga, max_features),
                          extract_pair_features(gb, max_features),
                          min_inliers=min_inliers, ncc_threshold=ncc_threshold,
                          min_region=min_region, pad_frac=pad_frac,
                          max_models=max_models, max_self_overlap=max_self_overlap,
                          scale_a=sa, scale_b=sb)


def extract_pair_features(gray: np.ndarray, max_features: int = 1000,
                          max_dim: int = 900) -> dict:
    """
    提取一张图用于跨图配对的特征（含镜像），并缓存纹理掩膜。

    单独抽出来是为了能按路径缓存：不做缓存时每对候选都要重跑两次 SIFT，
    1200 张规模下这部分会从秒级涨到分钟级。
    """
    h, w = gray.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    g = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))),
                   interpolation=cv2.INTER_AREA) if s < 1 else gray
    coords, desc, _ = build_match_set(g, max_features)
    return {'gray': g, 'coords': coords, 'desc': desc,
            'tex': _texture_mask(g), 'shape': g.shape[:2], 'scale': s}


def match_features(fa: dict, fb: dict,
                   min_inliers: int = 12,
                   ncc_threshold: float = 0.60,
                   min_region: int = 32,
                   pad_frac: float = 0.05,
                   max_models: int = 3,
                   max_self_overlap: float = 0.30,
                   scale_a: float = 1.0, scale_b: float = 1.0
                   ) -> List[RegionMatch]:
    """已提取特征版本的两图同源判定。bbox 会映射回原始分辨率。"""
    pa, da = fa['coords'], fa['desc']
    pb, db = fb['coords'], fb['desc']
    ga, gb = fa['gray'], fb['gray']
    if len(pa) < min_inliers or len(pb) < min_inliers:
        return []

    # coords 把 A、B 的特征拼在一起，pairs 的 src 落在 A 段、dst 落在 B 段，
    # 这样可以直接复用单图版那套迭代 RANSAC。
    # 偏移必须是 len(pa)（A 段长度），不是 len(pb)。
    offset = len(pa)
    coords = np.vstack([pa, pb])
    raw = ratio_match(pa, da, pb, db)
    if len(raw) < min_inliers:
        return []
    pairs = [(i, offset + j, 0.0) for i, j in raw]

    tex_a = fa['tex']
    out: List[RegionMatch] = []

    for M, idx in iterative_ransac(coords, pairs, min_inliers, max_models=max_models):
        src_pts = np.array([coords[pairs[k][0]] for k in idx], np.float32)
        dst_pts = np.array([coords[pairs[k][1]] for k in idx], np.float32)

        ax, ay, aw, ah = cv2.boundingRect(src_pts.reshape(-1, 1, 2))
        bx, by, bw, bh = cv2.boundingRect(dst_pts.reshape(-1, 1, 2))
        if min(aw, ah, bw, bh) < min_region:
            continue
        # 退化检测：长直线（趋势线/坐标轴）能让 RANSAC 凑出自洽变换，
        # 这是纯 SIFT 方案误报的主因，必须挡掉
        if _degenerate_line(src_pts) or _degenerate_line(dst_pts):
            continue

        pax, pay = int(aw * pad_frac), int(ah * pad_frac)
        pbx, pby = int(bw * pad_frac), int(bh * pad_frac)
        abox = (max(0, ax - pax), max(0, ay - pay),
                min(ga.shape[1] - max(0, ax - pax), aw + 2 * pax),
                min(ga.shape[0] - max(0, ay - pay), ah + 2 * pay))
        bbox_ = (max(0, bx - pbx), max(0, by - pby),
                 min(gb.shape[1] - max(0, bx - pbx), bw + 2 * pbx),
                 min(gb.shape[0] - max(0, by - pby), bh + 2 * pby))

        ncc = _pair_ncc(ga, gb, M, abox, tex_a)
        if ncc < ncc_threshold:
            continue

        label, ang, sc, flip = classify_transform(M)
        frac = (aw * ah) / float(ga.shape[0] * ga.shape[1])
        if frac < 0.6:
            label = '局部' + label if label != '平移' else '局部平移/拼接'

        def _back(box, s):
            x, y, w_, h_ = box
            inv = 1.0 / s if s < 1 else 1.0
            return [int(round(x * inv)), int(round(y * inv)),
                    int(round(w_ * inv)), int(round(h_ * inv))]

        out.append(RegionMatch(
            a_bbox=_back(abox, scale_a), b_bbox=_back(bbox_, scale_b),
            transform=label, rotation_deg=round(ang, 1), scale=round(sc, 3),
            flip=flip, n_inliers=int(len(idx)), ncc=round(ncc, 4),
            score=round(float(ncc) * min(1.0, len(idx) / (min_inliers * 3)), 4),
            meta={'M': M.tolist(), 'coverage': round(frac, 3)},
        ))

    return dedupe_matches(out)


# ---------------------------------------------------------------- 主流程
def _apply_affine_bbox(M: np.ndarray, box: Tuple[int, int, int, int],
                       shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """把 box 的四个角经 M 变换后取包围盒"""
    h, w = shape


def find_region_matches(gray: np.ndarray,
                        min_inliers: int = 14,
                        ncc_threshold: float = 0.55,
                        min_region: int = 40,
                        min_sep_frac: float = 0.06,
                        max_features: int = 4000,
                        detect_max_dim: int = 1400,
                        pad_frac: float = 0.06,
                        max_models: int = 5,
                        max_self_overlap: float = 0.30,
                        salvage_self_maps: bool = False) -> List[RegionMatch]:
    """
    在图内寻找同源区域对。

    gray: 灰度图。若尺寸过大，会先降采样做特征检测，定位结果再映射回原尺寸。
    """
    h0, w0 = gray.shape[:2]
    scale = min(1.0, detect_max_dim / max(h0, w0))
    if scale < 1.0:
        gw = cv2.resize(gray, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                        interpolation=cv2.INTER_AREA)
    else:
        gw = gray
    h, w = gw.shape[:2]

    coords, desc, origin = build_match_set(gw, max_features)
    if len(coords) < min_inliers:
        return []

    min_sep = min_sep_frac * min(h, w)
    pairs = self_match(coords, desc, min_sep)
    if len(pairs) < min_inliers:
        return []

    inv = 1.0 / scale if scale < 1.0 else 1.0
    tex = _texture_mask(gw)
    out: List[RegionMatch] = []

    def _back(box):
        x, y, bw_, bh_ = box
        return [int(round(x * inv)), int(round(y * inv)),
                int(round(bw_ * inv)), int(round(bh_ * inv))]

    def _emit(M, warped, valid, abox, bbox_, n_inl):
        """验证 + 用一致掩膜细化区域 + 落盘一条线索"""
        ncc = _ncc_in(gw, warped, valid, bbox_, tex)
        if ncc < ncc_threshold:
            return
        mask = _agreement_mask(gw, warped, valid, abox, tex)
        if mask is not None:
            ra = _component_bbox(mask, abox, abox)
            rb = _component_bbox(mask, bbox_, bbox_)
            ncc_new = _ncc_in(gw, warped, valid, rb, tex)
            if ncc_new >= ncc - 0.05:
                abox, bbox_, ncc = ra, rb, ncc_new
        label, ang, sc, flip = classify_transform(M)
        out.append(RegionMatch(
            a_bbox=_back(abox), b_bbox=_back(bbox_),
            transform=label, rotation_deg=round(ang, 1), scale=round(sc, 3),
            flip=flip, n_inliers=int(n_inl), ncc=round(ncc, 4),
            score=round(float(ncc) * min(1.0, n_inl / (min_inliers * 3)), 4),
            meta={'M': M.tolist()},
        ))

    seed_radius = 0.12 * min(h, w)
    for M, idx in iterative_ransac(coords, pairs, min_inliers, max_models=max_models):
        warped, valid = _warp(gw, M)
        src_pts = np.array([coords[pairs[k][0]] for k in idx], np.float32)
        dst_pts = np.array([coords[pairs[k][1]] for k in idx], np.float32)

        ax, ay, aw, ah = cv2.boundingRect(src_pts.reshape(-1, 1, 2))
        bx, by, bw, bh = cv2.boundingRect(dst_pts.reshape(-1, 1, 2))
        if min(aw, ah, bw, bh) < min_region:
            continue
        if _degenerate_line(src_pts) or _degenerate_line(dst_pts):
            continue

        pax, pay = int(aw * pad_frac), int(ah * pad_frac)
        pbx, pby = int(bw * pad_frac), int(bh * pad_frac)
        abox = (max(0, ax - pax), max(0, ay - pay),
                min(w - max(0, ax - pax), aw + 2 * pax),
                min(h - max(0, ay - pay), ah + 2 * pay))
        bbox_ = (max(0, bx - pbx), max(0, by - pby),
                 min(w - max(0, bx - pbx), bw + 2 * pbx),
                 min(h - max(0, by - pby), bh + 2 * pby))

        if _iou(abox, bbox_) <= max_self_overlap:
            _emit(M, warped, valid, abox, bbox_, len(idx))
            continue

        # 自映射模型（整图对称 / 大面积重复纹理）：内点撒满整张图，包围盒
        # 退化成整图。改用「局部最密内点」当种子定位真实复用块。
        #
        # 默认关闭：实测能捞出「整图 180° 旋转」类漏报（如 fig_013），
        # 但密集内点会在无关处也凑出种子，误报增加得比分得多（F1 0.70→0.63）。
        # 整图对称仍然是已知短板，保留该路径供后续改进。
        if not salvage_self_maps:
            continue
        sa, sb = _dense_seed(src_pts, dst_pts, seed_radius, (h, w))
        if sa is None or min(sa[2], sa[3], sb[2], sb[3]) < min_region:
            continue
        if _iou(sa, sb) > max_self_overlap:
            continue
        n_in = int(np.sum((src_pts[:, 0] >= sa[0]) & (src_pts[:, 0] < sa[0] + sa[2]) &
                          (src_pts[:, 1] >= sa[1]) & (src_pts[:, 1] < sa[1] + sa[3])))
        _emit(M, warped, valid, sa, sb, max(n_in, min_inliers))

    return dedupe_matches(out)
