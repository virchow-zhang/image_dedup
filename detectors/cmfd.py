"""内部区域复制检测 (Copy-Move Forgery Detection, CMFD)。

方法: SIFT 特征自匹配 + 位移聚类 + 几何一致性校验。
同图内被复制粘贴的区域, 其 SIFT 特征互相匹配, 且匹配点对的位移
(粘贴位置 - 源位置) 高度一致 -> 聚类后即可定位源区域与粘贴区域。
"""
from typing import Optional

import cv2
import numpy as np

from detectors.base import MatchResult


class CmfdDetector:
    name = "cmfd"

    def __init__(self, config: dict):
        self.config = config

    def detect(self, info) -> Optional[MatchResult]:
        """检测单张图片内部是否存在区域复制粘贴。"""
        cfg = self.config
        min_matches = cfg.get("cmfd_min_matches", 6)
        max_dim = cfg.get("cmfd_max_dim", 2048)

        info.ensure_sift(max_dim=max_dim, nfeatures=3000)
        des = info.sift_des
        kp = info.sift_kp
        if des is None or len(des) < min_matches * 2:
            return None

        try:
            # 自匹配必须排除"自身描述子"作为最近邻:
            # FLANN 会把 query 自身返回为 distance=0 的最近邻,
            # 导致比例测试把真正的复制区域匹配丢弃。
            # 用 BFMatcher k=3, 取前两个非自身的邻居做比例测试。
            bf = cv2.BFMatcher(cv2.NORM_L2)
            knn = bf.knnMatch(des, des, k=3)

            matches = []
            for trio in knn:
                non_self = [m for m in trio if m.queryIdx != m.trainIdx]
                if len(non_self) < 2:
                    continue
                m, n = non_self[0], non_self[1]
                if m.distance < 0.8 * n.distance:
                    matches.append(m)

            if len(matches) < min_matches:
                return None

            kp_arr = np.array([p.pt for p in kp])
            q = kp_arr[[m.queryIdx for m in matches]]
            t = kp_arr[[m.trainIdx for m in matches]]
            disp = t - q

            ow, oh = info.size
            min_disp = max(64, min(ow, oh) // 8)

            clusters = _cluster_displacements(disp, cell=32)
            best = None
            for cluster in clusters:
                if len(cluster) < min_matches:
                    continue
                idx = np.array(cluster)
                d = disp[idx]
                mean_d = d.mean(axis=0)
                var = d.var(axis=0).sum()
                if np.abs(mean_d).sum() < min_disp:
                    continue
                if var > 2500:  # 位移方差过大, 几何不一致
                    continue
                qc, tc = q[idx], t[idx]
                sim = len(cluster) / max(10, len(kp))
                if best is None or sim > best[0]:
                    best = (sim, qc, tc, mean_d, var, len(cluster))

            if best is None:
                return None

            sim, qc, tc, mean_d, var, n_matches = best

            r1 = _bbox(qc, ow, oh)
            r2 = _bbox(tc, ow, oh)
            if _overlap_ratio(r1, r2) > 0.6:
                return None

            sev = "critical" if n_matches >= 12 and sim > 0.02 else "high"
            details = (
                f"匹配点数:{n_matches}, 位移:({mean_d[0]:.0f},{mean_d[1]:.0f}), "
                f"源区域{r1}, 粘贴区域{r2}"
            )
            return MatchResult(
                image1=info.path,
                image2=info.path,
                match_type="内部区域复制",
                similarity=round(sim, 4),
                severity=sev,
                details=details,
                is_cross_channel=False,
                region1=r1,
                region2=r2,
            )
        except Exception:
            return None


def _cluster_displacements(disp: np.ndarray, cell: int = 32):
    """按位移向量量化到 32px 网格聚类, 返回每个簇的匹配下标。"""
    bins = np.floor((disp + 0.5 * cell) / cell).astype(np.int64)
    keys = {}
    for i, b in enumerate(bins):
        keys.setdefault((b[0], b[1]), []).append(i)
    return list(keys.values())


def _bbox(pts: np.ndarray, ow: int, oh: int) -> tuple:
    x0 = max(0, int(pts[:, 0].min()))
    y0 = max(0, int(pts[:, 1].min()))
    x1 = min(ow, int(pts[:, 0].max()))
    y1 = min(oh, int(pts[:, 1].max()))
    return (x0, y0, x1 - x0, y1 - y0)


def _overlap_ratio(r1: tuple, r2: tuple) -> float:
    x0 = max(r1[0], r2[0])
    y0 = max(r1[1], r2[1])
    x1 = min(r1[0] + r1[2], r2[0] + r2[2])
    y1 = min(r1[1] + r1[3], r2[1] + r2[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    a1 = r1[2] * r1[3]
    return inter / max(1e-6, a1)
