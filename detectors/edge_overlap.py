from detectors.base import BaseDetector, MatchResult
from typing import Optional
import numpy as np


class EdgeOverlapDetector(BaseDetector):
    name = "edge_overlap"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        if relation == "SAME_FOV_DIFF_CH":
            return None

        cfg = self.config
        threshold = cfg.get("edge_threshold", 0.55)
        strip_w = 24

        try:
            e1 = info1.canny_256
            e2 = info2.canny_256

            h_score, h_conf = _check_overlap_h(e1, e2, strip_w)
            v_score, v_conf = _check_overlap_v(e1, e2, strip_w)

            candidates = []
            if h_score >= threshold:
                candidates.append((h_score, "水平", h_conf))
            if v_score >= threshold:
                candidates.append((v_score, "垂直", v_conf))

            if not candidates:
                return None

            best = max(candidates, key=lambda x: x[0])
            score, direction, conf = best

            details = f"边缘重合率={conf:.2f}"
            sev = "high" if score > 0.75 else "medium"
            return MatchResult(
                image1=info1.path,
                image2=info2.path,
                match_type=f"疑似{direction}边缘重叠/拼接",
                similarity=round(float(score), 4),
                severity=sev,
                details=details,
                is_cross_channel=False,
            )
        except Exception:
            pass
        return None


def _edge_overlap_ratio(ea: np.ndarray, eb: np.ndarray) -> float:
    overlap = np.logical_and(ea > 0, eb > 0).sum()
    total = np.logical_or(ea > 0, eb > 0).sum()
    if total == 0:
        return 0.0
    return float(overlap) / float(total)


def _ncc_edge(ea: np.ndarray, eb: np.ndarray) -> float:
    a = ea.astype(np.float64).ravel()
    b = eb.astype(np.float64).ravel()
    a_n = a - a.mean()
    b_n = b - b.mean()
    denom = (np.linalg.norm(a_n) * np.linalg.norm(b_n))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a_n, b_n) / denom)


def _check_overlap_h(e1: np.ndarray, e2: np.ndarray, w: int):
    rs = e1[:, -w:]
    ls = e2[:, :w]
    ratio = _edge_overlap_ratio(rs, ls)
    ncc = _ncc_edge(rs, ls)
    combined = ratio * 0.6 + max(0, ncc) * 0.4
    return (combined, ratio)


def _check_overlap_v(e1: np.ndarray, e2: np.ndarray, h: int):
    bs = e1[-h:, :]
    ts = e2[:h, :]
    ratio = _edge_overlap_ratio(bs, ts)
    ncc = _ncc_edge(bs, ts)
    combined = ratio * 0.6 + max(0, ncc) * 0.4
    return (combined, ratio)
