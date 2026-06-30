from detectors.base import BaseDetector, MatchResult
from typing import Optional
import numpy as np


class HistogramDetector(BaseDetector):
    name = "histogram"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        cfg = self.config
        threshold = cfg.get("hist_threshold", 0.9999)

        h1 = info1.hist - info1.hist.mean()
        h2 = info2.hist - info2.hist.mean()
        corr = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10)
        bhatta = np.sum(np.sqrt(info1.hist * info2.hist))
        avg_sim = (max(0, corr) + bhatta) / 2

        if avg_sim >= threshold:
            sev = "high" if avg_sim > 0.999 else "medium"
            details = f"相关性: {corr:.4f}, 巴氏系数: {bhatta:.4f}"
            return MatchResult(
                image1=info1.path,
                image2=info2.path,
                match_type="直方图相似（疑似曝光调整）",
                similarity=round(avg_sim, 6),
                severity=sev,
                details=details,
                is_cross_channel=(relation == "SAME_FOV_DIFF_CH"),
            )
        return None
