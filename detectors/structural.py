from detectors.base import BaseDetector, MatchResult
from typing import Optional
import numpy as np
import os


def _compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu1, mu2 = a.mean(), b.mean()
    s1, s2 = a.var(), b.var()
    s12 = np.mean((a - mu1) * (b - mu2))
    ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
    return float(ssim)


class StructuralDetector(BaseDetector):
    name = "structural"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        cfg = self.config
        base_threshold = cfg.get("ssim_threshold", 0.98)
        if relation == "SAME_FOV_DIFF_CH":
            boost = cfg.get("cross_channel", {}).get("ssim_boost", 1.1)
            threshold = min(0.999, base_threshold * boost)
        else:
            threshold = base_threshold

        ssim = _compute_ssim(info1.gray_256, info2.gray_256)
        if ssim >= threshold:
            sev = "critical" if ssim > 0.999 else "high" if ssim > 0.995 else "medium"

            g1 = info1.group_info
            g2 = info2.group_info
            ctx = ""
            if g1.get("group") and g2.get("group"):
                if g1["group"] == g2["group"]:
                    ctx = f"[同组] {g1['group']}"
                else:
                    ctx = f"[跨组] {g1['group']} vs {g2['group']}"
            if g1.get("fov") and g2.get("fov"):
                if g1["fov"] == g2["fov"]:
                    ctx += f" 同FOV({g1['fov']})"
                else:
                    ctx += f" 不同FOV({g1['fov']} vs {g2['fov']})"

            if relation == "SAME_FOV_DIFF_CH":
                sev = "medium"

            return MatchResult(
                image1=info1.path,
                image2=info2.path,
                match_type="结构相似（SSIM）",
                similarity=round(ssim, 4),
                severity=sev,
                details=f"SSIM: {ssim:.4f} {ctx}",
                is_cross_channel=(relation == "SAME_FOV_DIFF_CH"),
            )
        return None
