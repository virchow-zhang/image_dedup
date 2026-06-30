from detectors.base import BaseDetector, MatchResult
from typing import Optional


class PerceptualDetector(BaseDetector):
    name = "perceptual"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        cfg = self.config
        base_threshold = cfg.get("phash_threshold", 5)
        if relation == "SAME_FOV_DIFF_CH":
            boost = cfg.get("cross_channel", {}).get("phash_boost", 0.6)
            threshold = max(1, int(base_threshold * boost))
        else:
            threshold = base_threshold

        phash_diff = info1.phash - info2.phash
        dhash_diff = info1.dhash - info2.dhash
        ahash_diff = info1.ahash - info2.ahash

        best_diff = min(phash_diff, dhash_diff, ahash_diff)
        best_name = {phash_diff: "pHash", dhash_diff: "dHash", ahash_diff: "aHash"}[best_diff]

        if best_diff <= threshold:
            hash_size = info1.phash.hash.size
            sim = 1.0 - best_diff / hash_size
            sev = "critical" if best_diff <= 2 else "high"
            if relation == "SAME_FOV_DIFF_CH":
                sev = "medium"
            return MatchResult(
                image1=info1.path,
                image2=info2.path,
                match_type=f"感知哈希相似（{best_name}）",
                similarity=round(sim, 4),
                severity=sev,
                details=f"{best_name}差异: {best_diff}/{hash_size}",
                is_cross_channel=(relation == "SAME_FOV_DIFF_CH"),
            )
        return None
