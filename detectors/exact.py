from detectors.base import BaseDetector, MatchResult
from typing import Optional


class ExactDetector(BaseDetector):
    name = "exact"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        if info1.md5 == info2.md5 and info1.file_size == info2.file_size:
            return MatchResult(
                image1=info1.path,
                image2=info2.path,
                match_type="完全相同（MD5）",
                similarity=1.0,
                severity="critical",
                details=f"MD5: {info1.md5}",
                is_cross_channel=(relation == "SAME_FOV_DIFF_CH"),
            )
        return None
