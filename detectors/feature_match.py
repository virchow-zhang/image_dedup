from detectors.base import BaseDetector, MatchResult
from typing import Optional
import numpy as np


class FeatureMatchDetector(BaseDetector):
    name = "feature_match"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        if relation == "SAME_FOV_DIFF_CH":
            return None

        cfg = self.config
        min_matches = cfg.get("orb_min_matches", 10)

        des1, des2 = info1.orb_des, info2.orb_des
        kp1, kp2 = info1.orb_kp, info2.orb_kp
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return None

        try:
            import cv2

            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)

            good = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) < min_matches:
                return None

            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            if H is None:
                inliers = len(good)
            else:
                inliers = int(mask.sum())

            if inliers < min_matches:
                return None

            sim = min(1.0, inliers / max(len(kp1), len(kp2)))
            sev = "critical" if inliers >= 50 else "high" if inliers >= 20 else "medium"

            pt1_list = [(int(p[0][0]), int(p[0][1])) for p in pts1]
            pt2_list = [(int(p[0][0]), int(p[0][1])) for p in pts2]

            return MatchResult(
                image1=info1.path,
                image2=info2.path,
                match_type="特征点匹配（ORB）",
                similarity=round(sim, 4),
                severity=sev,
                details=f"匹配点: {len(good)}, RANSAC内点: {inliers}",
                is_cross_channel=False,
                match_points1=pt1_list,
                match_points2=pt2_list,
            )
        except Exception:
            return None
