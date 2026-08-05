from detectors.base import BaseDetector, MatchResult
from typing import Optional
import numpy as np
import cv2


IMG_SIZE = 256
EDGE_MARGIN = 60
N_BINS = 8


class FeatureMatchDetector(BaseDetector):
    name = "feature_match"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        if relation == "SAME_FOV_DIFF_CH":
            return None

        cfg = self.config
        min_inliers = cfg.get("sift_min_inliers", 10)
        max_dim = cfg.get("sift_max_dim", 1024)
        nfeatures = cfg.get("sift_nfeatures", 2000)

        info1.ensure_sift(max_dim=max_dim, nfeatures=nfeatures)
        info2.ensure_sift(max_dim=max_dim, nfeatures=nfeatures)

        des1, des2 = info1.sift_des, info2.sift_des
        kp1, kp2 = info1.sift_kp, info2.sift_kp
        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return None

        try:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            knn_matches = flann.knnMatch(des1, des2, k=2)

            good = []
            for pair in knn_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) < min_inliers:
                return None

            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            if H is None:
                inlier_mask = np.ones(len(good), dtype=bool)
                inliers = len(good)
            else:
                inlier_mask = mask.ravel().astype(bool)
                inliers = int(mask.sum())

            if inliers < min_inliers:
                return None

            in_pts1 = pts1[inlier_mask]
            in_pts2 = pts2[inlier_mask]

            # 覆盖率判据: 内点必须覆盖足够大的区域, 排除
            # 纹理自相似造成的少量伪内点簇
            ow, oh = info1.size
            xs, ys = in_pts1[:, 0], in_pts1[:, 1]
            bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
            coverage = (bw * bh) / max(1, ow * oh)
            edge_conc = _edge_concentration(in_pts1, IMG_SIZE, EDGE_MARGIN)
            if edge_conc < 0.7 and (coverage < 0.04 or inliers < 12):
                return None

            sim = min(1.0, inliers / max(len(kp1), len(kp2)))

            transform_type, angle, scale = _analyze_transform(H)
            dx, dy = _median_translation(in_pts1, in_pts2)
            edge_cont = _edge_continuity(in_pts1, IMG_SIZE, EDGE_MARGIN, N_BINS)
            disp_var = _displacement_variance(in_pts1, in_pts2)

            n_total = len(kp1) + len(kp2)
            sev = _severity(inliers, n_total, edge_conc, edge_cont, disp_var)

            pt1_list = [(int(p[0]), int(p[1])) for p in in_pts1]
            pt2_list = [(int(p[0]), int(p[1])) for p in in_pts2]

            details = _build_details(inliers, len(good), transform_type, angle,
                                     scale, dx, dy, edge_conc, edge_cont)

            return MatchResult(
                image1=info1.path,
                image2=info2.path,
                match_type=f"特征匹配（{transform_type}）",
                similarity=round(sim, 4),
                severity=sev,
                details=details,
                is_cross_channel=False,
                match_points1=pt1_list,
                match_points2=pt2_list,
                transform_type=transform_type,
                rotation_deg=round(angle, 1),
                scale=round(scale, 3),
                dx=round(dx, 1),
                dy=round(dy, 1),
                edge_concentration=round(edge_conc, 2),
                edge_continuity=round(edge_cont, 2),
                inlier_count=inliers,
            )
        except Exception:
            return None


def _analyze_transform(H: np.ndarray) -> tuple:
    R = H[:2, :2]
    sx = np.linalg.norm(R[:, 0])
    sy = np.linalg.norm(R[:, 1])
    scale = (sx + sy) / 2.0
    R_norm = R / sx if sx > 1e-6 else R
    angle = np.degrees(np.arctan2(R_norm[1, 0], R_norm[0, 0]))

    if abs(scale - 1.0) > 0.05:
        return ("缩放", angle, scale)
    if abs(angle) > 5:
        return ("旋转", angle, scale)
    return ("平移", angle, scale)


def _median_translation(pts1: np.ndarray, pts2: np.ndarray):
    d = pts2 - pts1
    m = np.median(d, axis=0)
    return float(m[0]), float(m[1])


def _edge_concentration(pts: np.ndarray, img_size: int, margin: int) -> float:
    if len(pts) < 3:
        return 0.0
    left = (pts[:, 0] < margin).sum()
    right = (pts[:, 0] > img_size - margin).sum()
    top = (pts[:, 1] < margin).sum()
    bottom = (pts[:, 1] > img_size - margin).sum()
    total = len(pts)
    return max(left, right, top, bottom) / total


def _edge_continuity(pts: np.ndarray, img_size: int, margin: int, n_bins: int) -> float:
    sides = {
        "left": (pts[:, 0] < margin),
        "right": (pts[:, 0] > img_size - margin),
        "top": (pts[:, 1] < margin),
        "bottom": (pts[:, 1] > img_size - margin),
    }
    best_ratio = 0.0
    for side, mask in sides.items():
        if mask.sum() < 2:
            continue
        side_pts = pts[mask]
        if side in ("left", "right"):
            positions = side_pts[:, 1]
        else:
            positions = side_pts[:, 0]
        if positions.max() == positions.min():
            continue
        bins = np.linspace(0, img_size, n_bins + 1)
        counts, _ = np.histogram(positions, bins=bins)
        filled = (counts > 0).sum()
        ratio = filled / n_bins
        if ratio > best_ratio:
            best_ratio = ratio
    return best_ratio


def _displacement_variance(pts1: np.ndarray, pts2: np.ndarray) -> float:
    if len(pts1) < 3:
        return 999.0
    d = pts2 - pts1
    return float(np.var(d, axis=0).sum())


def _severity(inliers: int, n_total: int, edge_conc: float,
              edge_cont: float, disp_var: float) -> str:
    if inliers < 10:
        return "low"
    high_confidence = (inliers >= 30 and edge_conc < 0.5)
    edge_overlap = (edge_conc >= 0.7 and edge_cont >= 0.6 and disp_var < 30)
    if high_confidence:
        return "critical"
    if edge_overlap:
        return "high"
    if inliers >= 15:
        return "high"
    return "medium"


def _build_details(inliers, total_matches, transform_type, angle,
                   scale, dx, dy, edge_conc, edge_cont) -> str:
    parts = [f"内点: {inliers}/{total_matches}"]
    if transform_type == "旋转":
        parts.append(f"旋转: {angle:.1f}°")
    elif transform_type == "缩放":
        parts.append(f"缩放: {scale:.2f}x")
    else:
        parts.append(f"平移: ({dx:.0f}, {dy:.0f})")
    if edge_conc >= 0.7:
        parts.append(f"边缘集中: {edge_conc:.0%}")
    if edge_cont >= 0.6:
        parts.append(f"连续: {edge_cont:.0%}")
    return ", ".join(parts)
