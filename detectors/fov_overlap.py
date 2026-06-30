import os
from typing import Optional, List, Tuple
import numpy as np
import cv2

from detectors.base import BaseDetector, MatchResult


class FovOverlapMatch:
    def __init__(self, image1: str, image2: str, match_type: str,
                 similarity: float, severity: str, details: str,
                 is_cross_channel: bool,
                 overlap_direction: str = "",
                 estimated_dx: float = 0,
                 estimated_dy: float = 0,
                 cluster_count: int = 0,
                 clusters_a: list = None,
                 clusters_b: list = None):
        self.image1 = image1
        self.image2 = image2
        self.match_type = match_type
        self.similarity = similarity
        self.severity = severity
        self.details = details
        self.is_cross_channel = is_cross_channel
        self.overlap_direction = overlap_direction
        self.estimated_dx = estimated_dx
        self.estimated_dy = estimated_dy
        self.cluster_count = cluster_count
        self.clusters_a = clusters_a or []  # list of (cx, cy, diameter)
        self.clusters_b = clusters_b or []


class FovOverlapDetector(BaseDetector):
    name = "fov_overlap"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[FovOverlapMatch]:
        if relation == "SAME_FOV_DIFF_CH":
            return None

        if info1.sift_des is None or info2.sift_des is None:
            return None
        if len(info1.sift_des) < 8 or len(info2.sift_des) < 8:
            return None

        cfg = self.config.get("fov_overlap", {})
        min_inliers = cfg.get("min_inliers", 8)
        dbscan_eps = cfg.get("dbscan_eps", 30)
        dbscan_min_samples = cfg.get("dbscan_min_samples", 3)
        edge_margin = cfg.get("edge_margin", 60)
        cluster_min_diameter = cfg.get("cluster_min_diameter", 15)
        disp_var_threshold = cfg.get("displacement_var_threshold", 15)

        des1 = info1.sift_des
        des2 = info2.sift_des
        kp1 = info1.sift_kp
        kp2 = info2.sift_kp

        try:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            knn_matches = flann.knnMatch(des1, des2, k=2)
        except Exception:
            return None

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
            inliers = len(good)
            mask = np.ones(len(good), dtype=np.uint8)
        else:
            inliers = int(mask.sum())

        if inliers < min_inliers:
            return None

        inlier_idx = mask.ravel().astype(bool)
        in_pts1 = pts1[inlier_idx]
        in_pts2 = pts2[inlier_idx]

        clusters = self._dbscan_cluster(in_pts1, eps=dbscan_eps,
                                         min_samples=dbscan_min_samples)
        if not clusters:
            return None

        valid_clusters = []
        for clus in clusters:
            diameter = self._cluster_diameter(clus)
            if diameter < cluster_min_diameter:
                continue

            cx1, cy1 = clus.mean(axis=0)
            cx2, cy2 = self._corresponding_center(clus, in_pts1, in_pts2)

            is_edge_a = (cx1 < edge_margin or
                         cx1 > 256 - edge_margin or
                         cy1 < edge_margin or
                         cy1 > 256 - edge_margin)
            if not is_edge_a:
                continue

            edge_side_a = self._edge_side(cx1, cy1, edge_margin, 256)
            edge_side_b = self._edge_side(cx2, cy2, edge_margin, 256)

            is_opposite = self._is_opposite_edge(edge_side_a, edge_side_b)
            if not is_opposite:
                continue

            displacements = in_pts2[clus_indices(clus, in_pts1)] - in_pts1[clus_indices(clus, in_pts1)]
            if len(displacements) >= 2:
                var = np.var(displacements, axis=0).sum()
                if var > disp_var_threshold:
                    continue

            valid_clusters.append({
                "center_a": (float(cx1), float(cy1)),
                "center_b": (float(cx2), float(cy2)),
                "diameter": float(diameter),
                "edge_side_a": edge_side_a,
                "edge_side_b": edge_side_b,
            })

        if not valid_clusters:
            return None

        cluster_count = len(valid_clusters)
        severity = "high" if cluster_count >= 2 else "medium"
        direction = self._infer_direction(valid_clusters)
        dx, dy = self._estimate_translation(in_pts1, in_pts2)

        details = (
            f"视野重叠: {direction}方向, {cluster_count}个细胞簇, "
            f"平移(dx={dx:.0f}, dy={dy:.0f})"
        )

        clusters_a = [(c["center_a"][0], c["center_a"][1], c["diameter"])
                      for c in valid_clusters]
        clusters_b = [(c["center_b"][0], c["center_b"][1], c["diameter"])
                      for c in valid_clusters]

        result = FovOverlapMatch(
            image1=info1.path,
            image2=info2.path,
            match_type="视野重叠（FOV）",
            similarity=round(cluster_count / max(cluster_count + 2, 1), 4),
            severity=severity,
            details=details,
            is_cross_channel=False,
            overlap_direction=direction,
            estimated_dx=float(dx),
            estimated_dy=float(dy),
            cluster_count=cluster_count,
            clusters_a=clusters_a,
            clusters_b=clusters_b,
        )
        return result

    def _dbscan_cluster(self, pts: np.ndarray, eps: float, min_samples: int) -> list:
        if len(pts) < min_samples:
            return []
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
        labels = clustering.labels_
        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            mask = labels == label
            clusters.append(pts[mask])
        return clusters

    def _cluster_diameter(self, cluster: np.ndarray) -> float:
        if len(cluster) < 2:
            return 0.0
        center = cluster.mean(axis=0)
        dists = np.linalg.norm(cluster - center, axis=1)
        return float(2.0 * dists.max())

    def _corresponding_center(self, cluster, all_pts1, all_pts2):
        mask = np.isin(all_pts1, cluster).all(axis=1)
        corr_pts = all_pts2[mask]
        if len(corr_pts) > 0:
            return corr_pts.mean(axis=0)
        return all_pts2.mean(axis=0)

    def _edge_side(self, cx: float, cy: float, margin: float, img_size: int) -> str:
        sides = []
        if cx < margin:
            sides.append("left")
        if cx > img_size - margin:
            sides.append("right")
        if cy < margin:
            sides.append("top")
        if cy > img_size - margin:
            sides.append("bottom")
        return "+".join(sides) if sides else "center"

    def _is_opposite_edge(self, side_a: str, side_b: str) -> bool:
        pairs = [("left", "right"), ("right", "left"),
                 ("top", "bottom"), ("bottom", "top")]
        for sa, sb in pairs:
            if sa in side_a and sb in side_b:
                return True
        return False

    def _infer_direction(self, clusters: list) -> str:
        sides_a = [c["edge_side_a"] for c in clusters]
        if all("left" in s for s in sides_a):
            return "水平(左→右)"
        if all("right" in s for s in sides_a):
            return "水平(右→左)"
        if all("top" in s for s in sides_a):
            return "垂直(上→下)"
        if all("bottom" in s for s in sides_a):
            return "垂直(下→上)"
        return "水平+垂直"

    def _estimate_translation(self, pts1, pts2) -> Tuple[float, float]:
        d = pts2 - pts1
        median_d = np.median(d, axis=0)
        return float(median_d[0]), float(median_d[1])


def clus_indices(clus: np.ndarray, all_pts: np.ndarray) -> np.ndarray:
    indices = []
    for p in clus:
        matches = np.where((all_pts == p).all(axis=1))[0]
        if len(matches) > 0:
            indices.append(matches[0])
    return np.array(indices)
