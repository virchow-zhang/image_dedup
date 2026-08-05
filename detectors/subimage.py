from detectors.base import BaseDetector, MatchResult
from typing import Optional
import cv2
import numpy as np

from core.loader import read_gray_capped


class SubimageDetector(BaseDetector):
    """子图/裁剪检测: 多尺度金字塔 + cv2.matchTemplate (C 内核)。

    相比 v2 的 Python 滑动窗口 NCC, 此处用 OpenCV 原生模板匹配,
    并在 256 -> 512 -> 1024 三级金字塔上由粗到精定位, 每对毫秒级。
    """

    name = "subimage"

    LEVELS = [256, 512, 1024]

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        if relation == "SAME_FOV_DIFF_CH":
            return None

        cfg = self.config
        threshold = cfg.get("subimage_threshold", 0.85)

        try:
            big_path = info1.path if info1.file_size >= info2.file_size else info2.path
            small_path = info2.path if info1.file_size >= info2.file_size else info1.path

            big = read_gray_capped(big_path, max_dim=1024)
            small = read_gray_capped(small_path, max_dim=1024)

            bh, bw = big.shape[:2]
            sh, sw = small.shape[:2]

            # 面积太接近整图, 不是裁剪场景
            if sh * sw > bh * bw * 0.8:
                return None
            if min(sh, sw) < 24 or min(bh, bw) < 64:
                return None
            # 小图比大图还大(尺寸异常)则跳过
            if sw >= bw or sh >= bh:
                return None

            best_score, best_pos, best_scale = self._coarse_to_fine(big, small)

            if best_score >= threshold:
                x, y = best_pos
                return MatchResult(
                    image1=big_path,
                    image2=small_path,
                    match_type="疑似子图/裁剪",
                    similarity=round(float(best_score), 4),
                    severity="critical",
                    details=f"原图位置({x},{y}), 缩放比:{best_scale:.2f}, 相关系数:{best_score:.3f}",
                    is_cross_channel=False,
                )
        except Exception:
            pass
        return None

    def _coarse_to_fine(self, big: np.ndarray, small: np.ndarray):
        """由粗到精: 每级只在前一级最优位置附近的小窗口内搜索。"""
        best_score = -1.0
        best_pos = None
        best_scale = 1.0
        prev_center = None
        prev_radius = None
        prev_big_shape = None

        for level in self.LEVELS:
            f = level / max(big.shape)
            if f >= 1.0:
                bigL = big
                f = 1.0
            else:
                bigL = cv2.resize(big, None, fx=f, fy=f,
                                  interpolation=cv2.INTER_AREA)
            bhL, bwL = bigL.shape[:2]

            for sc in (1.0, 0.85, 0.7, 0.55):
                tw, th = int(small.shape[1] * f * sc), int(small.shape[0] * f * sc)
                if tw < 24 or th < 24 or tw >= bwL or th >= bhL:
                    continue
                tmpl = cv2.resize(small, (tw, th), interpolation=cv2.INTER_AREA)
                tmpl_f = tmpl.astype(np.float32)
                tmpl_n = (tmpl_f - tmpl_f.mean()) / (tmpl_f.std() + 1e-8)

                if prev_center is None:
                    res = cv2.matchTemplate(bigL.astype(np.float32), tmpl_n,
                                            cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                else:
                    # 把前一级最优位置映射到本级坐标, 再开窗
                    cx = prev_center[0] * bwL / prev_big_shape[1]
                    cy = prev_center[1] * bhL / prev_big_shape[0]
                    r = int(max(16, prev_radius * bwL / prev_big_shape[1]))
                    x0 = max(0, int(cx - r))
                    y0 = max(0, int(cy - r))
                    x1 = min(bwL, int(cx + r) + tw)
                    y1 = min(bhL, int(cy + r) + th)
                    if x1 - x0 < tw or y1 - y0 < th:
                        x0 = 0
                        y0 = 0
                        x1, y1 = bwL, bhL
                    win = bigL[y0:y1, x0:x1].astype(np.float32)
                    res = cv2.matchTemplate(win, tmpl_n, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    max_loc = (max_loc[0] + x0, max_loc[1] + y0)

                if max_val > best_score:
                    best_score = float(max_val)
                    best_pos = (int(max_loc[0] / f), int(max_loc[1] / f))
                    best_scale = sc
                    prev_center = (max_loc[0], max_loc[1])
                    prev_radius = max(bwL, bhL) // 4
                    prev_big_shape = (bhL, bwL)

        return best_score, best_pos, best_scale
