from detectors.base import BaseDetector, MatchResult
from typing import Optional
import numpy as np
from PIL import Image


class SubimageDetector(BaseDetector):
    name = "subimage"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        if relation == "SAME_FOV_DIFF_CH":
            return None

        cfg = self.config
        threshold = cfg.get("subimage_threshold", 0.85)

        try:
            big_path = info1.path if info1.file_size >= info2.file_size else info2.path
            small_path = info2.path if info1.file_size >= info2.file_size else info1.path

            big = Image.open(big_path).convert('L')
            small = Image.open(small_path).convert('L')
            bw, bh = big.size
            sw, sh = small.size

            if sw >= bw or sh >= bh:
                return None

            best_val = -1.0
            best_scale = 1.0
            best_pos = (0, 0)

            for scale in [1.0, 0.9, 0.8, 0.7, 0.6]:
                nw, nh = int(sw * scale), int(sh * scale)
                if nw < 20 or nh < 20 or nw > bw or nh > bh:
                    continue
                tmpl = np.array(small.resize((nw, nh)), dtype=np.float64)
                t_norm = (tmpl - tmpl.mean()) / (tmpl.std() + 1e-10)
                arr = np.array(big, dtype=np.float64)
                step = max(1, min(bw, bh) // 100)
                for y in range(0, bh - nh + 1, step):
                    for x in range(0, bw - nw + 1, step):
                        patch = arr[y:y+nh, x:x+nw]
                        p_norm = (patch - patch.mean()) / (patch.std() + 1e-10)
                        score = float(np.mean(p_norm * t_norm))
                        if score > best_val:
                            best_val = score
                            best_pos = (x, y)
                            best_scale = scale

            if best_val >= threshold:
                return MatchResult(
                    image1=big_path,
                    image2=small_path,
                    match_type="疑似子图/裁剪",
                    similarity=round(float(best_val), 4),
                    severity="critical",
                    details=f"原图位置({best_pos[0]},{best_pos[1]}), 缩放比:{best_scale:.1f}",
                    is_cross_channel=False,
                )
        except Exception:
            pass
        return None
