from detectors.base import BaseDetector, MatchResult
from typing import Optional
import numpy as np
from PIL import Image
import imagehash


class TransformDetector(BaseDetector):
    name = "transform"

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        if relation == "SAME_FOV_DIFF_CH":
            return None

        cfg = self.config
        threshold = cfg.get("phash_threshold", 5)
        hash_size = cfg.get("phash_hash_size", 16)

        transforms = {
            "旋转90°": lambda im: np.rot90(im, k=1),
            "旋转180°": lambda im: np.rot90(im, k=2),
            "旋转270°": lambda im: np.rot90(im, k=3),
            "水平翻转": lambda im: np.fliplr(im),
            "垂直翻转": lambda im: np.flipud(im),
        }

        hash2 = info2.phash
        g1 = info1.gray_256

        for name, fn in transforms.items():
            try:
                t_arr = fn(g1)
                t_pil = Image.fromarray(t_arr)
                ht = imagehash.phash(t_pil, hash_size=hash_size)
                diff = ht - hash2
                if diff <= threshold:
                    sim = 1.0 - diff / ht.hash.size
                    return MatchResult(
                        image1=info1.path,
                        image2=info2.path,
                        match_type=f"疑似{name}",
                        similarity=round(sim, 4),
                        severity="critical",
                        details=f"变换后哈希差异: {diff}",
                        is_cross_channel=False,
                    )
            except Exception:
                continue
        return None
