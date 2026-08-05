"""合成科研图片查重测试集生成器。

生成"类显微"基底图片 (细胞/颗粒/噪声), 施加 10+ 类常见造假/复用变换,
输出 ground-truth manifest.json 供 eval.py 评估召回率与误报率。

用法:
    python tests/gen_synthetic.py --out tests/data/syn1 --baselines 30
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np

TRANSFORMS = [
    "exact_copy", "rotate90", "rotate180", "rotate270",
    "flip_h", "flip_v", "rot_free", "scale_07", "crop_60",
    "brightness_70", "contrast_150", "jpeg_blur", "copy_move", "splice",
]


def make_baseline(seed: int, size: int = 512) -> np.ndarray:
    """生成类显微荧光图: 暗背景 + 随机椭圆细胞 + 噪声。"""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 18 + rng.integers(0, 12), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    n_cells = rng.integers(40, 90)
    for _ in range(n_cells):
        cx, cy = rng.integers(0, size, 2)
        rx, ry = rng.integers(3, 22, 2)
        angle = rng.uniform(0, np.pi)
        a = rng.uniform(0, 120)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), int(np.degrees(angle)),
                    0, 360, 255, -1)
        cv2.circle(mask, (int(cx), int(cy)), max(1, int(min(rx, ry) * 0.5)),
                   int(60 + a), -1)
    g = rng.normal(0, 8, (size, size))
    img = np.clip(img.astype(np.float32) + mask * 0.35 + g, 0, 255).astype(np.uint8)
    return img


def apply_transform(img: np.ndarray, kind: str, rng) -> np.ndarray:
    if kind == "exact_copy":
        return img.copy()
    if kind == "rotate90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if kind == "rotate180":
        return cv2.rotate(img, cv2.ROTATE_180)
    if kind == "rotate270":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if kind == "flip_h":
        return cv2.flip(img, 1)
    if kind == "flip_v":
        return cv2.flip(img, 0)
    if kind == "rot_free":
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 17.0, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if kind == "scale_07":
        return cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    if kind == "crop_60":
        h, w = img.shape[:2]
        x0, y0 = int(w * 0.2), int(h * 0.2)
        return img[y0:y0 + int(h * 0.6), x0:x0 + int(w * 0.6)]
    if kind == "brightness_70":
        return np.clip(img * 0.7, 0, 255).astype(np.uint8)
    if kind == "contrast_150":
        mean = img.mean()
        return np.clip((img - mean) * 1.5 + mean, 0, 255).astype(np.uint8)
    if kind == "jpeg_blur":
        out = cv2.GaussianBlur(img, (3, 3), 0.8)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if kind == "copy_move":
        out = img.copy()
        h, w = out.shape[:2]
        bh, bw = 120, 120
        x1, y1 = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        x2, y2 = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        while abs(x2 - x1) < 180 and abs(y2 - y1) < 180:
            x2, y2 = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        block = out[y1:y1 + bh, x1:x1 + bw].copy()
        noise = rng.normal(0, 3, block.shape)
        block = np.clip(block.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        out[y2:y2 + bh, x2:x2 + bw] = block
        return out
    if kind == "splice":
        return img  # splice 需要另一张图, 由生成器单独处理
    raise ValueError(kind)


def splice_images(img_a: np.ndarray, img_b: np.ndarray, rng) -> np.ndarray:
    """把 img_b 中心区域拼进 img_a, 两张图共享该区域。"""
    out = img_a.copy()
    h, w = out.shape[:2]
    bh, bw = int(h * 0.35), int(w * 0.35)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    block = img_b[y0:y0 + bh, x0:x0 + bw].copy()
    noise = rng.normal(0, 3, block.shape)
    block = np.clip(block.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    out[y0:y0 + bh, x0:x0 + bw] = block
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/data/syn1")
    ap.add_argument("--baselines", type=int, default=30)
    ap.add_argument("--source-dir", default=None,
                    help="用真实图片代替程序化生成 (可选)")
    args = ap.parse_args()

    out_dir = args.out
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(20260701)
    manifest = {"pairs": [], "files": []}

    baselines = {}
    if args.source_dir:
        from core.loader import scan_images
        imgs = scan_images(args.source_dir, {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
        for i, p in enumerate(imgs[:args.baselines]):
            import cv2 as _cv
            raw = _cv.imread(p, _cv.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            if max(raw.shape) > 768:
                s = 768 / max(raw.shape)
                raw = _cv.resize(raw, None, fx=s, fy=s, interpolation=_cv.INTER_AREA)
            baselines[i] = raw
    else:
        for i in range(args.baselines):
            baselines[i] = make_baseline(1000 + i)

    n_bl = len(baselines)
    pairs = []  # (file1, file2, transform_kind)

    for i, img in baselines.items():
        name = f"base_{i:04d}.png"
        cv2.imwrite(os.path.join(out_dir, name), img)
        manifest["files"].append(name)

        for t in TRANSFORMS:
            if t == "splice":
                continue
            tname = f"{t}_{i:04d}.png"
            out = apply_transform(img, t, rng)
            cv2.imwrite(os.path.join(out_dir, tname), out)
            manifest["files"].append(tname)
            pairs.append((name, tname, t))

    # splice: A 与 B 共享区域
    for i in range(n_bl):
        j = (i + 1) % n_bl
        a, b = baselines[i], baselines[j]
        out = splice_images(a, b, rng)
        sname = f"splice_{i:04d}.png"
        cv2.imwrite(os.path.join(out_dir, sname), out)
        manifest["files"].append(sname)
        pairs.append((f"base_{i:04d}.png", sname, "splice"))

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"pairs": pairs, "files": manifest["files"]}, f,
                  ensure_ascii=False, indent=1)

    print(f"生成完成: {out_dir}")
    print(f"  基底图: {n_bl}, 变换图: {n_bl * (len(TRANSFORMS) - 1) + n_bl}, "
          f"ground-truth 对: {len(pairs)}")


if __name__ == "__main__":
    main()
