"""合成测试集评估: 逐变换类型统计召回率, 并统计误报率。

用法:
    python tests/eval.py --data tests/data/syn1 [--ai] [--threshold N]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import run_detection

TRANSFORM_NAMES = {
    "exact_copy": "完全相同", "rotate90": "旋转90°", "rotate180": "旋转180°",
    "rotate270": "旋转270°", "flip_h": "水平翻转", "flip_v": "垂直翻转",
    "rot_free": "自由旋转17°", "scale_07": "缩放0.7×", "crop_60": "裁剪60%",
    "brightness_70": "亮度×0.7", "contrast_150": "对比度×1.5",
    "jpeg_blur": "JPEG+模糊", "copy_move": "同图复制粘贴", "splice": "跨图拼接",
}


def load_manifest(data_dir: str):
    with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def build_config(threshold: int, use_ai: bool):
    return {
        "scan": {
            "extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
            "max_workers": 8,
        },
        "grouping": {"patterns": []},
        "index": {"mih_tables": 16, "rotation_variants": True,
                  "bow_enabled": True, "bow_min_shared": 6,
                  "candidate_threshold": 15},
        "detection": {
            "phash_threshold": threshold,
            "phash_hash_size": 16,
            "rotation_enabled": True,
            "ssim_threshold": 0.95,
            "hist_enabled": False,
            "sift_enabled": True,
            "sift_min_inliers": 8,
            "sift_max_dim": 1024,
            "sift_nfeatures": 2000,
            "edge_threshold": 0.55,
            "subimage_enabled": True,
            "subimage_threshold": 0.80,
            "cmfd_enabled": True,
            "cmfd_min_matches": 5,
            "cmfd_max_dim": 2048,
        },
        "ai": {
            "enabled": use_ai,
            "model_path": os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "models", "mobilenetv2-7.onnx"),
            "cosine_threshold": 0.95,
            "top_k": 10,
            "match_threshold": 0.97,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tests/data/syn1")
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--ai", action="store_true")
    ap.add_argument("--baselines-only-fp", action="store_true",
                    help="只统计随机基底对之间的误报")
    args = ap.parse_args()

    data_dir = args.data
    manifest = load_manifest(data_dir)
    pairs = manifest["pairs"]

    expected = {(a, b) for a, b, _ in pairs}
    by_type = {}
    for a, b, t in pairs:
        by_type.setdefault(t, []).append((a, b))

    cfg = build_config(args.threshold, args.ai)
    t0 = time.time()
    infos, matches, _ = run_detection(data_dir, cfg)
    elapsed = time.time() - t0

    found = set()
    for m in matches:
        if m.image1 != m.image2:
            found.add((os.path.basename(m.image1), os.path.basename(m.image2)))

    print(f"\n===== 评估结果 (阈值={args.threshold}, AI={'开' if args.ai else '关'}, "
          f"耗时 {elapsed:.1f}s) =====")
    print(f"{'变换类型':<14}{'GT对数':>8}{'检出':>8}{'召回率':>10}")
    print("-" * 44)
    recall_by_type = {}
    for t, ps in by_type.items():
        hit = sum(1 for a, b in ps if (a, b) in found or (b, a) in found)
        recall = hit / len(ps)
        recall_by_type[t] = recall
        print(f"{TRANSFORM_NAMES.get(t, t):<14}{len(ps):>8}{hit:>8}{recall:>9.1%}")

    base_files = [f for f in manifest["files"] if f.startswith("base_")]
    fp = 0
    base_set = set()
    for a, b in found:
        if a.startswith("base_") and b.startswith("base_"):
            if (a, b) not in expected and (b, a) not in expected:
                fp += 1
                base_set.add((a, b))
    n_base_pairs = len(base_files) * (len(base_files) - 1) // 2
    print("-" * 44)
    print(f"误报(随机基底对): {fp}/{n_base_pairs} ({fp / max(1, n_base_pairs):.2%})")
    for a, b in sorted(base_set)[:10]:
        print(f"  FP: {a} vs {b}")

    overall_hit = sum(1 for a, b in expected if (a, b) in found or (b, a) in found)
    print(f"总体召回: {overall_hit}/{len(expected)} ({overall_hit / len(expected):.1%})")
    print(f"检出总数: {len(found)} (含{len(base_set)}个基底间误报)")


if __name__ == "__main__":
    main()
