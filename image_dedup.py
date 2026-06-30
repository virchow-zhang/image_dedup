#!/usr/bin/env python3
"""
科研图片查重工具
================
递归扫描目录下的所有图片，进行全量两两比对，检测：
  - 完全相同 (MD5)
  - 微小修改 / 亮度/对比度调整 (pHash/dHash/aHash)
  - 结构相似 (SSIM)
  - 曝光度调整 (直方图)
  - 旋转 / 翻转
  - 特征匹配 (ORB + RANSAC)
  - 边缘重叠 / 拼接
  - 子图 / 裁剪

用法:
    python image_dedup.py
    python image_dedup.py -d D:\\IF_images\\20260629
    python image_dedup.py -d D:\\IF_images --quick
    python image_dedup.py -d D:\\IF_images --strict
"""

import os
import sys
import argparse
import shutil
import yaml

from core.engine import run_detection
from viz.annotator import annotate_pair
from viz.report_html import generate_report


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_config(config_path: str, cli_args: dict) -> dict:
    cfg = load_config(config_path)

    if cli_args.get("directory"):
        cfg["scan"]["directory"] = cli_args["directory"]

    if cli_args.get("quick"):
        cfg["detection"]["orb_enabled"] = False
        cfg["detection"]["subimage_threshold"] = 0.99
        cfg["detection"]["edge_threshold"] = 0.99

    if cli_args.get("strict"):
        cfg["detection"]["phash_threshold"] = 3
        cfg["detection"]["ssim_threshold"] = 0.90
        cfg["detection"]["hist_threshold"] = 0.85
        cfg["detection"]["orb_min_matches"] = 8

    if cli_args.get("threads"):
        cfg["detection"]["max_workers"] = cli_args["threads"]

    if cli_args.get("threshold") is not None:
        cfg["detection"]["phash_threshold"] = cli_args["threshold"]

    if cli_args.get("ssim") is not None:
        cfg["detection"]["ssim_threshold"] = cli_args["ssim"]

    if cli_args.get("no_orb"):
        cfg["detection"]["orb_enabled"] = False

    if cli_args.get("no_rotation"):
        pass

    if cli_args.get("no_subimage"):
        cfg["detection"]["subimage_threshold"] = 1.1

    if cli_args.get("no_edge"):
        cfg["detection"]["edge_threshold"] = 1.1

    if cli_args.get("output"):
        cfg["report"]["output_dir"] = cli_args["output"]

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="科研图片查重工具 - 检测科研图片重复/造假",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python image_dedup.py
  python image_dedup.py -d D:\\IF_images\\20260629
  python image_dedup.py -d D:\\IF_images --quick
  python image_dedup.py -d D:\\IF_images --strict
  python image_dedup.py -d D:\\IF_images --threshold 3 --threads 8
        """)

    parser.add_argument("-d", "--directory", help="待扫描的顶层目录")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument("-o", "--output", help="报告输出目录")
    parser.add_argument("--quick", action="store_true", help="快速模式（跳过ORB/子图/边缘检测）")
    parser.add_argument("--strict", action="store_true", help="严格模式（降低所有阈值）")
    parser.add_argument("--no-orb", action="store_true", help="跳过ORB特征匹配")
    parser.add_argument("--no-rotation", action="store_true", help="跳过旋转/翻转检测")
    parser.add_argument("--no-subimage", action="store_true", help="跳过于图/裁剪检测")
    parser.add_argument("--no-edge", action="store_true", help="跳过边缘重叠检测")
    parser.add_argument("--threads", type=int, help="并行线程数")
    parser.add_argument("--threshold", type=int, help="pHash阈值覆盖")
    parser.add_argument("--ssim", type=float, help="SSIM阈值覆盖")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    parser.add_argument("--version", action="version", version="image_dedup 2.0")

    args = parser.parse_args()
    cli = {k: v for k, v in vars(args).items() if v is not None}

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    cfg = resolve_config(args.config, cli)
    scan_dir = cfg["scan"]["directory"]

    if not scan_dir:
        scan_dir = os.getcwd()
        cfg["scan"]["directory"] = scan_dir

    if not os.path.isdir(scan_dir):
        print(f"[ERROR] Directory not found: {scan_dir}")
        sys.exit(1)

    output_dir = cfg["report"].get("output_dir", "dedup_output")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(scan_dir), output_dir)

    if cfg["report"].get("clean_output", True) and os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)

    infos, matches, _ = run_detection(scan_dir, cfg)

    if not infos:
        print("[WARN] No images found. Check file extensions.")
        return

    if not matches:
        print("No suspicious duplicates found.")
        index_path = os.path.join(output_dir, "report", "index.html")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        from viz.report_html import _build_html
        html = _build_html([], scan_dir, len(infos),
                           len(infos) * (len(infos) - 1) // 2,
                           {"critical": 0, "high": 0, "medium": 0},
                           {}, cfg)
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Report: {index_path}")
        return

    max_vis = len(matches)

    print(f"Rendering all {max_vis} visualization images...")
    vis_dir = os.path.join(output_dir, "report", "vis")
    os.makedirs(vis_dir, exist_ok=True)

    for i in range(max_vis):
        m = matches[i]
        vis_fn = f"{i + 1:04d}_{m.severity.upper()[:3]}_{os.path.basename(m.image1)[:20]}_vs_{os.path.basename(m.image2)[:20]}.jpg"
        vis_path = os.path.join(vis_dir, vis_fn)
        try:
            annotate_pair(m, vis_path, cfg["report"].get("thumbnail_height", 500))
        except Exception as e:
            if not args.quiet:
                print(f"  [WARN] vis failed for pair {i + 1}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  Vis progress: {i + 1}/{max_vis}", end='\r')

    print()
    print("Generating report...")
    index_path, csv_path = generate_report(
        matches, infos, output_dir, scan_dir, cfg, vis_dir="vis"
    )
    print(f"Report: {index_path}")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
