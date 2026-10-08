"""
命令行入口（v3）
================
三个模式：
  cross  跨文件查重（同一张图是否在多处出现）
  intra  图内复用（同一张组图里 panel 之间是否重叠/复用/复制粘贴）
  both   两者都跑（默认）

用法：
    python image_dedup_v3.py <目录>
    python image_dedup_v3.py <目录> --mode intra
    python image_dedup_v3.py <目录> --mode both --report report.html
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import report as report_mod
from .crossfile import find_cross_duplicates, scan_directory
from .features import find_region_matches
from .panels import segment_panels

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
             '.gif', '.webp', '.svs', '.ndpi', '.vsi'}


def _fix_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


def _panel_of(panels, bbox):
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    for p in panels:
        if p.x <= cx <= p.x + p.w and p.y <= cy <= p.y + p.h:
            return p.index
    return None


def analyze_intra(root: Path, min_inliers: int, ncc_threshold: float,
                  verbose: bool = True):
    """对目录下每张图做组图切分 + 图内复用检测"""
    files = sorted(p for p in root.rglob('*')
                   if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    results = []
    for i, path in enumerate(files, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        panels = segment_panels(img)
        raw = find_region_matches(gray, min_inliers=min_inliers,
                                  ncc_threshold=ncc_threshold)
        findings = []
        for m in raw:
            findings.append({
                'figure': str(path), 'a_bbox': m.a_bbox, 'b_bbox': m.b_bbox,
                'transform': m.transform, 'ncc': m.ncc,
                'n_inliers': m.n_inliers, 'score': m.score,
                'a_panel': _panel_of(panels, m.a_bbox),
                'b_panel': _panel_of(panels, m.b_bbox),
            })
        results.append({'path': str(path), 'panels': panels, 'findings': findings})
        if verbose and findings:
            for f in findings:
                pa = f'P{f["a_panel"]}' if f['a_panel'] is not None else '?'
                pb = f'P{f["b_panel"]}' if f['b_panel'] is not None else '?'
                print(f'  [{i}/{len(files)}] {path.name}: {pa}↔{pb} '
                      f'{f["transform"]} 相似度={f["ncc"]:.3f} 内点={f["n_inliers"]}')
    return results


def main(argv=None):
    _fix_console()
    ap = argparse.ArgumentParser(
        prog='image_dedup_v3',
        description='科研图片查重工具 v3 —— 跨文件查重 + 图内 panel 复用检测')
    ap.add_argument('directory', nargs='?', default='.', help='要扫描的目录')
    ap.add_argument('--mode', choices=['cross', 'intra', 'both'], default='both')
    ap.add_argument('--report', default=None, help='HTML 报告输出路径')
    ap.add_argument('--json', default=None, help='JSON 结果输出路径')
    ap.add_argument('--workers', type=int, default=min(16, (os.cpu_count() or 4)))
    ap.add_argument('--hash-threshold', type=int, default=12,
                    help='pHash 差异阈值（256 位），越小越严格')
    ap.add_argument('--min-votes', type=int, default=2,
                    help='跨文件投票：至少几种结构检测器通过')
    ap.add_argument('--min-inliers', type=int, default=14,
                    help='图内匹配：RANSAC 最少内点数')
    ap.add_argument('--ncc-threshold', type=float, default=0.55,
                    help='图内匹配：归一化互相关验证阈值')
    ap.add_argument('--no-thumbnails', action='store_true',
                    help='报告不内嵌图片（报告很小，但看不到图）')
    ap.add_argument('--no-subimage', action='store_true',
                    help='关闭子图/裁剪检测（金字塔模板匹配）')
    ap.add_argument('--subimage-threshold', type=float, default=0.85,
                    help='子图/裁剪相关系数阈值')
    ap.add_argument('--no-edge', action='store_true',
                    help='关闭边缘重叠/拼接检测')
    ap.add_argument('--edge-threshold', type=float, default=0.55,
                    help='边缘重叠得分阈值')
    args = ap.parse_args(argv)

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f'错误: 目录不存在: {root}')
        return 1

    t_start = time.perf_counter()
    print('=' * 64)
    print('  🔬 科研图片查重工具 v3')
    print(f'  扫描目录: {root}')
    print(f'  模式: {args.mode}')
    print('=' * 64)

    cross_matches, recs = [], []
    if args.mode in ('cross', 'both'):
        t0 = time.perf_counter()
        recs = scan_directory(str(root), workers=args.workers)
        print(f'[1] 载入 {len(recs)} 张图片 ({time.perf_counter()-t0:.1f}s)')
        t1 = time.perf_counter()
        cross_matches = find_cross_duplicates(
            recs, phash_threshold=args.hash_threshold,
            min_votes=args.min_votes, verbose=True,
            detect_subimage=not args.no_subimage,
            subimage_threshold=args.subimage_threshold,
            detect_edge=not args.no_edge,
            edge_threshold=args.edge_threshold)
        print(f'[2] 跨文件查重: {len(cross_matches)} 对 '
              f'({time.perf_counter()-t1:.1f}s)')

    intra_results = []
    if args.mode in ('intra', 'both'):
        t2 = time.perf_counter()
        intra_results = analyze_intra(root, args.min_inliers, args.ncc_threshold)
        n_find = sum(len(r['findings']) for r in intra_results)
        print(f'[3] 图内复用检测: {n_find} 处 '
              f'({time.perf_counter()-t2:.1f}s)')

    elapsed = time.perf_counter() - t_start

    # ---- 报告 ----
    report_path = args.report
    if report_path is None:
        report_path = str(root / 'image_dedup_v3_report.html')
    if not args.no_thumbnails:
        cross_items = []
        for m in cross_matches:
            it = report_mod.build_cross_item(m)
            if it:
                cross_items.append(it)
        intra_items = []
        for r in intra_results:
            if not r['findings']:
                continue
            for f in r['findings']:
                it = report_mod.build_intra_item(r['path'], f, r['panels'])
                if it:
                    intra_items.append(it)
        report_mod.generate_report(intra_items, cross_items, report_path,
                                   {'directory': str(root), 'n_images': len(recs),
                                    'elapsed': round(elapsed, 2)})
        print(f'[4] HTML 报告: {report_path}')

    if args.json:
        import json as _json
        payload = {
            'directory': str(root), 'elapsed': round(elapsed, 3),
            'n_images': len(recs),
            'cross': [{'image1': m.image1, 'image2': m.image2,
                       'severity': m.severity, 'similarity': m.similarity,
                       'confidence': m.confidence, 'match_type': m.match_type,
                       'details': m.details} for m in cross_matches],
            'intra': [{'figure': r['path'],
                       'panels': [p.bbox for p in r['panels']],
                       'findings': r['findings']} for r in intra_results
                      if r['findings']],
        }
        Path(args.json).write_text(_json.dumps(payload, indent=1, ensure_ascii=False),
                                   encoding='utf-8')
        print(f'[5] JSON 结果: {args.json}')

    print('=' * 64)
    print(f'  完成，耗时 {elapsed:.1f}s')
    print(f'  图内复用/重叠: {sum(len(r["findings"]) for r in intra_results)} 处')
    print(f'  跨文件重复:   {len(cross_matches)} 对')
    print('=' * 64)
    return 0
