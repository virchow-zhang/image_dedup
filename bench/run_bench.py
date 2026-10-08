#!/usr/bin/env python3
"""
基准测试框架
=============
在合成 ground-truth 数据集上评测检测器，输出多维度指标。

评测任务：
  A) cross  —— 跨文件查重：pair-level Precision / Recall / F1
  B) intra  —— 图内 panel 复用：region-level P/R/F1 + 定位精度(IoU) + figure-level P/R

用法:
    python bench/run_bench.py --detector v2 --out bench/results/v2.json
    python bench/run_bench.py --detector v3 --out bench/results/v3.json
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _console import fix_console_encoding  # noqa: E402

fix_console_encoding()

REPO = Path(__file__).resolve().parent.parent
IOU_MATCH_THRESHOLD = 0.5


# ---------------------------------------------------------------- 工具


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': round(p, 4),
            'recall': round(r, 4), 'f1': round(f, 4)}


@contextlib.contextmanager
def capture_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


# ---------------------------------------------------------------- v2 检测器


def run_v2(data_dir: Path, workers: int = 8, hash_size: int = 16, **kw):
    """调用 legacy/v2 的 image_dedup_optimized.py"""
    import psutil

    v2 = load_module(REPO / 'legacy' / 'v2' / 'image_dedup_optimized.py', 'dedup_v2')
    cross_root = data_dir / 'cross'

    proc = psutil.Process()
    mem0 = proc.memory_info().rss
    peak = [mem0]

    t0 = time.perf_counter()
    with capture_stdout() as log:
        images, stats = v2.scan_directory_parallel(str(cross_root), hash_size, workers)
        t_load = time.perf_counter() - t0
        peak.append(proc.memory_info().rss)

        t1 = time.perf_counter()
        matches = v2.find_duplicates_optimized(
            images, stats, pthresh=5, sthresh=0.92, hthresh=0.80,
            check_rotations=True, use_lsh=True, min_votes=2,
            orb_confirm=True, orb_strict=False, min_std=8.0,
            hash_size=hash_size, check_subimage=True)
        matches = v2.deduplicate_matches(matches)
        t_detect = time.perf_counter() - t1
        peak.append(proc.memory_info().rss)

    pairs = set()
    for m in matches:
        pairs.add(frozenset((Path(m.image1).resolve().relative_to(data_dir.resolve()).as_posix(),
                             Path(m.image2).resolve().relative_to(data_dir.resolve()).as_posix())))

    return {
        'detector': 'v2',
        'cross_pairs': [sorted(p) for p in pairs],
        'intra_findings': [],
        'timings': {'load_s': round(t_load, 2), 'detect_s': round(t_detect, 2),
                    'total_s': round(t_load + t_detect, 2)},
        'memory': {'rss_start_mb': round(mem0 / 2**20, 1),
                   'rss_peak_mb': round(max(peak) / 2**20, 1)},
        'n_images': len(images),
        'log': log.getvalue(),
    }


# ---------------------------------------------------------------- v3 检测器


def run_v3(data_dir: Path, workers: int = 8, **kw):
    """v3：向量化跨文件查重 + 图内 panel 复用检测"""
    import psutil
    sys.path.insert(0, str(REPO))
    from dedup.crossfile import find_cross_duplicates, scan_directory
    from dedup.features import find_region_matches
    from dedup.panels import segment_panels

    import cv2

    proc = psutil.Process()
    mem0 = proc.memory_info().rss
    peak = [mem0]

    t0 = time.perf_counter()
    recs = scan_directory(str(data_dir / 'cross'), workers=workers)
    t_load = time.perf_counter() - t0
    peak.append(proc.memory_info().rss)

    t1 = time.perf_counter()
    matches = find_cross_duplicates(recs)
    t_detect = time.perf_counter() - t1
    peak.append(proc.memory_info().rss)

    dres = data_dir.resolve()
    cross_pairs = []
    for m in matches:
        try:
            a = Path(m.image1).resolve().relative_to(dres).as_posix()
            b = Path(m.image2).resolve().relative_to(dres).as_posix()
        except ValueError:
            continue
        cross_pairs.append(sorted([a, b]))

    # ---- 图内 panel 复用 ----
    t2 = time.perf_counter()
    findings = []
    for fig in sorted((data_dir / 'intra').glob('*')):
        if not fig.is_file():
            continue
        img = cv2.imread(str(fig))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        panels = segment_panels(img)

        def _panel_of(bbox):
            cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            for p in panels:
                if p.x <= cx <= p.x + p.w and p.y <= cy <= p.y + p.h:
                    return p.index
            return None

        for m in find_region_matches(gray):
            findings.append({'figure': f'intra/{fig.name}',
                             'a_bbox': m.a_bbox, 'b_bbox': m.b_bbox,
                             'transform': m.transform, 'ncc': m.ncc,
                             'n_inliers': m.n_inliers, 'score': m.score,
                             'a_panel': _panel_of(m.a_bbox),
                             'b_panel': _panel_of(m.b_bbox)})
    t_intra = time.perf_counter() - t2
    peak.append(proc.memory_info().rss)

    return {
        'detector': 'v3',
        'cross_pairs': cross_pairs,
        'intra_findings': findings,
        'intra_supported': True,
        'timings': {'load_s': round(t_load, 2), 'detect_s': round(t_detect, 2),
                    'intra_s': round(t_intra, 2),
                    'total_s': round(t_load + t_detect + t_intra, 2)},
        'memory': {'rss_start_mb': round(mem0 / 2**20, 1),
                   'rss_peak_mb': round(max(peak) / 2**20, 1)},
        'n_images': len(recs),
        'log': '',
    }


# ---------------------------------------------------------------- 评测


def score_cross(res, gt, data_dir):
    pred = {frozenset(p) for p in res['cross_pairs']}
    pos_list = gt['cross']['positives']
    pos = {frozenset((p['a'], p['b'])): p for p in pos_list}

    tp = pred & set(pos)
    fp = pred - set(pos)
    fn = set(pos) - pred

    overall = prf(len(tp), len(fp), len(fn))

    # 按变换类型细分召回
    by_transform = {}
    for p in pos_list:
        key = frozenset((p['a'], p['b']))
        rec = by_transform.setdefault(p['transform'], {'hit': 0, 'total': 0})
        rec['total'] += 1
        if key in pred:
            rec['hit'] += 1
    by_transform = {k: {'hit': v['hit'], 'total': v['total'],
                        'recall': round(v['hit'] / v['total'], 3)}
                    for k, v in sorted(by_transform.items())}

    by_naming = {}
    for p in pos_list:
        key = frozenset((p['a'], p['b']))
        rec = by_naming.setdefault(p['naming'], {'hit': 0, 'total': 0})
        rec['total'] += 1
        if key in pred:
            rec['hit'] += 1
    by_naming = {k: {'hit': v['hit'], 'total': v['total'],
                     'recall': round(v['hit'] / v['total'], 3)}
                 for k, v in sorted(by_naming.items())}

    missed = [{'transform': p['transform'], 'naming': p['naming'],
               'a': Path(p['a']).name, 'b': Path(p['b']).name}
              for p in pos_list if frozenset((p['a'], p['b'])) not in pred]
    false_pos = [sorted(x) for x in fp]

    return {'overall': overall, 'by_transform': by_transform, 'by_naming': by_naming,
            'missed_positives': missed, 'false_positives': false_pos}


def score_intra(res, gt):
    figs = {f['file']: f for f in gt['intra']['figures']}
    preds = {}
    for f in res.get('intra_findings', []):
        preds.setdefault(f['figure'], []).append(f)

    # ---- region level ----
    tp = fp = fn = 0
    matched_ious, center_errs = [], []
    per_relation = {}
    fig_hit = 0

    for fname, fig in figs.items():
        gt_reuse = fig['reuse']
        plist = list(preds.get(fname, []))
        used = [False] * len(plist)
        rel = fig['relation']
        rec = per_relation.setdefault(rel, {'hit': 0, 'total': 0, 'iou': []})
        rec['total'] += len(gt_reuse)

        for g in gt_reuse:
            best, best_i = None, -1
            for i, p in enumerate(plist):
                if used[i]:
                    continue
                same = iou(p['a_bbox'], g['a_bbox']) >= IOU_MATCH_THRESHOLD and \
                    iou(p['b_bbox'], g['b_bbox']) >= IOU_MATCH_THRESHOLD
                swap = iou(p['a_bbox'], g['b_bbox']) >= IOU_MATCH_THRESHOLD and \
                    iou(p['b_bbox'], g['a_bbox']) >= IOU_MATCH_THRESHOLD
                if same or swap:
                    score = max(iou(p['a_bbox'], g['a_bbox']) + iou(p['b_bbox'], g['b_bbox']),
                                iou(p['a_bbox'], g['b_bbox']) + iou(p['b_bbox'], g['a_bbox']))
                    if best is None or score > best:
                        best, best_i = score, i
            if best_i >= 0:
                used[best_i] = True
                tp += 1
                rec['hit'] += 1
                p = plist[best_i]
                a_iou = max(iou(p['a_bbox'], g['a_bbox']), iou(p['a_bbox'], g['b_bbox']))
                b_iou = max(iou(p['b_bbox'], g['b_bbox']), iou(p['b_bbox'], g['a_bbox']))
                matched_ious.append((a_iou + b_iou) / 2)
                rec['iou'].append((a_iou + b_iou) / 2)
                ca = (p['a_bbox'][0] + p['a_bbox'][2] / 2, p['a_bbox'][1] + p['a_bbox'][3] / 2)
                ga = (g['a_bbox'][0] + g['a_bbox'][2] / 2, g['a_bbox'][1] + g['a_bbox'][3] / 2)
                center_errs.append(float(np.hypot(ca[0] - ga[0], ca[1] - ga[1])))
            else:
                fn += 1

        fp += sum(1 for u in used if not u)
        if gt_reuse and any(used):
            fig_hit += 1

    region = prf(tp, fp, fn)
    per_relation = {k: {'hit': v['hit'], 'total': v['total'],
                        'recall': round(v['hit'] / v['total'], 3),
                        'mean_iou': round(float(np.mean(v['iou'])), 3) if v['iou'] else 0.0}
                    for k, v in sorted(per_relation.items())}

    # ---- figure level ----
    pos_figs = [f for f in figs.values() if f['reuse']]
    fig_tp = sum(1 for f in pos_figs if preds.get(f['file']))
    fig_fp = sum(1 for f in figs.values() if not f['reuse'] and preds.get(f['file']))
    fig_fn = len(pos_figs) - fig_tp
    figure = prf(fig_tp, fig_fp, fig_fn)

    return {
        'region': region,
        'figure': figure,
        'mean_iou': round(float(np.mean(matched_ious)), 3) if matched_ious else 0.0,
        'median_center_err_px': round(float(np.median(center_errs)), 1) if center_errs else None,
        'by_relation': per_relation,
        'capability': bool(res.get('intra_findings')) or res.get('intra_supported', False),
    }


# ---------------------------------------------------------------- main


DETECTORS = {'v2': run_v2, 'v3': run_v3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', default='v2')
    ap.add_argument('--data', default='bench/data')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()

    data_dir = (REPO / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    gt = json.loads((data_dir / 'ground_truth.json').read_text(encoding='utf-8'))

    if args.detector not in DETECTORS:
        print(f'未知检测器 {args.detector}；已注册: {list(DETECTORS)}')
        sys.exit(1)

    print(f'[·] 运行检测器 {args.detector} @ {data_dir}')
    t0 = time.perf_counter()
    try:
        res = DETECTORS[args.detector](data_dir, workers=args.workers)
        err = None
    except Exception:
        err = traceback.format_exc()
        res = {'detector': args.detector, 'cross_pairs': [], 'intra_findings': [],
               'timings': {}, 'memory': {}, 'log': ''}
    wall = time.perf_counter() - t0

    out = {'detector': args.detector, 'label': args.label or args.detector,
           'data': str(data_dir), 'wall_s': round(wall, 2),
           'timings': res.get('timings', {}), 'memory': res.get('memory', {}),
           'n_images': res.get('n_images'), 'error': err}

    out['cross'] = score_cross(res, gt, data_dir)
    out['intra'] = score_intra(res, gt)

    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
        logp = p.with_suffix('.log')
        logp.write_text(res.get('log', '') + (f'\n\n=== ERROR ===\n{err}' if err else ''),
                        encoding='utf-8')
        print(f'[✓] 结果 -> {p}')

    c, i = out['cross'], out['intra']
    print(f'\n{"="*62}\n  检测器: {out["label"]}   总耗时: {wall:.1f}s\n{"="*62}')
    print(f'  [cross] P={c["overall"]["precision"]:.3f} R={c["overall"]["recall"]:.3f} '
          f'F1={c["overall"]["f1"]:.3f}  (TP={c["overall"]["tp"]} FP={c["overall"]["fp"]} '
          f'FN={c["overall"]["fn"]})')
    print(f'  [intra] region P={i["region"]["precision"]:.3f} R={i["region"]["recall"]:.3f} '
          f'F1={i["region"]["f1"]:.3f} | figure P={i["figure"]["precision"]:.3f} '
          f'R={i["figure"]["recall"]:.3f} | meanIoU={i["mean_iou"]:.3f}'
          + ('' if i['capability'] else '   ⚠ 不支持'))
    print(f'  耗时: load={out["timings"].get("load_s")}s detect={out["timings"].get("detect_s")}s')
    print(f'  内存峰值: {out["memory"].get("rss_peak_mb")} MB')
    if err:
        print(f'  ⚠ 检测器异常:\n{err[-1500:]}')

    import threading
    alive = [t for t in threading.enumerate() if t is not threading.main_thread()]
    if alive:
        print(f'\n  [diag] 退出时仍有 {len(alive)} 个非主线程存活:')
        for t in alive:
            print(f'    - {t.name}  daemon={t.daemon} alive={t.is_alive()}')


if __name__ == '__main__':
    main()
    # 载入 cv2/numpy/torch 等重型 C 扩展后，解释器 finalize 阶段可能停滞在
    # 某个 C 析构器上（实测：所有输出已落盘，进程却不退出）。
    # 基准框架只关心测量结果，落盘后直接硬退出，避免污染计时与占用进程。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
