#!/usr/bin/env python3
"""基准结果查看器：打印单个结果文件的多维度明细。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _console import fix_console_encoding  # noqa: E402

fix_console_encoding()


def show(path, verbose=True):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    print('=' * 66)
    print(f"  检测器: {d['label']}   总耗时: {d.get('wall_s')}s")
    print('=' * 66)
    if d.get('error'):
        print('  ⚠ ERROR:\n' + d['error'][-1200:])

    c = d['cross']
    o = c['overall']
    print(f"\n[cross 跨文件查重]  P={o['precision']:.3f}  R={o['recall']:.3f}  F1={o['f1']:.3f}"
          f"   (TP={o['tp']} FP={o['fp']} FN={o['fn']})")
    print('  按变换类型召回:')
    for k, v in c['by_transform'].items():
        bar = '█' * int(v['recall'] * 10) + '·' * (10 - int(v['recall'] * 10))
        print(f"    {k:12s} {bar} {v['hit']}/{v['total']}")
    print('  按命名风格召回:')
    for k, v in c['by_naming'].items():
        print(f"    {k:14s} {v['hit']}/{v['total']}  ({v['recall']:.0%})")
    if verbose:
        if c['missed_positives']:
            print('  漏报明细:')
            for m in c['missed_positives']:
                print(f"    ✗ {m['transform']:10s} {m['naming']:14s} {m['a']} <-> {m['b']}")
        if c['false_positives']:
            print(f"  误报 {len(c['false_positives'])} 对:")
            for p in c['false_positives'][:12]:
                print(f"    ✗ {p[0]} <-> {p[1]}")

    i = d['intra']
    print(f"\n[intra 图内 panel 复用]  支持={i['capability']}")
    r, f = i['region'], i['figure']
    print(f"  区域级  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}"
          f"  (TP={r['tp']} FP={r['fp']} FN={r['fn']})")
    print(f"  图像级  P={f['precision']:.3f} R={f['recall']:.3f} F1={f['f1']:.3f}")
    print(f"  平均 IoU={i['mean_iou']:.3f}   中位中心误差={i['median_center_err_px']} px")
    if i['by_relation']:
        print('  按复用类型:')
        for k, v in i['by_relation'].items():
            print(f"    {k:24s} {v['hit']}/{v['total']}  IoU={v['mean_iou']}")

    print(f"\n[性能]  load={d['timings'].get('load_s')}s  detect={d['timings'].get('detect_s')}s"
          f"  峰值内存={d['memory'].get('rss_peak_mb')} MB  图片数={d.get('n_images')}")


if __name__ == '__main__':
    show(sys.argv[1] if len(sys.argv) > 1 else 'bench/results/v2_baseline.json')
