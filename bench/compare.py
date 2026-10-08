#!/usr/bin/env python3
"""
新旧方案多维度对比
==================
读取若干基准结果 JSON，输出 Markdown 对比表（准确率 / 召回 / 定位 / 性能 / 能力）。

用法：
    python bench/compare.py --results bench/results/v2_baseline.json bench/results/v3.json \
        --title "120 张合成图" --out bench/results/compare_small.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _console import fix_console_encoding  # noqa: E402

fix_console_encoding()


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def row(d):
    c, i = d['cross'], d['intra']
    return {
        'label': d.get('label') or d['detector'],
        'wall': d.get('wall_s'),
        'load': d['timings'].get('load_s'),
        'detect': d['timings'].get('detect_s'),
        'mem': d['memory'].get('rss_peak_mb'),
        'n': d.get('n_images'),
        'c_p': c['overall']['precision'], 'c_r': c['overall']['recall'],
        'c_f1': c['overall']['f1'],
        'c_tp': c['overall']['tp'], 'c_fp': c['overall']['fp'], 'c_fn': c['overall']['fn'],
        'i_p': i['region']['precision'], 'i_r': i['region']['recall'],
        'i_f1': i['region']['f1'], 'i_iou': i['mean_iou'],
        'i_fig_f1': i['figure']['f1'],
        'i_ok': i['capability'],
    }


def table(rows, title):
    hdr = ['指标'] + [r['label'] for r in rows]
    lines = [f'### {title}', '',
             '| ' + ' | '.join(hdr) + ' |',
             '|' + '---|' * len(hdr)]

    def add(name, fn, fmt='{}'):
        lines.append('| ' + name + ' | ' +
                     ' | '.join(fmt.format(fn(r)) for r in rows) + ' |')

    add('图片数', lambda r: r['n'] or 0)
    add('**跨文件查重**', lambda r: '')
    add('精确率 P', lambda r: r['c_p'], '{:.3f}')
    add('召回率 R', lambda r: r['c_r'], '{:.3f}')
    add('F1', lambda r: r['c_f1'], '{:.3f}')
    add('TP / FP / FN', lambda r: f"{r['c_tp']} / {r['c_fp']} / {r['c_fn']}")
    add('**图内复用检测**', lambda r: '')
    add('是否支持', lambda r: '✅' if r['i_ok'] else '❌ 无此能力')
    add('区域 P / R / F1', lambda r: f"{r['i_p']:.3f} / {r['i_r']:.3f} / {r['i_f1']:.3f}"
        if r['i_ok'] else '—')
    add('定位平均 IoU', lambda r: f"{r['i_iou']:.3f}" if r['i_ok'] else '—')
    add('图像级 F1', lambda r: f"{r['i_fig_f1']:.3f}" if r['i_ok'] else '—')
    add('**性能**', lambda r: '')
    add('总耗时 (s)', lambda r: r['wall'], '{:.1f}')
    add('其中载入 (s)', lambda r: r['load'], '{:.2f}')
    add('其中检测 (s)', lambda r: r['detect'], '{:.2f}')
    add('峰值内存 (MB)', lambda r: r['mem'], '{:.1f}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--title', default='对比')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = [row(load(p)) for p in args.results]
    md = table(rows, args.title)
    print(md)
    if args.out:
        Path(args.out).write_text(md + '\n', encoding='utf-8')
        print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()
