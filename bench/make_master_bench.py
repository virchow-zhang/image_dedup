#!/usr/bin/env python3
"""
把 master 线的合成测试集（tests/gen_synthetic.py + manifest.json）
转换成 main 线基准框架能评分的 ground_truth.json。

为什么要转换而不是直接用它的 eval.py：
  1. 它的 manifest 只登记 (base, 各变换) 对，但**同一基底的变换之间**
     （rotate90_0000 ↔ flip_h_0000）同样是真重复。若按"不在正样本里就算误报"
     评分，会把真检出错判成误报，对任何检测器都不公平。
     → 这里取同一基底分组内的**传递闭包**作为正样本。
  2. splice_XXXX 是 base_i 与 base_j **共享一块区域**的产物，
     它与 base_i 是正样本；但它与 base_i 的其他变换之间语义模糊
     （只共享局部），既不该算正样本也不该算误报。
     → 这类对放进 ignore 列表，评分时既不计 TP 也不计 FP。
  3. 它的负样本只有"随机基底对"（彼此长得就不像），**没有难负样本**，
     因此绝对指标会明显高于 main 线那套刻意加难的基准。两次结果不可直接横比，
     但**同一数据集上不同检测器之间**可以横比。

用法:
    python bench/make_master_bench.py --out bench/data_master --baselines 30
"""

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO.parent / 'image_dedup_master'

sys.path.insert(0, str(REPO / 'bench'))
from _console import fix_console_encoding  # noqa: E402

fix_console_encoding()

TRANSFORMS = ["exact_copy", "rotate90", "rotate180", "rotate270",
              "flip_h", "flip_v", "rot_free", "scale_07", "crop_60",
              "brightness_70", "contrast_150", "jpeg_blur", "copy_move"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='bench/data_master')
    ap.add_argument('--baselines', type=int, default=30)
    args = ap.parse_args()

    out = (REPO / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    # raw 必须放在被评分的目录树**之外**：检测器会递归扫描 data_dir，
    # 若 raw/ 与 cross/all/ 都在里面，每张图会出现两次，
    # 凭空产生一整批"副本互相重复"的假象（实测让 master 的误报从 ~0 变成 9887）。
    raw = out.parent / f'{out.name}_raw'

    # 1) 调用 master 自己的生成器，保证数据集与它 README 里那份一致
    print(f'[1/3] 调用 master 生成器 → {raw}')
    gen = MASTER / 'tests' / 'gen_synthetic.py'
    if not gen.exists():
        print(f'缺少 {gen}；先 git worktree add ../image_dedup_master archive/master-v3-2026-08')
        return 1
    r = subprocess.run([sys.executable, str(gen), '--out', str(raw),
                        '--baselines', str(args.baselines)],
                       cwd=str(MASTER), capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    print(r.stdout.strip() or r.stderr.strip()[-500:])

    manifest = json.loads((raw / 'manifest.json').read_text(encoding='utf-8'))

    # 2) 重组正样本 / ignore
    print('[2/3] 重组评分用的正样本与 ignore 集')
    groups = {}                      # baseline idx -> [文件名]
    splice_of = {}
    for f in manifest['files']:
        if f.startswith('base_'):
            groups.setdefault(f.split('_')[1].split('.')[0], []).append(f)
        elif f.startswith('splice_'):
            splice_of[f] = f.split('_')[1].split('.')[0]
        else:
            idx = f.rsplit('_', 1)[1].split('.')[0]
            groups.setdefault(idx, []).append(f)

    positives, ignore = [], []
    for idx, files in groups.items():
        # 按「两个文件各自的变换名」给正样本打标签，才能看出漏在哪种组合上
        def tname(f):
            if f.startswith('base_'):
                return 'base'
            return f.rsplit('_', 1)[0]
        for a, b in combinations(sorted(files), 2):
            positives.append({'a': f'cross/all/{a}', 'b': f'cross/all/{b}',
                              'transform': f'{tname(a)}|{tname(b)}', 'case': idx,
                              'generator': 'master_synthetic', 'naming': 'master'})
        sp = f'splice_{idx}.png'
        if sp in splice_of and sp in manifest['files']:
            base = f'base_{idx}.png'
            positives.append({'a': f'cross/all/{base}', 'b': f'cross/all/{sp}',
                              'transform': 'splice', 'case': idx,
                              'generator': 'master_synthetic', 'naming': 'master'})
            # splice 与该基底的其他变换：语义模糊，排除评分
            for t in TRANSFORMS:
                tf = f'{t}_{idx}.png'
                if tf in manifest['files']:
                    # 注意用 tf（带序号），写成 t 会得到 'cross/all/contrast_150'
                    # 这种不存在的路径，ignore 全部失效，约 330 个同基底对
                    # 会被误记成误报
                    ignore.append(sorted([f'cross/all/{tf}', f'cross/all/{sp}']))

    # 3) 落盘成 main 线基准格式
    dstdir = out / 'cross' / 'all'
    dstdir.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in manifest['files']:
        src = raw / f
        if src.exists():
            shutil.copy2(src, dstdir / f)

    files = {}
    for f in manifest['files']:
        files[f'cross/all/{f}'] = {'case': f.rsplit('_', 1)[1].split('.')[0],
                                   'generator': 'master_synthetic', 'role': 'member'}
    gt = {'seed': 20260701,
          'cross': {'cases': [{'id': k, 'dir': 'cross/all',
                               'files': sorted(v)} for k, v in groups.items()],
                    'positives': positives, 'files': files, 'ignore': ignore},
          'intra': {'figures': []}}
    (out / 'ground_truth.json').write_text(
        json.dumps(gt, indent=1, ensure_ascii=False), encoding='utf-8')

    print(f'[3/3] → {out}/ground_truth.json')
    print(f'  文件 {len(manifest["files"])} 张，正样本对 {len(positives)}，'
          f'ignore 对 {len(ignore)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
