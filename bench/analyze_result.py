#!/usr/bin/env python3
"""分析某次基准结果里的误报构成，并对比两条线的能力覆盖。"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _console import fix_console_encoding  # noqa: E402

fix_console_encoding()

p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding='utf-8'))
c = d['cross']
print(f"=== {d['label']} ===")
print(f"cross P={c['overall']['precision']:.3f} R={c['overall']['recall']:.3f} "
      f"F1={c['overall']['f1']:.3f}  TP={c['overall']['tp']} FP={c['overall']['fp']} "
      f"FN={c['overall']['fn']}")

fps = c['false_positives']
print(f"\n误报 {len(fps)} 对。按「是否同一 case」分类：")

same_case = 0
diff_case = 0
for pair in fps:
    a, b = pair
    ca = a.split('/')[1] if a.count('/') > 1 else a
    cb = b.split('/')[1] if b.count('/') > 1 else b
    if ca == cb:
        same_case += 1
    else:
        diff_case += 1
print(f"  同一 case（同风格难负样本）: {same_case}")
print(f"  跨 case（完全不同的实验）  : {diff_case}")

print("\n漏报明细：")
for m in c['missed_positives']:
    print(f"  ✗ {m['transform']:10s} {m['naming']:14s} {m['a']} <-> {m['b']}")

print("\n按变换召回：")
for k, v in c['by_transform'].items():
    if v['hit'] < v['total']:
        print(f"  {k:12s} {v['hit']}/{v['total']}")

i = d['intra']
print(f"\nintra 支持={i['capability']} region P={i['region']['precision']:.3f} "
      f"R={i['region']['recall']:.3f} F1={i['region']['f1']:.3f} "
      f"TP={i['region']['tp']} FP={i['region']['fp']} FN={i['region']['fn']}")
if i['by_relation']:
    for k, v in i['by_relation'].items():
        print(f"  {k:24s} {v['hit']}/{v['total']}  IoU={v['mean_iou']}")
