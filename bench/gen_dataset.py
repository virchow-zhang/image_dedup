#!/usr/bin/env python3
"""
合成基准数据集生成器 (v3 benchmark)
=====================================
生成带 ground-truth 标注的两类数据集：

  A) cross/  —— 跨文件查重（v2 的主战场）
     每个 case: 1 张 base + 1 张变体(单一变换) + K 张同实验难负样本
     正样本对 = base↔变体；同 case 内其余对 + 跨 case 全部为负样本

  B) intra/  —— 单张组图内的 panel 复用/重叠（v3 新增能力）
     每个 figure: rows×cols 网格 panel，注入已知的复用/重叠/图内复制

所有随机性由固定种子驱动，结果完全可复现。
输出 data/ 目录 + ground_truth.json

用法:
    python bench/gen_dataset.py --out bench/data
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from _console import fix_console_encoding

fix_console_encoding()

SEED = 20261007

# ---------------------------------------------------------------- 基础工具


def _gauss_mask(h, w, cx, cy, sx, sy, theta=0.0):
    """椭圆高斯掩膜，用于模拟印迹条带 / 荧光斑点 / 细胞核"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy
    return np.exp(-0.5 * ((xr / max(sx, .5)) ** 2 + (yr / max(sy, .5)) ** 2)).astype(np.float32)


def _fractal_noise(h, w, rng, octaves=6):
    """多倍频噪声，用于生成组织切片类纹理"""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        res = 2 ** (o + 1)
        small = (rng.random((res, res)) * 255).astype(np.uint8)
        up = np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC), np.float32) / 255.
        out += amp * up
        total += amp
        amp *= 0.55
    return out / total


def _jpeg_roundtrip(img, quality):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img.copy()
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- 内容生成器
# 每个生成器返回 HxWx3 uint8 RGB，模拟一类科研图


def gen_blot(h, w, rng):
    """Western blot：浅背景 + 深色条带（锐利核心 + 柔和晕圈）"""
    img = 242.0 + np.linspace(-8, 8, h, dtype=np.float32)[:, None] + rng.normal(0, 2.2, (h, w))
    n_lanes = int(rng.integers(4, 7))
    lane_w = w / n_lanes
    for i in range(n_lanes):
        cx = (i + 0.5) * lane_w + rng.uniform(-2, 2)
        # 泳道本底差异
        img[:, int(cx - lane_w * .45):int(cx + lane_w * .45)] += rng.uniform(-5, 5)
        for _ in range(int(rng.integers(1, 4))):
            cy = rng.uniform(0.15, 0.85) * h
            bw = lane_w * rng.uniform(0.32, 0.46)
            bh = h * rng.uniform(0.012, 0.026)
            amp = rng.uniform(85, 185)
            core = _gauss_mask(h, w, cx, cy, bw, bh)             # 锐利核心
            halo = _gauss_mask(h, w, cx, cy, bw * 1.25, bh * 2.6)  # 扩散晕圈
            img -= amp * (0.75 * core + 0.25 * halo)
    img += rng.normal(0, 2.0, (h, w))
    return np.clip(np.stack([img] * 3, -1), 0, 255).astype(np.uint8)


def gen_fluorescence(h, w, rng):
    """荧光显微：暗背景 + 彩色高斯斑点"""
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 6
    img[..., 1] = 9
    img[..., 2] = 7
    for _ in range(int(rng.integers(28, 62))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        s = rng.uniform(3.5, 10.0)
        m = _gauss_mask(h, w, cx, cy, s, s * rng.uniform(0.65, 1.4), rng.uniform(0, np.pi))
        img[..., int(rng.integers(0, 3))] += m * rng.uniform(90, 235)
    # 纤维状结构：低对比度、沿长度强度起伏的柔和纹理（而非硬直线特征）
    for _ in range(int(rng.integers(5, 12))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, np.pi)
        m = _gauss_mask(h, w, cx, cy, rng.uniform(.25, .55) * w, rng.uniform(3.0, 7.0), ang)
        mod = 0.5 + 0.5 * _fractal_noise(h, w, rng, octaves=4)
        img[..., 1] += m * mod * rng.uniform(22, 60)
    img += rng.normal(0, 2.5, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def gen_histology(h, w, rng):
    """组织切片：粉紫色分形纹理"""
    n1, n2 = _fractal_noise(h, w, rng), _fractal_noise(h, w, rng)
    img = np.stack([255 * (0.45 + 0.50 * n1),
                    255 * (0.22 + 0.42 * n2),
                    255 * (0.45 + 0.32 * n2)], -1).astype(np.float32)
    mask = (n1 > 0.42).astype(np.float32)
    img *= (0.80 + 0.20 * mask)[..., None]
    for _ in range(int(rng.integers(4, 12))):          # 细胞核
        m = _gauss_mask(h, w, rng.uniform(0, w), rng.uniform(0, h),
                        rng.uniform(2, 5), rng.uniform(2, 5))
        img -= m[..., None] * np.array([70, 40, 60], np.float32)
    img += rng.normal(0, 3.0, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def gen_gel(h, w, rng):
    """核酸胶：暗背景 + 亮条带"""
    img = 22.0 + rng.normal(0, 3.0, (h, w))
    n_lanes = int(rng.integers(5, 9))
    lane_w = w / n_lanes
    for i in range(n_lanes):
        cx = (i + 0.5) * lane_w
        img[:, int(cx - lane_w * .42):int(cx + lane_w * .42)] += 6
        for _ in range(int(rng.integers(1, 4))):
            cy = rng.uniform(0.2, 0.8) * h
            m = _gauss_mask(h, w, cx, cy, lane_w * rng.uniform(0.25, 0.40),
                            h * rng.uniform(0.015, 0.035))
            img += m * rng.uniform(120, 230)
    return np.clip(np.stack([img * .92, img, img * .88], -1), 0, 255).astype(np.uint8)


def gen_barplot(h, w, rng):
    """柱状图：白底 + 灰轴 + 彩柱 + 误差棒"""
    img = np.full((h, w, 3), 255, np.float32)
    ml, mr, mt, mb = int(w * .14), int(w * .06), int(h * .08), int(h * .16)
    cv2.line(img, (ml, mt), (ml, h - mb), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(img, (ml, h - mb), (w - mr, h - mb), (60, 60, 60), 1, cv2.LINE_AA)
    n = int(rng.integers(4, 8))
    slot = (w - mr - ml) / n
    palette = [(70, 110, 180), (200, 90, 80), (110, 170, 110), (180, 140, 70), (140, 110, 180)]
    for i in range(n):
        x0 = int(ml + i * slot + slot * .22)
        x1 = int(ml + i * slot + slot * .78)
        top = int(rng.uniform(mt + 8, h - mb - 10))
        cv2.rectangle(img, (x0, top), (x1, h - mb), palette[i % len(palette)], -1, cv2.LINE_AA)
        cx = (x0 + x1) // 2
        err = int(rng.uniform(3, 14))
        cv2.line(img, (cx, top - err), (cx, top + err), (30, 30, 30), 1, cv2.LINE_AA)
    img += rng.normal(0, 1.2, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def gen_scatter(h, w, rng):
    """散点图：白底 + 轴 + 点 + 拟合线"""
    img = np.full((h, w, 3), 255, np.float32)
    ml, mr, mt, mb = int(w * .14), int(w * .06), int(h * .08), int(h * .16)
    cv2.line(img, (ml, mt), (ml, h - mb), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(img, (ml, h - mb), (w - mr, h - mb), (60, 60, 60), 1, cv2.LINE_AA)
    n = int(rng.integers(25, 70))
    xs = rng.random(n)
    ys = np.clip(xs * rng.uniform(.5, 1.2) + rng.normal(0, .12, n), 0, 1)
    px = (ml + xs * (w - mr - ml)).astype(int)
    py = (h - mb - ys * (h - mb - mt)).astype(int)
    for x, y in zip(px, py):
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 4)), (60, 90, 170), -1, cv2.LINE_AA)
    k = rng.uniform(.6, 1.1)
    cv2.line(img, (ml, int(h - mb - .05 * (h - mb - mt))),
             (w - mr, int(h - mb - min(1., k) * (h - mb - mt))), (190, 70, 60), 2, cv2.LINE_AA)
    img += rng.normal(0, 1.2, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


GENERATORS = {
    'blot': gen_blot,
    'fluorescence': gen_fluorescence,
    'histology': gen_histology,
    'gel': gen_gel,
    'barplot': gen_barplot,
    'scatter': gen_scatter,
}


# ---------------------------------------------------------------- 变换注入


def apply_transform(img, kind, rng):
    """对图像施加一种已知变换（模拟造假者的复制-粘贴-微调）"""
    if kind == 'exact':
        return img.copy()
    if kind == 'recompress':
        return _jpeg_roundtrip(img, 72)
    if kind == 'jpeg55':
        return _jpeg_roundtrip(img, 55)
    if kind == 'brightness':
        return np.clip(img.astype(np.float32) * 1.12 + 18, 0, 255).astype(np.uint8)
    if kind == 'contrast':
        m = float(img.mean())
        return np.clip((img.astype(np.float32) - m) * 1.35 + m, 0, 255).astype(np.uint8)
    if kind == 'gamma':
        return np.clip(255 * (img.astype(np.float32) / 255) ** 1.25, 0, 255).astype(np.uint8)
    if kind == 'noise':
        return np.clip(img.astype(np.float32) + rng.normal(0, 7, img.shape), 0, 255).astype(np.uint8)
    if kind == 'scale075':
        return cv2.resize(img, None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA)
    if kind == 'scale150':
        return cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    if kind == 'crop80':
        h, w = img.shape[:2]
        return img[int(h * .1):int(h * .9), int(w * .1):int(w * .9)].copy()
    if kind == 'crop60':
        h, w = img.shape[:2]
        return img[int(h * .2):int(h * .8), int(w * .2):int(w * .8)].copy()
    if kind == 'rot180':
        return np.ascontiguousarray(img[::-1, ::-1])
    if kind == 'rot90':
        return np.ascontiguousarray(np.rot90(img))
    if kind == 'flipH':
        return np.ascontiguousarray(img[:, ::-1])
    if kind == 'flipV':
        return np.ascontiguousarray(img[::-1, :])
    raise ValueError(f'unknown transform: {kind}')


# ---------------------------------------------------------------- 跨文件数据集

CROSS_TRANSFORMS = [
    'exact', 'recompress', 'jpeg55', 'brightness', 'contrast', 'gamma',
    'noise', 'scale075', 'scale150', 'crop80', 'crop60',
    'rot180', 'rot90', 'flipH', 'flipV',
]

# 文件名风格：mixed = 部分复制件与原件同基名(触发同实验跳过启发式)，部分不同
SAME_BASE_SUFFIX = '_rep'


def build_cross_dataset(root: Path, rng, n_cases=24, n_neg=3, save_quality=92):
    cases = []
    positives = []
    gt_files = {}

    for ci in range(n_cases):
        gen_name = list(GENERATORS)[ci % len(GENERATORS)]
        gen = GENERATORS[gen_name]
        transform = CROSS_TRANSFORMS[ci % len(CROSS_TRANSFORMS)]

        # 尺寸随 case 变化（含非方形，用于暴露 rot90 的宽高比预过滤问题）
        if ci % 3 == 0:
            w, h = 420, 300
        elif ci % 3 == 1:
            w, h = 384, 384
        else:
            w, h = 360, 480

        cdir = root / f'case_{ci:03d}'
        cdir.mkdir(parents=True, exist_ok=True)

        base = gen(h, w, rng)
        variant = apply_transform(base, transform, rng)
        negatives = [gen(h, w, rng) for _ in range(n_neg)]

        # 命名风格（对应真实造假/正常命名的不同情形）
        group = f'{gen_name}_cond{ci % 5}'
        if ci % 4 == 3:
            # 通道别名：把 GFP 通道图当作 RFP 通道重复使用 —— v2 的同实验跳过启发式盲区
            naming = 'channel_alias'
            base_name, var_name = f'{group}_GFP', f'{group}_RFP'
        elif ci % 2 == 0:
            naming = 'same_base'          # 复制件保留原件名 + 后缀
            base_name, var_name = f'{group}_1', f'{group}_1_rep'
        else:
            naming = 'renamed'            # 复制件改名为另一个样本号
            base_name, var_name = f'{group}_1', f'{group}_2'

        files = []
        for name, arr in [(base_name, base), (var_name, variant)] + \
                         [(f'{group}_{3 + k}', a) for k, a in enumerate(negatives)]:
            fn = f'{name}.jpg'
            cv2.imwrite(str(cdir / fn), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), save_quality])
            files.append(fn)

        rel = lambda fn: f'{cdir.relative_to(root.parent).as_posix()}/{fn}'
        positives.append({'a': rel(files[0]), 'b': rel(files[1]),
                          'transform': transform, 'case': ci, 'generator': gen_name,
                          'naming': naming, 'alias_blind_spot': naming == 'channel_alias'})
        cases.append({'id': ci, 'dir': cdir.relative_to(root.parent).as_posix(),
                      'files': files, 'transform': transform, 'generator': gen_name,
                      'naming': naming, 'base': rel(files[0]), 'variant': rel(files[1])})
        for fn in files:
            gt_files[rel(fn)] = {'case': ci, 'generator': gen_name, 'role':
                                 'base' if fn == files[0] else
                                 'variant' if fn == files[1] else 'negative'}

    return {'cases': cases, 'positives': positives, 'files': gt_files}


# ---------------------------------------------------------------- 图内复用数据集

INTRA_RELATIONS = [
    'panel_reuse_exact', 'panel_reuse_recompress', 'panel_reuse_brightness',
    'panel_reuse_rot180', 'panel_reuse_flipH', 'panel_reuse_scale075',
    'panel_reuse_crop', 'panel_partial_overlap', 'panel_reuse_contrast',
    'copymove_within',
]


def _paste(canvas, content, box):
    x, y, cw, ch = box
    canvas[y:y + ch, x:x + cw] = content


def build_intra_dataset(root: Path, rng, n_figures=20,
                        cell=(320, 240), gutter=18, rows=2, cols=3):
    cw, ch = cell
    W = cols * cw + (cols - 1) * gutter
    H = rows * ch + (rows - 1) * gutter
    figures = []

    for fi in range(n_figures):
        gen_name = list(GENERATORS)[fi % len(GENERATORS)]
        gen = GENERATORS[gen_name]
        relation = INTRA_RELATIONS[fi % len(INTRA_RELATIONS)]

        canvas = np.full((H, W, 3), 255, np.uint8)
        boxes = []
        for r in range(rows):
            for c in range(cols):
                boxes.append((c * (cw + gutter), r * (ch + gutter), cw, ch))

        n_panels = rows * cols
        panels = [gen(ch, cw, rng) for _ in range(n_panels)]
        panel_ids = [f'p{i}' for i in range(n_panels)]

        # 复用目标：panel 0 为源，panel 4 为受体（不相邻，避免相邻块误判）
        src_i, dst_i = 0, 4
        reuse = []
        note = ''

        if relation == 'copymove_within':
            # 图内复制：在 panel 0 内部复制一块区域
            rw, rh = 90, 70
            x1, y1 = 40, 50
            x2, y2 = 190, 130
            reg = panels[src_i][y1:y1 + rh, x1:x1 + rw].copy()
            panels[src_i][y2:y2 + rh, x2:x2 + rw] = reg
            bx, by, _, _ = boxes[src_i]
            reuse.append({
                'relation': relation,
                'a_bbox': [bx + x1, by + y1, rw, rh],
                'b_bbox': [bx + x2, by + y2, rw, rh],
                'transform': 'exact', 'a_panel': panel_ids[src_i], 'b_panel': panel_ids[src_i],
            })
        elif relation == 'panel_partial_overlap':
            # 部分重叠：受体 panel 是源 panel 的平移裁剪
            src = panels[src_i]
            x0, y0 = 110, 80
            bw, bh = 180, 140
            sub = src[y0:y0 + bh, x0:x0 + bw].copy()
            dst = np.full((ch, cw, 3), 255, np.uint8)
            px, py = (cw - bw) // 2, (ch - bh) // 2
            dst[py:py + bh, px:px + bw] = sub
            panels[dst_i] = dst
            bx, by, _, _ = boxes[src_i]
            dx, dy, _, _ = boxes[dst_i]
            reuse.append({
                'relation': relation,
                'a_bbox': [bx + x0, by + y0, bw, bh],
                'b_bbox': [dx + px, dy + py, bw, bh],
                'transform': 'translation', 'a_panel': panel_ids[src_i],
                'b_panel': panel_ids[dst_i],
            })
        else:
            src = panels[src_i]
            subregion_gt = None
            if relation == 'panel_reuse_exact':
                dst = src.copy()
                tname = 'exact'
            elif relation == 'panel_reuse_recompress':
                dst = _jpeg_roundtrip(src, 60)
                tname = 'recompress'
            elif relation == 'panel_reuse_brightness':
                dst = np.clip(src.astype(np.float32) * 1.15 + 20, 0, 255).astype(np.uint8)
                tname = 'brightness'
            elif relation == 'panel_reuse_contrast':
                m = float(src.mean())
                dst = np.clip((src.astype(np.float32) - m) * 1.3 + m, 0, 255).astype(np.uint8)
                tname = 'contrast'
            elif relation == 'panel_reuse_rot180':
                dst = np.ascontiguousarray(src[::-1, ::-1])
                tname = 'rot180'
            elif relation == 'panel_reuse_flipH':
                dst = np.ascontiguousarray(src[:, ::-1])
                tname = 'flipH'
            elif relation == 'panel_reuse_scale075':
                small = cv2.resize(src, None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA)
                dst = np.full((ch, cw, 3), 255, np.uint8)
                sh, sw = small.shape[:2]
                px, py = (cw - sw) // 2, (ch - sh) // 2
                dst[py:py + sh, px:px + sw] = small
                tname = 'scale075'
            elif relation == 'panel_reuse_crop':
                x0, y0, bw, bh = 60, 40, 210, 170
                sub = src[y0:y0 + bh, x0:x0 + bw]
                dst = np.full((ch, cw, 3), 255, np.uint8)
                sh, sw = sub.shape[:2]
                px, py = (cw - sw) // 2, (ch - sh) // 2
                dst[py:py + sh, px:px + sw] = sub
                tname = 'crop'
                bx, by, _, _ = boxes[src_i]
                dx, dy, _, _ = boxes[dst_i]
                subregion_gt = {
                    'relation': relation,
                    'a_bbox': [bx + x0, by + y0, bw, bh],
                    'b_bbox': [dx + px, dy + py, bw, bh],
                    'transform': tname, 'a_panel': panel_ids[src_i],
                    'b_panel': panel_ids[dst_i],
                }
                panels[dst_i] = dst
            else:
                raise ValueError(relation)

            panels[dst_i] = dst
            if subregion_gt is not None:
                # 裁剪复用：真值就是「子区域↔子区域」，不是整块 panel
                reuse.append(subregion_gt)
            else:
                bx, by, _, _ = boxes[src_i]
                dx, dy, _, _ = boxes[dst_i]
                reuse.append({
                    'relation': relation,
                    'a_bbox': [bx, by, cw, ch], 'b_bbox': [dx, dy, cw, ch],
                    'transform': tname, 'a_panel': panel_ids[src_i], 'b_panel': panel_ids[dst_i],
                })

        for i, b in enumerate(boxes):
            _paste(canvas, panels[i], b)

        fn = f'fig_{fi:03d}.png'
        cv2.imwrite(str(root / fn), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        figures.append({'file': f'{root.name}/{fn}', 'relation': relation,
                        'generator': gen_name, 'panels': [
                            {'id': pid, 'bbox': list(b)} for pid, b in zip(panel_ids, boxes)],
                        'reuse': reuse, 'n_panels': n_panels})

    return {'figures': figures, 'cell': list(cell), 'gutter': gutter, 'rows': rows, 'cols': cols}


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description='生成合成基准数据集')
    ap.add_argument('--out', default='bench/data', help='输出目录')
    ap.add_argument('--cross-cases', type=int, default=24)
    ap.add_argument('--cross-negatives', type=int, default=3)
    ap.add_argument('--intra-figures', type=int, default=20)
    ap.add_argument('--clean', action='store_true', help='先清空输出目录')
    args = ap.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    cross_root = out / 'cross'
    cross_root.mkdir(exist_ok=True)
    cross = build_cross_dataset(cross_root, rng, args.cross_cases, args.cross_negatives)

    intra_root = out / 'intra'
    intra_root.mkdir(exist_ok=True)
    intra = build_intra_dataset(intra_root, rng, args.intra_figures)

    gt = {'seed': SEED, 'cross': cross, 'intra': intra}
    gt_path = out / 'ground_truth.json'
    gt_path.write_text(json.dumps(gt, indent=1, ensure_ascii=False), encoding='utf-8')

    n_cross_files = sum(len(c['files']) for c in cross['cases'])
    print(f'[✓] cross : {len(cross["cases"])} cases, {n_cross_files} files, '
          f'{len(cross["positives"])} 正样本对')
    print(f'[✓] intra : {len(intra["figures"])} figures')
    for f in intra['figures'][:3]:
        print(f'      {f["file"]}  {f["relation"]}  reuse={len(f["reuse"])}')
    print(f'[✓] ground truth -> {gt_path}')


if __name__ == '__main__':
    main()
