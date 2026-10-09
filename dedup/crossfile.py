"""
跨文件查重（v3）
================
在 v2 基础上的三处实质改动：

1. **候选生成向量化**
   v2 对每个候选对调用一次 Python 函数做 Hamming 距离，n 大时是纯 Python 循环。
   这里把所有哈希压成 uint64 数组，用 numpy 分块算 XOR + 查表 popcount，
   一次算出一整块的距离矩阵。同样的结果，纯 Python 层开销几乎归零。

2. **修掉 v2 的两个漏报盲区**
   a) `_is_same_experiment_channel`：同目录下同基名、仅通道后缀不同（_GFP/_RFP）
      的对会被 **整对跳过**。但「把 GFP 通道图当作 RFP 通道复用」恰恰是常见的
      图片复用方式。v3 不再直接丢弃，而是照常检测、只下调严重程度。
   b) 宽高比预过滤（>1.3 直接跳过）会把非方形图的 90° 旋转副本直接筛掉。
      v3 的候选生成基于「多变换哈希」，旋转副本照样能进候选。

3. **旋转/翻转召回内建到主索引**
   v2 把旋转检测放在独立的 3c 阶段，需要额外一轮候选构建。v3 在生成候选时
   就把 6 种几何变换的哈希一起索引，旋转副本与普通副本走同一条路径。
"""

import hashlib
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from .bow import bow_candidates, compute_bow_keys
from .edge_overlap import check_edge_overlap
from .subimage import (SubimageCache, check_subimage_arrays, read_gray_capped,
                        subimage_candidate_pairs)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                    '.gif', '.webp', '.svs', '.ndpi', '.vsi'}
EXCLUDE_DIRS = {'visualization', 'venv', '.git', '__pycache__', 'node_modules',
                '_thumbnails'}
CHANNEL_SUFFIXES = ['_CH1', '_CH2', '_CH3', '_CH4', '_CH5', '_Overlay',
                    '_Bright', '_DIC', '_GFP', '_RFP', '_DAPI', '_Cy3',
                    '_Cy5', '_FITC', '_TRITC']

# 参与索引的几何变换。
# ±15°/±30° 这四档是关键：只索引 90° 整数倍时，**任意角度旋转**的副本
# （如 17°）哈希差异巨大、根本进不了候选集，召回直接为 0。
# 加上粗角度覆盖后，17° 旋转可由 rot345 档补到 2° 残差，落进哈希容差内。
GEOMETRY = ('id', 'rot90', 'rot180', 'rot270', 'flipH', 'flipV',
            'rot15', 'rot345', 'rot30', 'rot330')

_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], np.uint8)


# ---------------------------------------------------------------- 记录


@dataclass
class ImageRecord:
    path: str
    md5: str
    width: int
    height: int
    gray_std: float
    phash: np.ndarray            # 每行 uint64 位（hash_size² 位，packbits 后按 64 对齐）
    dhash: np.ndarray
    geom_hash: Dict[str, np.ndarray] = field(default_factory=dict)
    sub_hash: Optional[np.ndarray] = None   # (K, words) 若干子窗口的 pHash
    bow_keys: Optional[np.ndarray] = None    # ORB 词袋键（uint32 去重数组）
    hist: Optional[np.ndarray] = None
    gray_small: Optional[np.ndarray] = None


def _bits_to_u64(bits: np.ndarray) -> np.ndarray:
    """bool 位图 -> uint64 数组（按位打包，右侧补零）"""
    flat = np.asarray(bits, np.uint8).ravel()
    pad = (-len(flat)) % 64
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
    packed = np.packbits(flat.reshape(-1, 8), axis=1, bitorder='big')
    return packed.reshape(-1, 8).copy().view(np.uint64).ravel()


def _phash_bits(gray: np.ndarray, hash_size: int) -> np.ndarray:
    small = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))
    low = dct[:hash_size, :hash_size].copy()
    flat = low.ravel()
    med = np.median(flat[1:])
    return (flat > med).astype(np.uint8)


def _dhash_bits(gray: np.ndarray, hash_size: int) -> np.ndarray:
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).astype(np.uint8).ravel()


# 子窗口布局：(中心相对坐标 x0, y0, 边长占比)。用于「裁剪副本」的候选生成。
SUB_WINDOWS = (
    (0.20, 0.20, 0.60),   # 居中 60%
    (0.10, 0.10, 0.80),   # 居中 80%
    (0.05, 0.05, 0.70),   # 左上
    (0.25, 0.05, 0.70),   # 右上
    (0.05, 0.25, 0.70),   # 左下
    (0.25, 0.25, 0.70),   # 右下
)


def _subwindow_hashes(gray: np.ndarray, hash_size: int) -> np.ndarray:
    """各子窗口的 pHash，堆成 (K, words)。"""
    h, w = gray.shape[:2]
    out = []
    for fx, fy, fs in SUB_WINDOWS:
        x0, y0 = int(w * fx), int(h * fy)
        sw, sh = int(w * fs), int(h * fs)
        sub = gray[y0:y0 + sh, x0:x0 + sw]
        if sub.size == 0:
            sub = gray
        out.append(_bits_to_u64(_phash_bits(sub, hash_size)))
    return np.stack(out)


def _md5_streaming(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def _load_gray(path: str, max_dim: int = 512) -> Optional[np.ndarray]:
    """
    读取灰度图；超大图用 PIL draft 降采样解码，避免内存爆炸。

    16bit TIF（荧光/共聚焦最常见）不能直接 convert('L')：PIL 会按 8bit 截断，
    高动态范围图像的绝大部分信息会丢成一片白。这里改用百分位拉伸归一化。
    """
    from PIL import Image
    g = None
    try:
        with Image.open(path) as im:
            if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
                arr = np.asarray(im, dtype=np.float32)
                lo, hi = np.percentile(arr, (1.0, 99.0))
                if hi - lo < 1e-6:
                    hi = lo + 1.0
                g = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            else:
                im.draft('L', (max_dim, max_dim))
                g = np.asarray(im.convert('L'))
    except Exception:
        g = None
    if g is None:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            import tifffile
            try:
                arr = tifffile.imread(path)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                lo, hi = np.percentile(arr.astype(np.float32), (1.0, 99.0))
                g = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255,
                            0, 255).astype(np.uint8)
            except Exception:
                return None
    if max(g.shape) > max_dim:
        s = max_dim / max(g.shape)
        g = cv2.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(g)


def load_record(path: str, hash_size: int = 16,
                compute_bow: bool = True) -> Optional[ImageRecord]:
    gray = _load_gray(path)
    if gray is None or gray.size == 0:
        return None
    try:
        md5 = _md5_streaming(path)
    except Exception:
        return None

    def _geom(g, name):
        if name == 'id':
            return g
        if name == 'rot90':
            return np.ascontiguousarray(np.rot90(g))
        if name == 'rot180':
            return np.ascontiguousarray(g[::-1, ::-1])
        if name == 'rot270':
            return np.ascontiguousarray(np.rot90(g, 3))
        if name == 'flipH':
            return np.ascontiguousarray(g[:, ::-1])
        if name == 'flipV':
            return np.ascontiguousarray(g[::-1, :])
        if name.startswith('rot'):                     # rot15 / rot345 / rot30 / rot330
            ang = int(name[3:])
            h, w = g.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        raise ValueError(name)

    ph = _bits_to_u64(_phash_bits(gray, hash_size))
    dh = _bits_to_u64(_dhash_bits(gray, hash_size))
    geom = {t: _bits_to_u64(_phash_bits(_geom(gray, t), hash_size)) for t in GEOMETRY}
    sub = _subwindow_hashes(gray, hash_size)
    # ORB 词袋键只在启用该层时才算：每图上千个 int 装进 set，
    # 1200 张会让常驻内存从 ~73MB 涨到 ~257MB，而该层默认关闭。
    bow = compute_bow_keys(gray) if compute_bow else None

    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-9)

    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return ImageRecord(path=str(Path(path).resolve()), md5=md5, width=w, height=h,
                       gray_std=float(gray.std()), phash=ph, dhash=dh,
                       geom_hash=geom, sub_hash=sub, bow_keys=bow, hist=hist, gray_small=small)


def scan_directory(root: str, workers: int = 8, hash_size: int = 16,
                   compute_bow: bool = True) -> List[ImageRecord]:
    root_p = Path(root)
    paths = []
    seen = set()
    for p in root_p.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part in EXCLUDE_DIRS for part in p.relative_to(root_p).parts):
            continue
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        paths.append(str(p))
    paths.sort()

    out: List[ImageRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(lambda p: load_record(p, hash_size, compute_bow), paths):
            if rec is not None:
                out.append(rec)
    return out


# ---------------------------------------------------------------- 向量化候选


def hamming_close_pairs(hashes: np.ndarray, max_dist: int, chunk: int = 512
                        ) -> List[Tuple[int, int, int]]:
    """
    uint64 哈希数组的成对 Hamming 距离，返回距离 <= max_dist 的 (i, j, d)，i<j。
    分块 + 查表 popcount，全程 numpy，没有逐对 Python 调用。
    """
    n = len(hashes)
    out: List[Tuple[int, int, int]] = []
    if n < 2:
        return out
    nbytes = hashes.dtype.itemsize
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        xor = hashes[i0:i1, None] ^ hashes[None, :]
        d = _POPCOUNT[xor.view(np.uint8).reshape(xor.shape[0], xor.shape[1], nbytes)].sum(-1)
        rows, cols = np.where(d <= max_dist)
        for r, c in zip(rows, cols):
            gi, gj = i0 + int(r), int(c)
            if gi < gj:
                out.append((gi, gj, int(d[r, c])))
    return out


def subwindow_candidates(records: Sequence[ImageRecord], max_dist: int,
                         chunk: int = 0) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    用「子窗口哈希」找裁剪候选，复杂度 O(n) 级别而不是 O(n²)。

    做法：给每张图预先算好 K 个子窗口的 pHash。若 B 是 A 的裁剪副本，
    则 B 的 **整图哈希** 应当接近 A 的 **某个子窗口哈希**。
    于是索引 K·n 个哈希，比较「所有图的整图哈希」×「所有图的子窗口哈希」。

    为什么必须这么做：按「面积比」枚举候选是 O(n²)——1200 张图会产生
    60034 个候选，任何上限都只能看前 3%，绝大多数真裁剪根本轮不到。
    子窗口哈希把候选降到极小且与 n 近似线性。

    返回 {(i, j): (窗口序号, 距离)}，i<j。
    """
    n = len(records)
    best: Dict[Tuple[int, int], Tuple[int, int]] = {}
    if n < 2 or records[0].sub_hash is None:
        return best

    wpt = records[0].phash.size
    k = records[0].sub_hash.shape[0]
    if chunk <= 0:
        chunk = max(16, min(512, 8_000_000 // max(n * wpt, 1)))

    ident = np.stack([r.phash for r in records])                 # (n, wpt)
    subs = np.stack([r.sub_hash for r in records])               # (n, k, wpt)

    for win in range(k):
        sub_w = subs[:, win, :]                                  # (n, wpt)
        for a0 in range(0, n, chunk):
            a1 = min(n, a0 + chunk)
            xor = ident[a0:a1, None, :] ^ sub_w[None, :, :]
            cnt = _POPCOUNT[xor.view(np.uint8).reshape(a1 - a0, n, wpt * 8)].sum(-1)
            rows, cols = np.where(cnt <= max_dist)
            for r, c in zip(rows, cols):
                gi, gj = a0 + int(r), int(c)
                if gi == gj:
                    continue
                key = (min(gi, gj), max(gi, gj))
                v = int(cnt[r, c])
                if key not in best or v < best[key][1]:
                    best[key] = (win, v)
    return best


def geometric_candidates(records: Sequence[ImageRecord], max_dist: int,
                         chunk: int = 0) -> Dict[Tuple[int, int], int]:
    """
    找出「存在某个几何变换使两者哈希接近」的图对，返回 {(i,j): 最小 Hamming 距离}。

    只需 6 次比较而不是 36 次：几何变换群是封闭的，
    dist(A^t1, B^t2) = dist(A^id, B^(t2∘t1⁻¹))，
    所以固定用 A 的恒等哈希去比 B 的 6 个变换即可覆盖全部组合。
    全程 numpy 分块 XOR + 查表 popcount，没有逐对 Python 调用。
    """
    n = len(records)
    best: Dict[Tuple[int, int], int] = {}
    if n < 2:
        return best

    wpt = records[0].phash.size
    if chunk <= 0:                      # 控制单块内存 ≈ 64MB
        chunk = max(16, min(512, 8_000_000 // max(n * wpt, 1)))

    left = np.stack([r.geom_hash['id'] for r in records])       # (n, wpt)
    for t in GEOMETRY:
        right = np.stack([r.geom_hash[t] for r in records])     # (n, wpt)
        for a0 in range(0, n, chunk):
            a1 = min(n, a0 + chunk)
            xor = left[a0:a1, None, :] ^ right[None, :, :]
            cnt = _POPCOUNT[xor.view(np.uint8).reshape(a1 - a0, n, wpt * 8)].sum(-1)
            rows, cols = np.where(cnt <= max_dist)
            for r, c in zip(rows, cols):
                gi, gj = a0 + int(r), int(c)
                if gi >= gj:
                    continue
                v = int(cnt[r, c])
                key = (gi, gj)
                if key not in best or v < best[key]:
                    best[key] = v
    return best


# ---------------------------------------------------------------- 验证


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """标准 SSIM（高斯窗），cv2 实现，比 v2 的 Python 可分离卷积快得多"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    saa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sbb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    return float(np.mean(num / den))


def _ssim_z(a: np.ndarray, b: np.ndarray) -> float:
    """亮度/对比度归一化后的 SSIM：专捕「同结构不同曝光」"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return _ssim(a * 40 + 128, b * 40 + 128)


def _hist_js(h1: np.ndarray, h2: np.ndarray) -> float:
    """Jensen-Shannon 相似度 -> 1 表示完全一致"""
    m = 0.5 * (h1 + h2)
    def _kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log((p[mask] + 1e-12) / (q[mask] + 1e-12))))
    js = 0.5 * _kl(h1, m) + 0.5 * _kl(h2, m)
    return float(max(0.0, 1.0 - js / np.log(2)))


@dataclass
class CrossMatch:
    image1: str
    image2: str
    severity: str
    similarity: float
    match_type: str
    details: str
    same_channel_base: bool = False
    confidence: float = 0.0
    vote_details: str = ''


def _same_channel_base(p1: str, p2: str) -> bool:
    """同目录、基名相同、仅通道后缀不同（如 S1_GFP vs S1_RFP）"""
    a, b = Path(p1), Path(p2)
    if a.parent != b.parent:
        return False

    def base(stem):
        for s in CHANNEL_SUFFIXES:
            if stem.upper().endswith(s.upper()):
                return stem[:len(stem) - len(s)]
        return stem
    return base(a.stem) == base(b.stem)


def _popcount(x: np.ndarray) -> int:
    return int(_POPCOUNT[np.asarray(x).view(np.uint8)].sum())


def _apply_geom(gray: np.ndarray, name: str) -> np.ndarray:
    if name == 'id':
        return gray
    if name == 'rot90':
        return np.ascontiguousarray(np.rot90(gray))
    if name == 'rot180':
        return np.ascontiguousarray(gray[::-1, ::-1])
    if name == 'rot270':
        return np.ascontiguousarray(np.rot90(gray, 3))
    if name == 'flipH':
        return np.ascontiguousarray(gray[:, ::-1])
    if name == 'flipV':
        return np.ascontiguousarray(gray[::-1, :])
    if name.startswith('rot'):
        ang = int(name[3:])
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(name)


def verify_pair(r1: ImageRecord, r2: ImageRecord,
                pthresh: int = 12, min_votes: int = 2) -> Optional[CrossMatch]:
    """
    结构类检测器投票：pHash / dHash / SSIM，至少 min_votes 票通过。
    候选阶段只保证「存在某个几何变换使哈希接近」，这里再确认究竟是哪个变换，
    并把旋转/翻转副本用变换后的图像去做 SSIM，避免二次漏报。
    """
    if min(r1.gray_std, r2.gray_std) < 4.0:
        return None

    ident_ph = _popcount(r1.phash ^ r2.phash)
    ident_dh = _popcount(r1.dhash ^ r2.dhash)
    best_geom, best_t = ident_ph, 'id'
    for t in GEOMETRY:
        d = _popcount(r1.phash ^ r2.geom_hash[t])
        if d < best_geom:
            best_geom, best_t = d, t

    votes, sims, names = [], [], []
    if best_geom <= pthresh:
        votes.append(True)
        sims.append(1 - best_geom / 256)
        names.append('pHash' + (f'[{best_t}]' if best_t != 'id' else ''))
    if ident_dh <= pthresh:
        votes.append(True)
        sims.append(1 - ident_dh / 256)
        names.append('dHash')

    g2 = _apply_geom(r2.gray_small, best_t) if best_t != 'id' else r2.gray_small
    if g2.shape == r1.gray_small.shape:
        s = _ssim(r1.gray_small, g2)
        sz = _ssim_z(r1.gray_small, g2)
        if s >= 0.90 or sz >= 0.985:
            votes.append(True)
            sims.append(max(s, sz))
            names.append('SSIM' + (f'/ZS{sz:.3f}' if sz >= 0.985 else ''))
    else:
        s, sz = 0.0, 0.0

    if len(votes) < min_votes:
        return None

    conf = float(np.mean(sims)) * (0.5 + 0.5 * len(votes) / 3)
    if r1.hist is not None and r2.hist is not None and _hist_js(r1.hist, r2.hist) >= 0.80:
        conf = min(1.0, conf + 0.05)

    same = _same_channel_base(r1.path, r2.path)
    if same:
        # 同基名不同通道：照常报告，但下调严重程度（也可能是合法的多通道展示）
        conf *= 0.75

    sev = 'critical' if best_geom <= 2 and best_t == 'id' else 'high'
    if same and sev == 'critical':
        sev = 'high'
    return CrossMatch(
        image1=r1.path, image2=r2.path, severity=sev,
        similarity=round(float(np.mean(sims)), 4),
        match_type='+'.join(names), confidence=round(conf, 4),
        details=(f'pHash差异 {best_geom}/256'
                 + (f'(经{best_t})' if best_t != 'id' else '')
                 + f', dHash差异 {ident_dh}/256, SSIM {s:.3f}'
                 + (f', 亮度归一化SSIM {sz:.3f}' if sz >= 0.985 else '')
                 + ('  [同基名不同通道]' if same else '')),
        same_channel_base=same, vote_details='+'.join(names))


def _feature_verify(records, pending, min_inliers: float, ncc: float,
                    max_features: int, workers: int, verbose: bool):
    """
    对「哈希投票没通过」的候选做特征级判定（SIFT 匹配 + RANSAC + 纹理 NCC + 退化检测）。

    这是召回「变换的叠加」（裁剪∘旋转、镜像∘任意角旋转）的唯一途径：
    这类副本的整图哈希距离远超任何可用阈值（crop_60 实测 118/256），
    但其中含可匹配的局部结构，SIFT 能找出来。

    特征按路径缓存 —— 不缓存时每对候选都要重跑两次 SIFT，
    这是把秒级拖成分钟级的元凶。判定并行跑（OpenCV 会释放 GIL）。
    """
    from collections import OrderedDict
    from concurrent.futures import ThreadPoolExecutor
    from .features import extract_pair_features, match_features

    # 特征缓存必须**有上限**：单张图的 desc 是 Nx128 float32（N=3000 时约 1.5MB），
    # 连同灰度图与纹理掩膜约 2.7MB。无上限地缓存所有出现过的图，
    # 1200 张规模就能把常驻内存顶到 GB 级。这里用 FIFO 淘汰。
    cache: "OrderedDict[str, object]" = OrderedDict()
    cache_cap = 512                      # 常规规模下等同不淘汰；超大目录才兜底
    lock = threading.Lock()

    def _feat(path):
        with lock:
            if path in cache:
                cache.move_to_end(path)
                return cache[path]
        g = read_gray_capped(path, 1024)
        f = extract_pair_features(g, max_features) if g is not None else None
        with lock:
            cache[path] = f
            while len(cache) > cache_cap:
                cache.popitem(last=False)
        return f

    def _one(pair):
        i, j = pair
        try:
            fa, fb = _feat(records[i].path), _feat(records[j].path)
            if fa is None or fb is None:
                return None
            ms = match_features(fa, fb, min_inliers=int(min_inliers),
                                ncc_threshold=ncc)
            return (pair, ms)
        except Exception:
            return None

    out = []
    if not pending:
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_one, pending):
            if res and res[1]:
                out.append(res)
    if verbose:
        print(f'  特征级判定: {len(pending)} 对 → 命中 {len(out)}')
    return out


def find_cross_duplicates(records: Sequence[ImageRecord],
                          phash_threshold: int = 12,
                          min_votes: int = 2,
                          verbose: bool = False,
                          detect_subimage: bool = True,
                          subimage_threshold: float = 0.85,
                          detect_edge: bool = True,
                          edge_threshold: float = 0.55,
                          max_subimage_pairs: int = 4000,
                          candidate_threshold: int = 0,
                          use_bow: bool = True,
                          bow_min_shared: int = 8,
                          detect_features: bool = True,
                          max_feature_pairs: int = 8000,
                          feature_min_inliers: int = 80,
                          feature_ncc: float = 0.60,
                          feature_max_features: int = 1000,
                          workers: int = 8) -> List[CrossMatch]:
    """
    candidate_threshold: 候选生成的 pHash 距离上限（0 = 用 phash_threshold*2）。
    单独暴露出来是因为**候选层与判定层的最优阈值不同**：
    任意角度旋转副本的哈希距离实测约 36/256（见 COMPARISON.md），
    按 24 卡会整类漏掉。

    use_bow: ORB 词袋候选层。早期把它单独打开时毫无收益（召回没涨、误报暴涨），
    因为那些候选在判定层被整图结构投票否决。现在判定层有了特征级验证
    （detect_features），BOW 才有了意义：它负责**把组合变换的候选挖出来**，
    特征级验证负责**判定**。

    detect_features: 对投票未通过的候选做 SIFT+RANSAC+纹理NCC 判定。
    这是覆盖「变换的叠加」（裁剪∘旋转、镜像∘任意角旋转）的唯一途径 ——
    这类副本整图哈希距离 118/256（crop_60），任何阈值都够不着。
    """
    if candidate_threshold <= 0:
        candidate_threshold = max(phash_threshold * 2, 40)
    n = len(records)
    matches: List[CrossMatch] = []

    # 1) 完全相同（MD5 分组）
    by_md5: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_md5[r.md5].append(i)
    exact_pairs = set()
    for _, group in by_md5.items():
        if len(group) > 1:
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    i, j = group[a], group[b]
                    matches.append(CrossMatch(
                        image1=records[i].path, image2=records[j].path,
                        severity='critical', similarity=1.0, match_type='MD5',
                        details='文件内容完全相同（MD5 一致）', confidence=1.0,
                        vote_details='MD5'))
                    exact_pairs.add((i, j))

    # 2) 多变换哈希候选（含旋转/翻转）
    cand = geometric_candidates(records, max_dist=candidate_threshold)
    cand = {k: v for k, v in cand.items() if k not in exact_pairs}

    if verbose:
        print(f'  多变换哈希候选: {len(cand)} 对 (共 {n} 张, 朴素 {n*(n-1)//2} 对)')

    # 2b) ORB 词袋候选：哈希层对「变换的组合」（裁剪∘旋转、镜像∘任意角旋转）
    #     无能为力 —— 实测把候选阈值从 24 放宽到 64 也只多 1.3% 召回。
    #     ORB 描述子天生抗旋转/缩放，用来补这一类。
    bow_cand = {}
    if use_bow and n >= 2:
        bow_cand = bow_candidates([r.bow_keys for r in records],
                                  min_shared=bow_min_shared)
        bow_cand = {k: v for k, v in bow_cand.items() if k not in exact_pairs}
        if verbose:
            print(f'  ORB 词袋候选: {len(bow_cand)} 对 (共享词 >= {bow_min_shared})')

    # 3) 投票验证 + 边缘拼接；未通过的候选留到第 3b 步做特征级判定
    reported = set(exact_pairs)
    pending = []
    # 注意 BOW 独有的候选**不走整图投票**：BOW 存在的意义就是「变换的叠加」，
    # 而这类副本本就不满足整图一致性。更要紧的是——两张不同的柱状图因为
    # 大片白底，SSIM/pHash 也会互相通过投票（实测 2 个误报全部来自这里）。
    # BOW 候选一律交给特征级判定，其内点分离度大得多（真同源 >=135，误报 <=38）。
    hash_pairs = set(cand)
    for (i, j) in sorted(hash_pairs):
        m = verify_pair(records[i], records[j],
                        pthresh=phash_threshold, min_votes=min_votes)
        if m:
            matches.append(m)
            reported.add((i, j))
            continue
        pending.append((i, j))
        if detect_edge:
            eo = check_edge_overlap(records[i].gray_small, records[j].gray_small,
                                    edge_threshold)
            if eo:
                direction, score, detail = eo
                matches.append(CrossMatch(
                    image1=records[i].path, image2=records[j].path,
                    severity='high' if score > 0.75 else 'medium',
                    similarity=round(score, 4), match_type=f'疑似{direction}边缘重叠/拼接',
                    details=detail, confidence=round(score, 4),
                    vote_details='edge'))
                reported.add((i, j))

    pending.extend(sorted(set(bow_cand) - hash_pairs))

    # 3b) 特征级判定：整图结构不一致、但含同源局部（变换叠加）才走得通
    if detect_features and pending:
        # 按哈希/共享词证据强弱排序后再截断，避免上限截掉最可能的那些
        def _rank(p):
            return (cand.get(p, 999), -bow_cand.get(p, 0))
        pending.sort(key=_rank)
        if len(pending) > max_feature_pairs:
            if verbose:
                print(f'  特征级候选 {len(pending)} 对，按证据强度取前 {max_feature_pairs}')
            pending = pending[:max_feature_pairs]
        # 截断后按图像下标重排：让相邻的候选对共享同一张图，
        # 特征缓存命中率大幅提高（否则小缓存会反复淘汰重算，
        # 实测 1200 张下检测耗时从 18s 涨到 98s）。
        pending.sort(key=lambda p: (p[0], p[1]))

        for (i, j), ms in _feature_verify(records, pending, feature_min_inliers,
                                          feature_ncc, feature_max_features,
                                          workers, verbose):
            if (i, j) in reported:
                continue
            best = max(ms, key=lambda r: r.score)
            reported.add((i, j))
            matches.append(CrossMatch(
                image1=records[i].path, image2=records[j].path,
                severity='high' if best.n_inliers >= 40 else 'medium',
                similarity=round(best.ncc, 4),
                match_type=f'特征匹配（{best.transform}）',
                details=(f'RANSAC 内点 {best.n_inliers}，纹理掩膜 NCC {best.ncc:.3f}，'
                         f'匹配区域占比 {best.meta.get("coverage", 0):.2f}'),
                confidence=round(best.score, 4), vote_details='feature'))

    # 4) 子图 / 裁剪（整图哈希的天然盲区，需要模板匹配单独兜）
    #    候选用子窗口哈希生成（O(n)），不再按面积比枚举（O(n²)，1200 张图会
    #    产生 6 万个候选，任何上限都只能覆盖前 3%）。
    if detect_subimage and n >= 2:
        sub_cands = subwindow_candidates(records, max_dist=phash_threshold * 2)
        # 面积比做二次确认：真裁剪必然一大一小
        filtered = []
        for (i, j), (win, dist) in sub_cands.items():
            if (i, j) in reported:
                continue
            a1 = records[i].width * records[i].height
            a2 = records[j].width * records[j].height
            ratio = max(a1, a2) / max(min(a1, a2), 1)
            if not (1.25 <= ratio <= 6.0):
                continue
            big_i, small_i = (i, j) if a1 >= a2 else (j, i)
            filtered.append((dist, big_i, small_i))

        filtered.sort()                     # 哈希最接近的先查
        filtered = filtered[:max_subimage_pairs]
        cache = SubimageCache()
        hits = 0
        for dist, big_i, small_i in filtered:
            res = check_subimage_arrays(cache.get(records[big_i].path),
                                        cache.get(records[small_i].path),
                                        subimage_threshold)
            if res:
                x, y, w, h = res['box']
                matches.append(CrossMatch(
                    image1=records[big_i].path, image2=records[small_i].path,
                    severity='critical', similarity=round(res['score'], 4),
                    match_type='疑似子图/裁剪',
                    details=(f"小图匹配到大图位置({x},{y})，缩放比 {res['scale']:.2f}，"
                             f"相关系数 {res['score']:.3f}"),
                    confidence=round(res['score'], 4), vote_details='subimage'))
                hits += 1
        if verbose:
            print(f'  子图/裁剪候选: {len(sub_cands)} → 过滤后 {len(filtered)} 对 → 命中 {hits}')

    return matches
