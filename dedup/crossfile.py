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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                    '.gif', '.webp', '.svs', '.ndpi', '.vsi'}
EXCLUDE_DIRS = {'visualization', 'venv', '.git', '__pycache__', 'node_modules',
                '_thumbnails'}
CHANNEL_SUFFIXES = ['_CH1', '_CH2', '_CH3', '_CH4', '_CH5', '_Overlay',
                    '_Bright', '_DIC', '_GFP', '_RFP', '_DAPI', '_Cy3',
                    '_Cy5', '_FITC', '_TRITC']

# 参与索引的几何变换
GEOMETRY = ('id', 'rot90', 'rot180', 'rot270', 'flipH', 'flipV')

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


def _md5_streaming(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def _load_gray(path: str, max_dim: int = 512) -> Optional[np.ndarray]:
    """读取灰度图；超大图用 PIL draft 降采样解码，避免内存爆炸"""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.draft('L', (max_dim, max_dim))
            g = np.asarray(im.convert('L'))
    except Exception:
        g = None
    if g is None:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return None
    if max(g.shape) > max_dim:
        s = max_dim / max(g.shape)
        g = cv2.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(g)


def load_record(path: str, hash_size: int = 16) -> Optional[ImageRecord]:
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
        return np.ascontiguousarray(g[::-1, :])

    ph = _bits_to_u64(_phash_bits(gray, hash_size))
    dh = _bits_to_u64(_dhash_bits(gray, hash_size))
    geom = {t: _bits_to_u64(_phash_bits(_geom(gray, t), hash_size)) for t in GEOMETRY}

    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-9)

    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return ImageRecord(path=str(Path(path).resolve()), md5=md5, width=w, height=h,
                       gray_std=float(gray.std()), phash=ph, dhash=dh,
                       geom_hash=geom, hist=hist, gray_small=small)


def scan_directory(root: str, workers: int = 8, hash_size: int = 16
                   ) -> List[ImageRecord]:
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
        for rec in ex.map(lambda p: load_record(p, hash_size), paths):
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
    return np.ascontiguousarray(gray[::-1, :])


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


def find_cross_duplicates(records: Sequence[ImageRecord],
                          phash_threshold: int = 12,
                          min_votes: int = 2,
                          md5_only_threshold: int = 0,
                          verbose: bool = False) -> List[CrossMatch]:
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
    cand = geometric_candidates(records, max_dist=phash_threshold * 2)
    cand = {k: v for k, v in cand.items() if k not in exact_pairs}

    if verbose:
        print(f'  多变换哈希候选: {len(cand)} 对 (共 {n} 张, 朴素 {n*(n-1)//2} 对)')

    # 3) 投票验证
    for (i, j), hd in cand.items():
        m = verify_pair(records[i], records[j],
                        pthresh=phash_threshold, min_votes=min_votes)
        if m:
            matches.append(m)
    return matches
