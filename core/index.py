"""多索引哈希 (Multi-Index Hashing, MIH) 候选生成

思路:
    将 n_bits 位的二进制哈希按 stride 拆成 n_tables 个子表,
    每个子表取固定 16 位片段作为桶键。
    两个哈希的汉明距离 <= T 时, 差异位最多落在 T 个不同子表,
    因此当 T < bits_per_table 时, 至少有一个子表桶键完全相同。
    -> 对每张子表按桶键排序分组, 组内两两配对即为候选对,
       再用精确汉明距离过滤, 保证 100% 召回、无漏检。

复杂度: O(N log N) 建表 + O(候选数) 验证, 替代朴素 O(N^2)。
"""
from itertools import combinations

import numpy as np


def imagehash_to_bits(h) -> np.ndarray:
    """将 imagehash 对象转为一维 0/1 位向量 (uint8)。"""
    return np.asarray(h.hash, dtype=np.uint8).reshape(-1)


def hamming_distances(bits_a: np.ndarray, bits_b: np.ndarray) -> np.ndarray:
    """批量计算汉明距离。

    bits_a / bits_b: (P, n_bits) uint8 -> (P,) 汉明距离
    """
    x = np.bitwise_xor(bits_a, bits_b)
    return np.bitwise_count(x).sum(axis=1)


class HashIndex:
    """二进制哈希的 MIH 索引。

    用法:
        idx = HashIndex(n_tables=16)
        idx.add(ids, bits_matrix)          # ids: list[int], bits: (N, n_bits)
        pairs = idx.candidates(threshold)  # set[(i, j)]
    """

    def __init__(self, n_tables: int = 16):
        self.n_tables = n_tables
        self._ids: list = []
        self._bits: np.ndarray = None
        self._n_bits = 0
        self._max_pairs_per_table = 300_000

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, ids, bits_2d: np.ndarray):
        bits = np.asarray(bits_2d, dtype=np.uint8)
        if bits.ndim != 2:
            raise ValueError("bits_2d 必须是 (N, n_bits) 二维数组")
        if self._bits is None:
            self._bits = bits
        else:
            if bits.shape[1] != self._n_bits:
                raise ValueError("位宽不一致")
            self._bits = np.concatenate([self._bits, bits], axis=0)
        self._n_bits = bits.shape[1]
        self._ids.extend(ids)

    def _fit_tables(self) -> int:
        """在保证每表 >= 8 位的前提下, 取尽量大的表数。"""
        n_bits = self._n_bits
        for t in range(min(self.n_tables, n_bits), 0, -1):
            if n_bits % t == 0 and n_bits // t >= 8:
                return t
        return 1

    def _table_keys(self, bpt: int) -> np.ndarray:
        """计算每张子表的桶键。bits[n, j*nt + t] -> key[n, t]。"""
        nt = self._n_bits // bpt
        reshaped = self._bits.reshape(self._bits.shape[0], bpt, nt)
        transposed = reshaped.transpose(0, 2, 1)  # (N, nt, bpt)
        shifts = np.arange(bpt, dtype=np.uint32)
        keys = (transposed.astype(np.uint32) << shifts).sum(axis=2)
        return keys

    def candidates(self, threshold: int, max_group: int = 3000) -> set:
        """返回所有汉明距离 <= threshold 的 (i, j) 候选对。

        max_group: 同一桶内元素超过该值时抽样, 防止病态数据爆炸。
        """
        n = self.size
        if n < 2:
            return set()

        nt = self._fit_tables()
        bpt = self._n_bits // nt
        if threshold >= bpt:
            return self._bruteforce(threshold)

        keys = self._table_keys(bpt)
        pairs = set()

        for t in range(nt):
            col = keys[:, t]
            order = np.argsort(col, kind="stable")
            srt = col[order]
            bounds = np.flatnonzero(srt[1:] != srt[:-1]) + 1
            starts = np.r_[0, bounds]
            ends = np.r_[bounds, n]

            for gs, ge in zip(starts, ends):
                gsz = int(ge - gs)
                if gsz < 2:
                    continue
                idxs = order[gs:ge]
                if gsz > max_group:
                    step = np.linspace(0, gsz - 1, max_group).astype(int)
                    idxs = idxs[step]
                    gsz = max_group
                if gsz * (gsz - 1) // 2 > self._max_pairs_per_table:
                    continue
                for i, j in combinations(idxs.tolist(), 2):
                    pairs.add((i, j) if i < j else (j, i))
            if len(pairs) >= self._max_pairs_per_table * nt:
                break

        if not pairs:
            return set()

        pid = np.array(sorted(pairs))
        i, j = pid[:, 0], pid[:, 1]
        d = hamming_distances(self._bits[i], self._bits[j])
        ok = pid[d <= threshold]
        return {tuple(p) for p in ok.tolist()}

    def _bruteforce(self, threshold: int, cap: int = 500_000) -> set:
        """阈值超过每表位数时的兜底全量比较 (带上限)。"""
        n = self.size
        if n > 8000:
            # 全量太大, 用随机采样近似, 并提示
            rng = np.random.default_rng(0)
            sel = rng.choice(n, 4000, replace=False)
            sub_bits = self._bits[sel]
            pairs = set()
            for k in range(len(sel)):
                d = hamming_distances(sub_bits[k:k + 1], sub_bits[k + 1:])
                for j, dd in zip(sel[k + 1:], d):
                    if dd <= threshold:
                        pairs.add((int(sel[k]), int(j)))
                if len(pairs) > cap:
                    break
            return pairs
        pairs = set()
        for i in range(n):
            d = hamming_distances(self._bits[i:i + 1], self._bits[i + 1:])
            for j, dd in zip(range(i + 1, n), d):
                if dd <= threshold:
                    pairs.add((i, int(j)))
        return pairs


def rotate_hash_bits(gray_256: np.ndarray, kind: str, hash_fn):
    """对 256 灰度图施加旋转/翻转后重新计算 pHash, 返回位向量。

    kind: 'rot90' | 'rot180' | 'rot270' | 'flipH' | 'flipV'
          | 'rot15' (顺时针15°) | 'rot345' (逆时针15°)
    """
    from PIL import Image
    if kind == "rot90":
        arr = np.rot90(gray_256, k=1)
    elif kind == "rot180":
        arr = np.rot90(gray_256, k=2)
    elif kind == "rot270":
        arr = np.rot90(gray_256, k=3)
    elif kind == "flipH":
        arr = np.fliplr(gray_256)
    elif kind == "flipV":
        arr = np.flipud(gray_256)
    elif kind == "rot15":
        arr = _rotate_free(gray_256, 15)
    elif kind == "rot345":
        arr = _rotate_free(gray_256, -15)
    else:
        raise ValueError(kind)
    h = hash_fn(Image.fromarray(arr))
    return imagehash_to_bits(h)


def _rotate_free(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """自由角度旋转 (扩大画布 + 复制边框), 与真实旋转行为一致。"""
    import cv2
    h, w = gray.shape[:2]
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    nw = int(abs(w * c) + abs(h * s)) + 1
    nh = int(abs(w * s) + abs(h * c)) + 1
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(gray, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
