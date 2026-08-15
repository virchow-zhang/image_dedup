#!/usr/bin/env python3
"""
科研图片查重工具 - 优化版
========================
借鉴 xImageDuplicateChecker (MIT License) 的优化思路

主要优化：
1. 投票制检测：多个检测器独立投票，至少2种结构类检测器一致才判定为可疑
2. 尺寸预过滤：宽高比/面积差异过大的图片直接跳过
3. Jensen-Shannon直方图：更鲁棒的直方图相似度度量（作为参考票）
4. 哈希索引加速：LSH/前缀索引大幅减少比较次数（高位索引，修复原低位索引漏召回）
5. 增强的可视化：精确框出重复区域

降低误报率（v2）：
6. 窗口化SSIM：替代全局标量SSIM，消除“背景相似但内容不同”的虚高
7. 低信息量图过滤：空白/纯色图（灰度std过低）不参与感知匹配
8. 投票去相关：直方图降为参考票，pHash/dHash/SSIM 结构票为主
9. ORB二次确认：边缘通过的候选对用特征匹配复核，失败降级/剔除

交互性能（v2）：
10. 目录单次遍历（原32次rglob）、MD5分块流式、超大图draft降采样解码
11. 缩略图按路径缓存+并行生成（原每对重复生成2张、串行）
12. HTML报告：文件名搜索、分页、出现次数badge、Esc关闭全屏

用法：
    python image_dedup_optimized.py <图片目录> [选项]

示例：
    python image_dedup_optimized.py D:\\proj_test\\proj_1
    python image_dedup_optimized.py D:\\proj_test\\proj_1 --workers 8
    python image_dedup_optimized.py D:\\proj_test\\proj_1 --min-votes 3   # 更严格
"""

import os
import sys
import argparse
import hashlib
import json
import csv
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from PIL import Image
import imagehash

# ============================================================
# 配置
# ============================================================

# 支持的图片格式
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
    '.gif', '.webp', '.svs', '.ndpi', '.vsi'
}

# 默认参数
DEFAULT_HASH_SIZE = 16          # 哈希大小（越大越精确但越慢）
DEFAULT_PHASH_THRESHOLD = 5     # 感知哈希差异阈值（原3→5：配合窗口SSIM/ORB确认兜底，
                                # 放宽容度以召回亮度/对比度调整类副本；纯黑背景图亮度偏移
                                # 会使pHash低频位翻转，阈值3时实测漏报）
DEFAULT_SSIM_THRESHOLD = 0.92   # SSIM 相似度阈值（内容加权窗口SSIM）
SSIM_ZS_THRESHOLD = 0.98        # 亮度归一化SSIM阈值：结构几乎完全一致时
                                # 判定为"亮度/对比度/曝光调整"副本
DEFAULT_HIST_THRESHOLD = 0.88   # 直方图相似度阈值（原0.80→0.88）
DEFAULT_FEATURE_MIN_MATCH = 10  # 特征匹配最小点数
DEFAULT_SUBIMAGE_THRESHOLD = 0.90  # 子图匹配阈值
DEFAULT_WORKERS = 4             # 并行工作进程数

# 投票制参数：至少 N 种检测器一致才判定为可疑
VOTE_MIN_AGREE = 2              # 最少一致通过的检测器数

# LSH索引配置
LSH_NUM_HASH_TABLES = 10        # LSH哈希表数量（原6→10）
LSH_NUM_HASH_FUNCTIONS = 8      # 每个哈希表的哈希函数数量（原3→8）
LSH_INDEX_PREFIX_BITS = 20      # 索引前缀位数（原16→20）

# ---- 误报率控制 ----
LOW_INFO_STD = 8.0              # 灰度标准差阈值：低于此值的低信息量图（空白/纯色）
                                # 不参与感知匹配（仅保留MD5精确匹配），消除空白图误报
SSIM_WIN_SIZE = 11              # 窗口化SSIM的高斯窗口大小（标准SSIM参数）
SSIM_SIGMA = 1.5                # 窗口化SSIM的高斯sigma

# ORB 二次确认（仅对边缘通过的候选对运行，控制计算量）
ORB_MAX_DIM = 800               # ORB确认时图片最大边长（下采样）
ORB_NFEATURES = 800             # ORB特征点数量
ORB_MIN_GOOD_MATCHES = 8        # 确认通过所需最少好匹配点数
ORB_MIN_RATIO = 0.04            # 确认通过所需最小匹配比（好匹配/较少特征数）
ORB_CONFIRM_MAX_CONF = 0.75     # 仅对置信度低于此值的边缘对做ORB确认

# ---- 性能控制 ----
MAX_DECODE_DIM = 2048           # 图片解码最大边长：超大图（SVS/NDPI/巨TIFF）
                                # 先用 draft() 无损降采样，防内存爆炸

# ---- 旋转/翻转召回 ----
# 旋转副本的哈希与原件差异巨大，永远进不了普通LSH/前缀索引的候选桶，
# 导致"旋转检测"形同虚设。这里为每张图额外把 5 种几何变换后的哈希
# 加入一个轻量前缀索引，使旋转副本成为候选，再做旋转校验。
ROT_AUG_PREFIX_BITS = 12        # 旋转增强索引的前缀位数（宽松桶，保证召回）
ROT_MAX_PAIRS = 3000            # 旋转候选对数量上限（防止大库爆炸）
ROTATION_PHASH_THRESHOLD = 24   # 变换后phash16(256bit)差异阈值：
                                # 真旋转/翻转≤10，随机图对≥90，两侧余量充足

# ---- 子图/裁剪检测（快速版） ----
# 原版有 O(w*h*scale) 的滑动窗口模板匹配，慢到不可用；优化版直接删掉了
# 该能力。这里用"面积比过滤 + 降采样 + 多尺度 cv2.matchTemplate"恢复：
# 仅在面积差足够大、数量有上限的候选对间做一次快速模板匹配。
SUBIMAGE_MIN_AREA_RATIO = 1.7   # 面积比阈值：小于此值不可能是裁剪关系
SUBIMAGE_NCC_THRESHOLD = 0.85   # 归一化互相关阈值
SUBIMAGE_MAX_PAIRS = 1500       # 检查对数量上限（防大库爆炸）
SUBIMAGE_MAX_DIM = 160          # 匹配用降采样最大边长
SUBIMAGE_SCALES = (1.0, 0.85, 0.7, 0.55, 1.3, 1.6, 1.9)  # 覆盖裁剪缩放±30%及整图缩放


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ImageInfo:
    """图片信息"""
    path: str
    size: tuple           # (width, height)
    file_size: int        # 文件大小(bytes)
    md5: str              # 文件MD5
    phash: object = None  # 感知哈希
    dhash: object = None  # 差异哈希
    ahash: object = None  # 平均哈希
    chash: object = None  # 轮廓哈希
    hist: object = None   # 颜色直方图
    gray_array: object = None  # 灰度图数组(用于SSIM等)
    phash_int: int = 0    # pHash整数形式（用于索引）
    dhash_int: int = 0    # dHash整数形式（用于索引）


@dataclass
class DuplicateMatch:
    """重复匹配结果"""
    image1: str
    image2: str
    match_type: str       # 匹配类型
    similarity: float     # 相似度 (0~1)
    details: str = ""     # 详细说明
    severity: str = "medium"  # low / medium / high / critical
    confidence: float = 0.0   # 置信度 (0~1)，越高越可信
    vote_details: str = ""    # 投票详情，如 "pHash+SSIM"


class PerformanceStats:
    """性能统计"""
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()
    
    def reset(self):
        self.total_files = 0
        self.hash_calculation_time = 0
        self.index_building_time = 0
        self.comparison_time = 0
        self.total_comparisons = 0
        self.candidate_pairs = 0
        self.optimization_ratio = 0
        self.start_time = time.time()
    
    def update(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def increment(self, key, value=1):
        with self.lock:
            if hasattr(self, key):
                setattr(self, key, getattr(self, key) + value)
    
    def get_summary(self) -> str:
        total_time = time.time() - self.start_time
        return f"""
性能统计:
  总文件数: {self.total_files}
  哈希计算耗时: {self.hash_calculation_time:.2f}s
  索引构建耗时: {self.index_building_time:.2f}s
  比较检测耗时: {self.comparison_time:.2f}s
  总耗时: {total_time:.2f}s
  总比较次数: {self.total_comparisons:,}
  候选对数量: {self.candidate_pairs:,}
  优化比例: {self.optimization_ratio:.1f}x
"""


# ============================================================
# LSH索引（借鉴 xImageDuplicateChecker）
# ============================================================

class LSHIndex:
    """
    局部敏感哈希索引，用于快速相似图片检索
    
    LSH的核心思想：
    - 相似的输入有更高概率被映射到相同的桶中
    - 通过多个哈希函数和多个哈希表提高召回率
    - 大幅减少需要精确比较的候选对数量
    """
    
    def __init__(self, 
                 num_hash_tables: int = LSH_NUM_HASH_TABLES,
                 num_hash_functions: int = LSH_NUM_HASH_FUNCTIONS,
                 hash_length: int = 64):
        self.num_hash_tables = num_hash_tables
        self.num_hash_functions = num_hash_functions
        self.hash_length = hash_length
        
        # LSH哈希表
        self.hash_tables: List[Dict[str, List[Dict]]] = [
            defaultdict(list) for _ in range(num_hash_tables)
        ]
        
        # 随机投影矩阵
        np.random.seed(42)  # 固定种子确保可重现性
        self.projection_matrices = [
            np.random.randn(num_hash_functions, hash_length).astype(np.float32)
            for _ in range(num_hash_tables)
        ]
    
    def _hash_to_binary_vector(self, hash_obj) -> np.ndarray:
        """
        将哈希对象转换为二进制向量。

        注意：imagehash 的整数按"第一个元素为最高位(MSB)"打包，
        而 MSB 一侧对应 DCT 低频 / 图像粗结构，信息量最大。
        旧实现取的是低 hash_length 位（高频噪声位），导致 LSH 签名
        区分力差、召回率低。这里改为取高 hash_length 位。
        """
        hash_int = int(str(hash_obj), 16)
        nbits = hash_obj.hash.size  # 完整位数，如 hash_size=16 -> 256
        start = max(0, nbits - self.hash_length)
        binary = np.array([(hash_int >> (start + i)) & 1 for i in range(self.hash_length)],
                          dtype=np.float32)
        return binary * 2 - 1  # 转换为 -1/+1
    
    def _compute_signature(self, binary_vector: np.ndarray, table_idx: int) -> str:
        """计算LSH签名"""
        projections = self.projection_matrices[table_idx] @ binary_vector
        return ''.join('1' if p > 0 else '0' for p in projections)
    
    def add(self, file_info: Dict):
        """添加文件到索引"""
        for hash_type in ['phash', 'dhash']:
            if hash_type not in file_info or file_info[hash_type] is None:
                continue
            
            binary_vector = self._hash_to_binary_vector(file_info[hash_type])
            
            for table_idx in range(self.num_hash_tables):
                signature = self._compute_signature(binary_vector, table_idx)
                bucket_key = f"{hash_type}_{signature}"
                self.hash_tables[table_idx][bucket_key].append(file_info)
    
    def build(self, file_hashes: List[Dict]):
        """批量构建索引"""
        for file_info in file_hashes:
            self.add(file_info)
    
    def get_candidate_pairs(self) -> Set[Tuple[str, str]]:
        """获取所有候选对"""
        candidates = set()
        
        for table in self.hash_tables:
            for bucket_key, files in table.items():
                if len(files) < 2 or len(files) > 1000:  # 跳过太大或太小的桶
                    continue
                for i in range(len(files)):
                    for j in range(i + 1, len(files)):
                        path1 = files[i]['path']
                        path2 = files[j]['path']
                        pair = (min(path1, path2), max(path1, path2))
                        candidates.add(pair)
        
        return candidates


# ============================================================
# 哈希前缀索引（简单高效）
# ============================================================

class PrefixIndex:
    """按哈希前缀分组的索引"""
    
    def __init__(self, prefix_bits: int = LSH_INDEX_PREFIX_BITS):
        self.prefix_bits = prefix_bits
        self.mask = (1 << prefix_bits) - 1
        self.buckets: Dict[int, List[Dict]] = defaultdict(list)
    
    def add(self, file_info: Dict):
        """添加文件到索引"""
        # 使用dHash的高位前缀作为主索引（高位=粗结构，区分力强；
        # 旧实现取低位 mask，两个近重复图可能因低频位差异而分桶，漏召回）
        if 'dhash_int' in file_info:
            nbits = file_info['dhash'].hash.size
            prefix = (file_info['dhash_int'] >> max(0, nbits - self.prefix_bits)) & self.mask
            self.buckets[prefix].append(file_info)
    
    def build(self, file_hashes: List[Dict]):
        """批量构建索引"""
        for file_info in file_hashes:
            self.add(file_info)
    
    def get_candidate_pairs(self) -> Set[Tuple[str, str]]:
        """获取候选对"""
        candidates = set()
        
        for prefix, files in self.buckets.items():
            if len(files) < 2 or len(files) > 500:
                continue
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    path1 = files[i]['path']
                    path2 = files[j]['path']
                    pair = (min(path1, path2), max(path1, path2))
                    candidates.add(pair)
        
        return candidates
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        group_sizes = [len(group) for group in self.buckets.values()]
        return {
            'num_groups': len(self.buckets),
            'avg_group_size': np.mean(group_sizes) if group_sizes else 0,
            'max_group_size': max(group_sizes) if group_sizes else 0,
        }


# ============================================================
# 同实验通道过滤
# ============================================================

CHANNEL_SUFFIXES = ['_CH1', '_CH2', '_CH3', '_CH4', '_CH5',
                    '_Overlay', '_Bright', '_DIC', '_GFP', '_RFP',
                    '_DAPI', '_Cy3', '_Cy5', '_FITC', '_TRITC']


def _get_experiment_key(filepath: str) -> tuple:
    """提取实验标识"""
    path = Path(filepath)
    parent = str(path.parent)
    stem = path.stem
    
    base = stem
    for suffix in CHANNEL_SUFFIXES:
        if stem.upper().endswith(suffix.upper()):
            base = stem[:len(stem) - len(suffix)]
            break
    
    return (parent, base)


def _is_same_experiment_channel(filepath1: str, filepath2: str) -> bool:
    """判断两张图是否来自同一样本的不同通道"""
    key1 = _get_experiment_key(filepath1)
    key2 = _get_experiment_key(filepath2)
    
    if key1 != key2:
        return False
    
    return filepath1 != filepath2


# ============================================================
# 尺寸预过滤
# ============================================================

def _prefilter_by_size(info1: Dict, info2: Dict,
                       max_aspect_ratio_diff: float = 1.3,
                       max_area_ratio: float = 4.0) -> bool:
    """
    尺寸预过滤：如果宽高比或面积差异过大，直接跳过比较。
    科研图片造假者通常不会大幅改变宽高比。
    
    返回 True = 可以继续比较，False = 跳过
    """
    w1, h1 = info1['size']
    w2, h2 = info2['size']
    
    # 宽高比检查
    aspect1 = w1 / h1 if h1 > 0 else 1.0
    aspect2 = w2 / h2 if h2 > 0 else 1.0
    aspect_ratio = max(aspect1, aspect2) / min(aspect1, aspect2)
    if aspect_ratio > max_aspect_ratio_diff:
        return False
    
    # 面积检查
    area1 = w1 * h1
    area2 = w2 * h2
    area_ratio = max(area1, area2) / min(area1, area2)
    if area_ratio > max_area_ratio:
        return False
    
    return True


def _prefilter_by_color_uniqueness(info1: Dict, info2: Dict,
                                   min_color_std: float = 15.0) -> bool:
    """
    颜色唯一性预过滤：如果两张图都是低色彩变化（纯色/渐变），
    但平均色调差异很大，则不太可能是重复或抄袭。
    
    hist 存储的是 192-bin (R,G,B各64bin) 全局归一化直方图，
    需要逐通道重新归一化后再计算颜色矩。
    """
    try:
        # 拆分并归一化各通道直方图
        bins = 64
        eps = 1e-10
        r_hist = info1['hist'][:bins] / (info1['hist'][:bins].sum() + eps)
        g_hist = info1['hist'][bins:2*bins] / (info1['hist'][bins:2*bins].sum() + eps)
        b_hist = info1['hist'][2*bins:] / (info1['hist'][2*bins:].sum() + eps)
        
        r_hist2 = info2['hist'][:bins] / (info2['hist'][:bins].sum() + eps)
        g_hist2 = info2['hist'][bins:2*bins] / (info2['hist'][bins:2*bins].sum() + eps)
        b_hist2 = info2['hist'][2*bins:] / (info2['hist'][2*bins:].sum() + eps)
        
        bin_centers = np.arange(bins) * (256 / bins) + (256 / bins / 2)
        
        def _color_moments(rh, gh, bh):
            r_mean = np.sum(rh * bin_centers)
            g_mean = np.sum(gh * bin_centers)
            b_mean = np.sum(bh * bin_centers)
            r_std = np.sqrt(np.sum(rh * (bin_centers - r_mean)**2))
            g_std = np.sqrt(np.sum(gh * (bin_centers - g_mean)**2))
            b_std = np.sqrt(np.sum(bh * (bin_centers - b_mean)**2))
            return (r_mean, g_mean, b_mean), (r_std + g_std + b_std) / 3
        
        (r1, g1, b1), std1 = _color_moments(r_hist, g_hist, b_hist)
        (r2, g2, b2), std2 = _color_moments(r_hist2, g_hist2, b_hist2)
        
        # 如果两张图的颜色丰富度都低于阈值（都是纯色/渐变）
        if std1 < min_color_std and std2 < min_color_std:
            color_dist = np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
            if color_dist > 80:
                return False
    except:
        pass
    
    return True


# ============================================================
# 图片加载与预处理（并行版）
# ============================================================

def _md5_streaming(filepath: str, chunk_size: int = 1 << 20) -> str:
    """分块流式计算MD5，避免把数GB的SVS/NDPI/TIFF一次性读入内存"""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_single_image(filepath: str, hash_size: int = DEFAULT_HASH_SIZE) -> Optional[Dict]:
    """加载单张图片并计算特征（用于并行处理）"""
    try:
        img = Image.open(filepath)

        # 超大图保护：先尝试 draft() 无损降采样解码（JPEG/TIFF系列有效），
        # 避免 SVS/NDPI 等全玻片图 convert/np.array 时内存爆炸
        if max(img.size) > MAX_DECODE_DIM:
            try:
                img.draft('RGB', (MAX_DECODE_DIM, MAX_DECODE_DIM))
            except Exception:
                pass
            # draft 不生效的格式（如超大PNG）：再强制缩略，限制后续数组规模
            try:
                if max(img.size) > MAX_DECODE_DIM:
                    img.thumbnail((MAX_DECODE_DIM, MAX_DECODE_DIM), Image.LANCZOS)
            except Exception:
                pass

        if hasattr(img, 'n_frames') and img.n_frames > 1:
            img.seek(0)

        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        elif img.mode == 'L':
            img = img.convert('RGB')

        size = img.size
        file_size = os.path.getsize(filepath)

        # 计算MD5（分块流式）
        md5 = _md5_streaming(filepath)

        # 计算各种哈希
        gray = img.convert('L')
        phash = imagehash.phash(gray, hash_size=hash_size)
        dhash = imagehash.dhash(gray, hash_size=hash_size)
        ahash = imagehash.average_hash(gray, hash_size=hash_size)
        chash = imagehash.colorhash(img)

        # 计算颜色直方图
        hist_r = np.histogram(np.array(img)[:,:,0], bins=64, range=(0,256))[0]
        hist_g = np.histogram(np.array(img)[:,:,1], bins=64, range=(0,256))[0]
        hist_b = np.histogram(np.array(img)[:,:,2], bins=64, range=(0,256))[0]
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
        hist = hist / hist.sum()

        # 灰度图数组
        gray_array = np.array(gray.resize((256, 256)))
        # 灰度标准差：低于阈值的图信息量过低（空白/纯色），
        # 是"两张任意空白图互报 critical"类误报的根源
        gray_std = float(gray_array.std())

        # 等比缩放入256x256画布（保持宽高比，黑边补齐）：
        # 旋转/翻转比较用。拉伸成方形的 gray_array 会引入各向异性变形，
        # 导致非方形图的旋转副本永远匹配不上（90°旋转后变形方向互换）。
        rot_gray = gray.copy()
        rot_gray.thumbnail((256, 256), Image.LANCZOS)
        rot_canvas = Image.new('L', (256, 256), 0)
        rot_canvas.paste(rot_gray, ((256 - rot_gray.size[0]) // 2,
                                    (256 - rot_gray.size[1]) // 2))
        rot_array = np.array(rot_canvas)

        # 转换为整数用于索引
        phash_int = int(str(phash), 16)
        dhash_int = int(str(dhash), 16)

        return {
            'path': filepath,
            'size': size,
            'file_size': file_size,
            'md5': md5,
            'phash': phash,
            'dhash': dhash,
            'ahash': ahash,
            'chash': chash,
            'hist': hist,
            'gray_array': gray_array,
            'gray_std': gray_std,
            'rot_array': rot_array,
            'phash_int': phash_int,
            'dhash_int': dhash_int,
        }
    except Exception as e:
        return None


def scan_directory_parallel(root_dir: str, hash_size: int = DEFAULT_HASH_SIZE,
                           workers: int = DEFAULT_WORKERS) -> Tuple[List[Dict], PerformanceStats]:
    """并行扫描目录，加载所有图片"""
    root = Path(root_dir)
    stats = PerformanceStats()
    
    # 需要排除的目录
    exclude_dirs = {'visualization', 'venv', '.git', '__pycache__', 'node_modules', '_thumbnails'}
    
    print(f"\n[1/4] 扫描目录: {root_dir}")
    
    # 收集所有图片路径
    # 旧实现按扩展名逐个 rglob（16种×大小写=32次全树遍历），大目录下极慢；
    # 改为单次遍历 + 后缀过滤
    image_paths = []
    lower_exts = {ext.lower() for ext in IMAGE_EXTENSIONS}
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in lower_exts:
            continue
        parts = p.relative_to(root).parts
        if not any(part in exclude_dirs for part in parts):
            image_paths.append(str(p))
    
    image_paths = sorted(set(image_paths))
    print(f"  找到 {len(image_paths)} 张图片")
    
    if not image_paths:
        print("  未找到任何图片文件！")
        return [], stats
    
    # 并行计算哈希
    print(f"\n[2/4] 并行计算图片哈希 (使用 {workers} 个线程)...")
    start_time = time.time()
    
    images = []
    failed = 0
    completed = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(load_single_image, path, hash_size): path 
            for path in image_paths
        }
        
        for future in as_completed(future_to_path):
            completed += 1
            result = future.result()
            
            if result:
                images.append(result)
            else:
                failed += 1
            
            if completed % 100 == 0 or completed == len(image_paths):
                print(f"  进度: {completed}/{len(image_paths)} ({completed/len(image_paths)*100:.1f}%)")
    
    hash_time = time.time() - start_time
    stats.update(
        total_files=len(images),
        hash_calculation_time=hash_time
    )
    
    print(f"  成功加载: {len(images)}, 失败: {failed}")
    print(f"  哈希计算耗时: {hash_time:.2f}s")
    
    return images, stats


# ============================================================
# 检测方法
# ============================================================

def _hamming_distance(h1, h2):
    """计算两个哈希之间的汉明距离"""
    return h1 - h2


def check_exact_duplicate(info1: Dict, info2: Dict) -> Optional[DuplicateMatch]:
    """检测完全相同的文件"""
    if info1['md5'] == info2['md5']:
        return DuplicateMatch(
            image1=info1['path'],
            image2=info2['path'],
            match_type="完全相同",
            similarity=1.0,
            details=f"MD5一致: {info1['md5']}",
            severity="critical"
        )
    return None


def check_phash_similarity(info1: Dict, info2: Dict,
                           threshold: int = DEFAULT_PHASH_THRESHOLD) -> Optional[DuplicateMatch]:
    """检测感知哈希相似度"""
    diff = _hamming_distance(info1['phash'], info2['phash'])
    if diff <= threshold:
        sim = 1.0 - diff / (info1['phash'].hash.size)
        return DuplicateMatch(
            image1=info1['path'],
            image2=info2['path'],
            match_type="感知哈希相似(pHash)",
            similarity=round(sim, 4),
            details=f"哈希差异: {diff}/{info1['phash'].hash.size}",
            severity="critical" if diff <= 2 else "high"
        )
    return None


def check_dhash_similarity(info1: Dict, info2: Dict,
                           threshold: int = DEFAULT_PHASH_THRESHOLD) -> Optional[DuplicateMatch]:
    """检测差异哈希相似度"""
    diff = _hamming_distance(info1['dhash'], info2['dhash'])
    if diff <= threshold:
        sim = 1.0 - diff / (info1['dhash'].hash.size)
        return DuplicateMatch(
            image1=info1['path'],
            image2=info2['path'],
            match_type="差异哈希相似(dHash)",
            similarity=round(sim, 4),
            details=f"哈希差异: {diff}/{info1['dhash'].hash.size}",
            severity="critical" if diff <= 2 else "high"
        )
    return None


def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """一维高斯核"""
    x = np.arange(size, dtype=np.float64) - size // 2
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _separable_convolve2d(arr: np.ndarray, kernel1d: np.ndarray) -> np.ndarray:
    """
    可分离二维卷积（reflect边界，与 scikit-image 的 SSIM 一致）。
    用 sliding_window_view + tensordot 向量化，256x256 输入约几毫秒。
    """
    pad = len(kernel1d) // 2
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        a = np.pad(arr, ((pad, pad), (0, 0)), mode='reflect')
        win = sliding_window_view(a, len(kernel1d), axis=0)
        out = np.tensordot(win, kernel1d, axes=([2], [0]))
        a2 = np.pad(out, ((0, 0), (pad, pad)), mode='reflect')
        win2 = sliding_window_view(a2, len(kernel1d), axis=1)
        return np.tensordot(win2, kernel1d, axes=([2], [0]))
    except ImportError:
        # 老版本numpy兜底：逐行/列卷积
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            out[i] = np.convolve(arr[i], kernel1d, mode='same')
        for j in range(arr.shape[1]):
            out[:, j] = np.convolve(out[:, j], kernel1d, mode='same')
        return out


def compute_ssim(img1_gray: np.ndarray, img2_gray: np.ndarray,
                 data_range: float = 255.0) -> float:
    """
    计算标准窗口化 SSIM（内容加权）。

    旧实现是"全局标量SSIM"：整张图只算一个均值/方差，科研图通常大片背景
    （黑/白/灰）占比很高，两张内容完全不同的图只要背景相似，全局SSIM就会
    虚高到 0.9+，是最大误报源。

    这里用 11x11 高斯窗口逐窗口比较，并按窗口局部方差加权求均值：
    大片均匀背景窗口权重≈0，内容区域（条带/细胞/斑块）主导最终分数。
    否则"两张黑背景+不同内容"的图仍会因背景窗口占多数而虚高（实测0.96）。
    """
    img1 = img1_gray.astype(np.float64)
    img2 = img2_gray.astype(np.float64)

    kernel = _gaussian_kernel_1d(SSIM_WIN_SIZE, SSIM_SIGMA)
    mu1 = _separable_convolve2d(img1, kernel)
    mu2 = _separable_convolve2d(img2, kernel)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _separable_convolve2d(img1 * img1, kernel) - mu1_sq
    sigma2_sq = _separable_convolve2d(img2 * img2, kernel) - mu2_sq
    sigma12 = _separable_convolve2d(img1 * img2, kernel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-10)

    # 内容加权：权重=窗口平均方差，避免大片均匀背景主导均值
    weight = (sigma1_sq + sigma2_sq) / 2.0
    wsum = weight.sum()
    if wsum < 1e-6:
        return float(ssim_map.mean())
    return float((ssim_map * weight).sum() / wsum)


def compute_ssim_zs(img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
    """
    亮度/对比度归一化窗口SSIM（结构一致性检测专用）。

    先对两张图各自做全局 z-score（去均值/方差），再计算内容加权窗口SSIM。
    对"同结构不同曝光"的副本（亮度/对比度/曝光调整）≈1.0，
    对"内容不同但背景风格相似"的图≈0甚至为负，是判别
    "亮度/对比度调整"类复制的专用信号。
    """
    img1 = img1_gray.astype(np.float64)
    img2 = img2_gray.astype(np.float64)
    for arr in (img1, img2):
        m = arr.mean()
        s = arr.std()
        if s > 1e-6:
            arr -= m
            arr /= s

    kernel = _gaussian_kernel_1d(SSIM_WIN_SIZE, SSIM_SIGMA)
    mu1 = _separable_convolve2d(img1, kernel)
    mu2 = _separable_convolve2d(img2, kernel)
    sigma1_sq = _separable_convolve2d(img1 * img1, kernel) - mu1 * mu1
    sigma2_sq = _separable_convolve2d(img2 * img2, kernel) - mu2 * mu2
    sigma12 = _separable_convolve2d(img1 * img2, kernel) - mu1 * mu2

    C1 = 0.01 ** 2   # 单位方差数据
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-10)

    weight = (sigma1_sq + sigma2_sq) / 2.0
    wsum = weight.sum()
    if wsum < 1e-6:
        return float(ssim_map.mean())
    return float((ssim_map * weight).sum() / wsum)


def check_ssim_similarity(info1: Dict, info2: Dict,
                          threshold: float = DEFAULT_SSIM_THRESHOLD) -> Optional[DuplicateMatch]:
    """检测结构相似性"""
    ssim = compute_ssim(info1['gray_array'], info2['gray_array'])

    if ssim >= threshold:
        return DuplicateMatch(
            image1=info1['path'],
            image2=info2['path'],
            match_type="结构相似(SSIM)",
            similarity=round(ssim, 4),
            details=f"SSIM值: {ssim:.4f}",
            severity="critical" if ssim > 0.95 else "high"
        )
    return None


def check_histogram_similarity(info1: Dict, info2: Dict,
                               threshold: float = DEFAULT_HIST_THRESHOLD) -> Optional[DuplicateMatch]:
    """检测颜色直方图相似度"""
    hist1 = info1['hist'] - info1['hist'].mean()
    hist2 = info2['hist'] - info2['hist'].mean()

    correlation = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2) + 1e-10)
    bhattacharyya = np.sum(np.sqrt(info1['hist'] * info2['hist']))
    avg_sim = (max(0, correlation) + bhattacharyya) / 2

    if avg_sim >= threshold:
        details = f"直方图相关性: {correlation:.3f}, 巴氏系数: {bhattacharyya:.3f}"
        if correlation > 0.95:
            details += " [疑似曝光度/亮度统一调整]"
        return DuplicateMatch(
            image1=info1['path'],
            image2=info2['path'],
            match_type="直方图相似(疑似曝光调整)",
            similarity=round(avg_sim, 4),
            details=details,
            severity="high" if avg_sim > 0.95 else "medium"
        )
    return None


def check_brightness_contrast(info1: Dict, info2: Dict,
                              threshold: float = 0.90) -> Optional[DuplicateMatch]:
    """检测亮度/对比度调整"""
    try:
        arr1 = info1['gray_array'].astype(np.float64)
        arr2 = info2['gray_array'].astype(np.float64)

        mean1, std1 = arr1.mean(), arr1.std()
        mean2, std2 = arr2.mean(), arr2.std()

        brightness_ratio = min(mean1, mean2) / (max(mean1, mean2) + 1e-10)
        contrast_ratio = min(std1, std2) / (max(std1, std2) + 1e-10)

        arr1_norm = (arr1 - mean1) / (std1 + 1e-10)
        arr2_norm = (arr2 - mean2) / (std2 + 1e-10)
        structure_sim = np.mean(arr1_norm * arr2_norm)
        structure_sim = max(0, structure_sim)

        if structure_sim >= threshold and (brightness_ratio < 0.9 or contrast_ratio < 0.9):
            return DuplicateMatch(
                image1=info1['path'],
                image2=info2['path'],
                match_type="疑似亮度/对比度调整",
                similarity=round(structure_sim, 4),
                details=f"结构相似度:{structure_sim:.3f}, 亮度比:{brightness_ratio:.3f}, 对比度比:{contrast_ratio:.3f}",
                severity="high"
            )
    except:
        pass
    return None


def check_rotation_flip(info1: Dict, info2: Dict,
                        threshold: int = ROTATION_PHASH_THRESHOLD) -> List[DuplicateMatch]:
    """
    检测旋转/翻转（优化版）

    用 phash hash_size=16（256-bit）做变换后比较：
    - 实测 whash8(64-bit) 对"深色背景+稀疏条带"图无区分度（真实旋转与
      随机图对的差异区间重叠），误报率极高；
    - phash16 真实旋转/翻转差异 ≤10，随机图对差异 ≥90，阈值24两侧
      余量充足（阈值可经 ROTATION_PHASH_THRESHOLD 调整）。

    直接使用加载阶段缓存的 rot_array（等比缩放+黑边补齐的256x256），
    避免重复读盘，也避免"拉伸成方形"导致的非方形图旋转比较失真。
    """
    results = []

    try:
        img1 = Image.fromarray(info1['rot_array'])
        img2 = Image.fromarray(info2['rot_array'])
    except Exception:
        return results

    transforms = {
        "旋转90°": lambda im: im.rotate(90, expand=False),
        "旋转180°": lambda im: im.rotate(180, expand=False),
        "旋转270°": lambda im: im.rotate(270, expand=False),
        "水平翻转": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        "垂直翻转": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    }

    # 用 phash 而不是 whash：whash 在小尺寸(8x8)下对稀疏条带图无区分力
    try:
        hash2 = imagehash.phash(img2, hash_size=16)
    except:
        return results

    for name, transform_fn in transforms.items():
        try:
            transformed = transform_fn(img1)
            hash_t = imagehash.phash(transformed, hash_size=16)
            diff = _hamming_distance(hash_t, hash2)
            if diff <= threshold:
                sim = 1.0 - diff / 256  # 256-bit hash
                results.append(DuplicateMatch(
                    image1=info1['path'],
                    image2=info2['path'],
                    match_type=f"疑似{name}",
                    similarity=round(sim, 4),
                    details=f"变换后哈希差异: {diff}",
                    severity="critical",
                    confidence=round(sim * 0.95, 4)
                ))
        except:
            continue

    return results


# ============================================================
# 投票制检测（核心改进）
# ============================================================

@dataclass
class VoteResult:
    """单个检测器的投票结果"""
    passed: bool
    similarity: float
    match_type: str
    severity: str
    details: str


def check_pair_voting(info1: Dict, info2: Dict,
                      pthresh: int = DEFAULT_PHASH_THRESHOLD,
                      sthresh: float = DEFAULT_SSIM_THRESHOLD,
                      hthresh: float = DEFAULT_HIST_THRESHOLD,
                      min_votes: int = VOTE_MIN_AGREE,
                      check_rotations: bool = True,
                      orb_confirm: bool = True,
                      orb_strict: bool = False,
                      min_std: float = LOW_INFO_STD) -> Optional[DuplicateMatch]:
    """
    投票制检测：多个检测器独立投票，至少 min_votes 个结构类检测器通过才判定。

    相比原版的重要改进（均为降低误报率）：
    1. 低信息量图过滤：空白/纯色图（灰度标准差过低）不参与感知匹配，
       消除"两张任意空白图互报 critical"的经典误报。
    2. 投票去相关：pHash 与 dHash 同源于灰度图、高度相关；直方图又是纯
       统计信号。原版"4 票里凑 2 票"实际可能是"1 个独立信号+1 个统计巧合"。
       现改为 pHash/dHash/SSIM 三个结构类检测器投票（--min-votes 指结构票数），
       直方图降为参考票：通过则小幅加分，不通过则对边缘对打折。
    3. ORB 二次确认：恰好 min_votes 票、置信度又不高的边缘候选对，
       用 ORB 特征匹配独立确认；确认失败默认降一级严重程度（--orb-strict
       则直接剔除），确认成功小幅提升置信度。
    """
    # ---- 低信息量图过滤 ----
    if min(info1.get('gray_std', 255.0), info2.get('gray_std', 255.0)) < min_std:
        return None  # 空白/纯色图只走MD5精确匹配，避免误报

    # ---- 尺寸预过滤 ----
    if not _prefilter_by_size(info1, info2):
        return None

    # ---- 颜色唯一性预过滤 ----
    if not _prefilter_by_color_uniqueness(info1, info2):
        return None

    votes: List[VoteResult] = []

    # ---- 1. pHash 投票（结构类） ----
    phash_diff = _hamming_distance(info1['phash'], info2['phash'])
    phash_passed = phash_diff <= pthresh
    phash_sim = 1.0 - phash_diff / info1['phash'].hash.size if phash_diff <= pthresh * 2 else 0.0
    votes.append(VoteResult(
        passed=phash_passed,
        similarity=round(phash_sim, 4),
        match_type="pHash",
        severity="critical" if phash_diff <= 2 else "high",
        details=f"差异: {phash_diff}/{info1['phash'].hash.size}"
    ))

    # ---- 2. dHash 投票（结构类） ----
    dhash_diff = _hamming_distance(info1['dhash'], info2['dhash'])
    dhash_passed = dhash_diff <= pthresh
    dhash_sim = 1.0 - dhash_diff / info1['dhash'].hash.size if dhash_diff <= pthresh * 2 else 0.0
    votes.append(VoteResult(
        passed=dhash_passed,
        similarity=round(dhash_sim, 4),
        match_type="dHash",
        severity="critical" if dhash_diff <= 2 else "high",
        details=f"差异: {dhash_diff}/{info1['dhash'].hash.size}"
    ))

    # ---- 3. SSIM 投票（结构类，内容加权窗口SSIM） ----
    ssim_val = compute_ssim(info1['gray_array'], info2['gray_array'])
    # 亮度/对比度归一化变体：专门捕获"同结构不同曝光"的调整副本
    ssim_zs_val = compute_ssim_zs(info1['gray_array'], info2['gray_array'])
    ssim_passed = ssim_val >= sthresh or ssim_zs_val >= SSIM_ZS_THRESHOLD
    ssim_details = f"SSIM值: {ssim_val:.4f}"
    if ssim_zs_val >= SSIM_ZS_THRESHOLD:
        ssim_details += f" (亮度归一化SSIM: {ssim_zs_val:.4f})"
    votes.append(VoteResult(
        passed=ssim_passed,
        similarity=round(max(ssim_val, ssim_zs_val), 4),
        match_type="SSIM",
        severity="critical" if ssim_val > 0.95 else "high",
        details=ssim_details
    ))

    # ---- 4. 直方图（参考票，不参与 min_votes 计数） ----
    hist_result = _check_histogram_js(info1['hist'], info2['hist'], hthresh)
    if hist_result is not None:
        hist_passed, hist_sim, hist_details = hist_result
    else:
        hist_passed, hist_sim, hist_details = False, 0.0, ""
    hist_vote = VoteResult(
        passed=hist_passed,
        similarity=round(hist_sim, 4),
        match_type="直方图",
        severity="medium",
        details=hist_details
    )

    # ---- 统计结构类投票结果 ----
    passed_votes = [v for v in votes if v.passed]
    structural_votes = len(passed_votes)

    if structural_votes < min_votes:
        return None  # 结构票不足，不是可疑重复

    # ---- 计算综合置信度 ----
    # 用通过结构检测器的平均相似度 × 结构票比例
    avg_sim = np.mean([v.similarity for v in passed_votes]) if passed_votes else 0.0
    vote_ratio = structural_votes / len(votes)
    confidence = avg_sim * (0.5 + 0.5 * vote_ratio)

    # 直方图参考票：通过→加分；边缘通过但直方图不合→打折
    if hist_passed:
        confidence = min(1.0, confidence + 0.05)
    elif structural_votes == min_votes:
        confidence *= 0.85

    # ---- 确定最终类型和严重程度 ----
    passed_names = [v.match_type for v in passed_votes]
    vote_details_str = "+".join(passed_names)

    # 按优先级取第一个通过的结构检测器的严重程度
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    best_severity = min(passed_votes, key=lambda v: severity_order.get(v.severity, 9)).severity

    # 构建详细说明
    details_parts = [f"结构票通过: {structural_votes}/{len(votes)}"]
    details_parts.extend([f"{v.match_type}(相似:{v.similarity:.3f})" for v in passed_votes])
    if hist_passed:
        details_parts.append(f"直方图(参考,相似:{hist_sim:.3f})")
    all_details = " | ".join(details_parts)

    # ---- 亮度/对比度特别标记 ----
    bc_note = _check_brightness_contrast_ratio(info1, info2)
    if bc_note:
        all_details += " | " + bc_note

    # ---- 旋转/翻转检测（仅在投票通过后执行，降低计算量） ----
    rotation_found = False
    if check_rotations:
        rot_matches = check_rotation_flip(info1, info2)
        if rot_matches:
            rotation_found = True
            best_rot = max(rot_matches, key=lambda m: m.similarity)
            all_details += f" | {best_rot.match_type}(差异:{best_rot.details})"
            vote_details_str += f"+{best_rot.match_type}"
            # 旋转检测通过说明存在变换关系，按匹配质量加成
            confidence = confidence * (0.95 + 0.15 * best_rot.similarity)

    # ---- ORB 二次确认（仅边缘通过 + 低置信度，控制计算量） ----
    if (orb_confirm and structural_votes == min_votes
            and not rotation_found and confidence < ORB_CONFIRM_MAX_CONF):
        orb_ratio = _orb_confirm(info1, info2)
        if orb_ratio is not None:
            if orb_ratio >= ORB_MIN_RATIO:
                all_details += f" | ORB确认(匹配比:{orb_ratio:.2f})"
                confidence = min(1.0, confidence * 1.05)
            else:
                if orb_strict:
                    return None  # 严格模式：ORB不确认直接剔除
                all_details += f" | ORB未确认(匹配比:{orb_ratio:.2f})"
                # 降一级严重程度
                downgrade = {"critical": "high", "high": "medium", "medium": "low"}
                best_severity = downgrade.get(best_severity, best_severity)
                confidence *= 0.7

    confidence = min(confidence, 1.0)  # 不超过1.0

    return DuplicateMatch(
        image1=info1['path'],
        image2=info2['path'],
        match_type=f"投票一致({vote_details_str})",
        similarity=round(avg_sim, 4),
        details=all_details,
        severity=best_severity,
        confidence=round(confidence, 4),
        vote_details=vote_details_str
    )


def _load_downscaled_gray(filepath: str, max_dim: int = ORB_MAX_DIM) -> Optional[np.ndarray]:
    """加载灰度图并限制最大边长（用于ORB确认等低频次计算）"""
    try:
        img = Image.open(filepath).convert('L')
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None


def _orb_confirm(info1: Dict, info2: Dict) -> Optional[float]:
    """
    ORB 特征点二次确认：返回好匹配比（good/total特征较少数），
    无OpenCV或特征不足时返回 None（表示无法判断，不否决）。
    ORB 是独立于哈希/SSIM 的信号，用于给"边缘通过"的候选对把关。
    """
    try:
        import cv2
    except ImportError:
        return None

    try:
        g1 = _load_downscaled_gray(info1['path'], ORB_MAX_DIM)
        g2 = _load_downscaled_gray(info2['path'], ORB_MAX_DIM)
        if g1 is None or g2 is None:
            return None

        orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)

        if des1 is None or des2 is None or len(kp1) < ORB_MIN_GOOD_MATCHES or len(kp2) < ORB_MIN_GOOD_MATCHES:
            return None  # 特征太少（低纹理图），无法确认也不否决

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        good = [m for m in matches if m.distance < 60]
        ratio = len(good) / min(len(kp1), len(kp2))
        return float(ratio)
    except Exception:
        return None


def _check_histogram_js(hist1: np.ndarray, hist2: np.ndarray,
                        threshold: float = DEFAULT_HIST_THRESHOLD) -> Optional[Tuple[bool, float, str]]:
    """
    使用 Jensen-Shannon 散度检测直方图相似度。
    JS散度是对称的、有界的（0~log2），比巴氏系数更鲁棒。
    """
    # 避免除零
    eps = 1e-10
    p = hist1 + eps
    q = hist2 + eps
    p = p / p.sum()
    q = q / q.sum()
    
    # KL散度
    m = (p + q) / 2
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    js_div = (kl_pm + kl_qm) / 2  # JS散度，范围 [0, log2≈1.0]
    
    # 转换为相似度 (1 - normalized JS divergence)
    js_sim = 1.0 - js_div  # 接近1 = 非常相似
    
    # 同时计算 Pearson 相关系数（保留负相关信号）
    p_centered = p - p.mean()
    q_centered = q - q.mean()
    corr = np.dot(p_centered, q_centered) / (np.linalg.norm(p_centered) * np.linalg.norm(q_centered) + eps)
    
    # 综合：JS相似度为主，相关性为辅
    # 如果相关性为负，大幅降低分数
    combined = js_sim
    if corr < 0:
        combined *= max(0, 1.0 + corr)  # 负相关越强，扣分越多
    
    if combined >= threshold:
        details = f"JS散度相似:{js_sim:.3f}, 相关性:{corr:.3f}"
        if corr > 0.95:
            details += " [疑似曝光度/亮度统一调整]"
        return (True, combined, details)
    
    return None


def _check_brightness_contrast_ratio(info1: Dict, info2: Dict) -> Optional[str]:
    """检测亮度/对比度差异，返回描述信息（不单独作为投票）"""
    try:
        arr1 = info1['gray_array'].astype(np.float64)
        arr2 = info2['gray_array'].astype(np.float64)
        mean1, std1 = arr1.mean(), arr1.std()
        mean2, std2 = arr2.mean(), arr2.std()
        brightness_ratio = min(mean1, mean2) / (max(mean1, mean2) + 1e-10)
        contrast_ratio = min(std1, std2) / (max(std1, std2) + 1e-10)
        
        notes = []
        if brightness_ratio < 0.9:
            notes.append(f"亮度比:{brightness_ratio:.3f}")
        if contrast_ratio < 0.9:
            notes.append(f"对比度比:{contrast_ratio:.3f}")
        
        if notes:
            return "疑似亮度/对比度调整: " + ", ".join(notes)
    except:
        pass
    return None


# ============================================================
# 结果去重与排序
# ============================================================

def deduplicate_matches(matches: List[DuplicateMatch]) -> List[DuplicateMatch]:
    """去除重复的匹配结果（按严重程度→置信度→相似度排序）"""
    seen = set()
    unique = []

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    for m in matches:
        if m.image1 == m.image2:
            key = (m.image1, m.image2, m.match_type)
        else:
            key = (tuple(sorted([m.image1, m.image2])), m.match_type)

        if key not in seen:
            seen.add(key)
            unique.append(m)

    # 按严重程度（升序）→ 置信度（降序）→ 相似度（降序）
    unique.sort(key=lambda m: (severity_order.get(m.severity, 9), -m.confidence, -m.similarity))
    return unique


# ============================================================
# 可视化
# ============================================================

def _extract_source_info(filepath: str) -> dict:
    """从文件路径提取来源信息"""
    path = Path(filepath)
    parts = path.parts

    info = {
        "group": "",
        "subfolder": "",
        "filename": path.name,
        "channel": ""
    }

    stem = path.stem.upper()
    for ch in ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'OVERLAY', 'BRIGHT', 'DIC', 'GFP', 'RFP', 'DAPI']:
        if ch in stem:
            info["channel"] = ch
            break

    for i, part in enumerate(parts):
        if part.upper().startswith('MOUSE') or part.upper().startswith('NOUSE'):
            info["group"] = part
            if i + 1 < len(parts) - 1:
                info["subfolder"] = parts[i + 1]
            break
        elif part.lower().startswith('mouse') or part.lower().startswith('nouse'):
            info["group"] = part
            if i + 1 < len(parts) - 1:
                info["subfolder"] = parts[i + 1]
            break

    return info


def _find_difference_region(img1_gray: np.ndarray, img2_gray: np.ndarray):
    """通过差异图找到两张相似图片的不同区域"""
    try:
        import cv2
        h, w = 400, 600
        img1 = cv2.resize(img1_gray, (w, h))
        img2 = cv2.resize(img2_gray, (w, h))
        
        diff = cv2.absdiff(img1, img2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((7,7), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            all_points = np.vstack(contours)
            x, y, bw, bh = cv2.boundingRect(all_points)
            if bw > 20 and bh > 20:
                return (x, y, bw, bh, w, h)
    except:
        pass
    return None


def create_visualization(match: DuplicateMatch, output_dir: str) -> Optional[str]:
    """为一对相似图片创建可视化对比图"""
    try:
        import cv2
    except ImportError:
        return None

    try:
        img1 = cv2.imread(match.image1)
        img2 = cv2.imread(match.image2)

        if img1 is None or img2 is None:
            return None

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        src1 = _extract_source_info(match.image1)
        src2 = _extract_source_info(match.image2)

        target_h = 450
        scale1 = target_h / img1.shape[0]
        scale2 = target_h / img2.shape[0]
        disp1 = cv2.resize(img1, (int(img1.shape[1] * scale1), target_h))
        disp2 = cv2.resize(img2, (int(img2.shape[1] * scale2), target_h))
        
        # 使用差异图找不同区域
        diff_region = _find_difference_region(gray1, gray2)
        if diff_region:
            x, y, w, h, ref_w, ref_h = diff_region
            sx1 = int(x * (disp1.shape[1] / ref_w))
            sy1 = int(y * (disp1.shape[0] / ref_h))
            sw1 = int(w * (disp1.shape[1] / ref_w))
            sh1 = int(h * (disp1.shape[0] / ref_h))
            cv2.rectangle(disp1, (sx1, sy1), (sx1+sw1, sy1+sh1), (0, 0, 255), 3)
            
            sx2 = int(x * (disp2.shape[1] / ref_w))
            sy2 = int(y * (disp2.shape[0] / ref_h))
            sw2 = int(w * (disp2.shape[1] / ref_w))
            sh2 = int(h * (disp2.shape[0] / ref_h))
            cv2.rectangle(disp2, (sx2, sy2), (sx2+sw2, sy2+sh2), (0, 0, 255), 3)
        else:
            cv2.rectangle(disp1, (5, 5), (disp1.shape[1]-5, disp1.shape[0]-5), (0, 0, 255), 3)
            cv2.rectangle(disp2, (5, 5), (disp2.shape[1]-5, disp2.shape[0]-5), (0, 0, 255), 3)

        # 布局参数
        header_h = 80
        gap = 15
        footer_h = 70

        img_w1 = disp1.shape[1]
        img_w2 = disp2.shape[1]
        total_w = img_w1 + gap + img_w2
        total_h = header_h + target_h + footer_h

        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)

        canvas[header_h:header_h+target_h, :img_w1] = disp1
        canvas[header_h:header_h+target_h, img_w1+gap:] = disp2

        # 绘制顶部来源信息
        font = cv2.FONT_HERSHEY_SIMPLEX

        x1_base = 10
        group_color = (0, 200, 255) if src1["group"] != src2["group"] else (100, 255, 100)
        cv2.putText(canvas, src1["group"], (x1_base, 30), font, 1.0, group_color, 2)
        cv2.putText(canvas, src1["subfolder"], (x1_base, 55), font, 0.65, (200, 200, 200), 1)
        file_text = src1["filename"]
        if src1["channel"]:
            file_text += f"  [{src1['channel']}]"
        cv2.putText(canvas, file_text, (x1_base, 73), font, 0.5, (160, 160, 160), 1)

        x2_base = img_w1 + gap + 10
        cv2.putText(canvas, src2["group"], (x2_base, 30), font, 1.0, group_color, 2)
        cv2.putText(canvas, src2["subfolder"], (x2_base, 55), font, 0.65, (200, 200, 200), 1)
        file_text2 = src2["filename"]
        if src2["channel"]:
            file_text2 += f"  [{src2['channel']}]"
        cv2.putText(canvas, file_text2, (x2_base, 73), font, 0.5, (160, 160, 160), 1)

        mid_x = img_w1 + gap // 2
        cv2.putText(canvas, "VS", (mid_x - 15, 40), font, 0.7, (0, 0, 255), 2)

        # 分割线
        cv2.line(canvas, (0, header_h-2), (total_w, header_h-2), (80, 80, 80), 1)
        cv2.line(canvas, (0, header_h+target_h), (total_w, header_h+target_h), (80, 80, 80), 1)

        # 底部信息
        footer_y = header_h + target_h + 5
        sim_text = f"Similarity: {match.similarity*100:.1f}%"
        sim_color = (0, 0, 255) if match.severity == "critical" else (0, 140, 255) if match.severity == "high" else (0, 255, 255)
        cv2.putText(canvas, sim_text, (10, footer_y + 25), font, 0.7, sim_color, 2)
        type_text = f"Type: {match.match_type}"
        cv2.putText(canvas, type_text, (10, footer_y + 50), font, 0.5, (180, 180, 180), 1)

        sev_labels = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
        sev_colors = {"critical": (0, 0, 200), "high": (0, 100, 200), "medium": (0, 180, 180), "low": (0, 150, 0)}
        sev_label = sev_labels.get(match.severity, "UNKNOWN")
        sev_color = sev_colors.get(match.severity, (100, 100, 100))
        (tw, th), _ = cv2.getTextSize(sev_label, font, 0.55, 1)
        label_x = total_w - tw - 20
        cv2.rectangle(canvas, (label_x-5, footer_y+10), (label_x+tw+10, footer_y+10+th+10), sev_color, -1)
        cv2.putText(canvas, sev_label, (label_x, footer_y+10+th+5), font, 0.55, (255, 255, 255), 1)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        idx = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')]) + 1
        sev_tag = match.severity.upper()[:3]
        g1 = src1["group"] or "UNK"
        g2 = src2["group"] or "UNK"
        filename = f"{idx:03d}_{sev_tag}_{match.similarity*100:.0f}pct_{g1}_vs_{g2}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return output_path

    except Exception as e:
        return None


# ============================================================
# 报告生成
# ============================================================

def _generate_thumbnail(filepath: str, output_path: str, max_size: int = 300):
    """
    生成图片缩略图，用于 HTML 报告中预览。
    先用 PIL 尝试，失败后用 OpenCV 兜底，最大化兼容性。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 方法1: PIL
    try:
        img = Image.open(filepath)
        ratio = max_size / max(img.size)
        if ratio < 1:
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=70)
        return True
    except Exception:
        pass
    
    # 方法2: OpenCV 兜底（处理某些 PIL 打不开的 TIFF 变体）
    try:
        import cv2
        img = cv2.imread(filepath)
        if img is not None:
            h, w = img.shape[:2]
            ratio = max_size / max(h, w)
            if ratio < 1:
                new_w, new_h = int(w * ratio), int(h * ratio)
                img = cv2.resize(img, (new_w, new_h))
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return True
    except Exception:
        pass
    
    return False


def generate_html_report(matches: List[DuplicateMatch], output_path: str,
                         total_images: int, scan_dir: str, stats: PerformanceStats,
                         no_thumbnails: bool = False):
    """生成HTML报告（含图片预览）"""
    severity_colors = {
        "critical": "#e74c3c",
        "high": "#e67e22",
        "medium": "#f39c12",
        "low": "#27ae60"
    }
    severity_labels = {
        "critical": "严重",
        "high": "高",
        "medium": "中",
        "low": "低"
    }
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    n_critical = sum(1 for m in matches if m.severity == 'critical')
    n_high = sum(1 for m in matches if m.severity == 'high')
    n_medium = sum(1 for m in matches if m.severity == 'medium')
    n_low = sum(1 for m in matches if m.severity == 'low')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>科研图片查重报告 - {len(matches)} 对可疑</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, 'Microsoft YaHei', 'PingFang SC', sans-serif; background: #f0f2f5; color: #333; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

    /* 头部 */
    .header {{ background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
    .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .header p {{ opacity: 0.85; font-size: 14px; }}

    /* 统计卡片 */
    .stats-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; }}
    .stat-card .num {{ font-size: 32px; font-weight: 700; }}
    .stat-card .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
    .stat-card.critical .num {{ color: #e74c3c; }}
    .stat-card.high .num {{ color: #e67e22; }}
    .stat-card.medium .num {{ color: #f39c12; }}
    .stat-card.low .num {{ color: #27ae60; }}
    .stat-card.total .num {{ color: #1a73e8; }}

    /* 性能栏 */
    .perf-bar {{ background: #e8f5e9; padding: 16px 20px; border-radius: 10px; margin-bottom: 24px; font-size: 14px; color: #2e7d32; display: flex; flex-wrap: wrap; gap: 16px; }}
    .perf-bar span {{ white-space: nowrap; }}

    /* 筛选栏 */
    .filter-bar {{ background: white; padding: 16px 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 20px; display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }}
    .filter-bar label {{ font-size: 14px; color: #555; }}
    .filter-bar select, .filter-bar input {{ padding: 6px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }}
    .filter-btn {{ padding: 6px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; color: white; }}
    .filter-btn.active {{ background: #1a73e8; }}
    .filter-btn:not(.active) {{ background: #ccc; }}
    .filter-btn:hover {{ opacity: 0.85; }}
    .result-count {{ margin-left: auto; font-size: 14px; color: #888; }}

    /* 匹配卡片 */
    .match {{ background: white; margin-bottom: 16px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); overflow: hidden; transition: all .2s; }}
    .match:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.12); }}
    .match-header {{ padding: 16px 20px 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
    .match-title {{ font-weight: 600; font-size: 15px; }}
    .sev-badge {{ padding: 3px 12px; border-radius: 20px; color: white; font-size: 12px; font-weight: 600; }}
    .match-body {{ padding: 0 20px 16px; }}
    .match-meta {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: #666; margin-bottom: 10px; }}
    .match-meta strong {{ color: #333; }}
    .match-details {{ font-size: 13px; color: #888; margin-bottom: 12px; line-height: 1.6; word-break: break-all; }}
    .sim-bar {{ height: 6px; background: #eee; border-radius: 3px; margin-bottom: 12px; overflow: hidden; }}
    .sim-bar-fill {{ height: 100%; border-radius: 3px; transition: width .3s; }}

    /* 图片对比区 */
    .image-comparison {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .image-comparison .img-box {{ flex: 1; min-width: 200px; max-width: 48%; }}
    .image-comparison .img-box label {{ display: block; font-size: 11px; color: #999; margin-bottom: 4px; word-break: break-all; }}
    .image-comparison .img-box img {{ width: 100%; height: 200px; object-fit: contain; background: #fafafa; border: 1px solid #eee; border-radius: 6px; cursor: zoom-in; }}
    .image-comparison .img-box img:hover {{ border-color: #1a73e8; }}
    .vs-badge {{ display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 700; color: #e74c3c; min-width: 40px; }}

    /* 全屏预览 */
    .fullscreen-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 9999; justify-content: center; align-items: center; cursor: zoom-out; }}
    .fullscreen-overlay.show {{ display: flex; }}
    .fullscreen-overlay img {{ max-width: 95%; max-height: 95%; object-fit: contain; }}

    .footer {{ text-align: center; color: #aaa; margin-top: 40px; padding: 20px; font-size: 13px; }}
    @media (max-width: 600px) {{ .image-comparison {{ flex-direction: column; }} .image-comparison .img-box {{ max-width: 100%; }} }}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>🔬 科研图片查重报告</h1>
    <p>扫描目录: {scan_dir} &nbsp;|&nbsp; 扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; 共 {total_images} 张图片</p>
</div>

<div class="stats-row">
    <div class="stat-card total"><div class="num">{len(matches)}</div><div class="label">可疑重复对</div></div>
    <div class="stat-card critical"><div class="num">{n_critical}</div><div class="label">🔴 严重 (完全相同)</div></div>
    <div class="stat-card high"><div class="num">{n_high}</div><div class="label">🟠 高风险</div></div>
    <div class="stat-card medium"><div class="num">{n_medium}</div><div class="label">🟡 中等</div></div>
    <div class="stat-card low"><div class="num">{n_low}</div><div class="label">🟢 低风险</div></div>
</div>

<div class="perf-bar">
    <span>⚡ 哈希计算: {stats.hash_calculation_time:.1f}s</span>
    <span>🔍 比较检测: {stats.comparison_time:.1f}s</span>
    <span>📊 总比较: {stats.total_comparisons:,}</span>
    <span>📈 优化: {stats.optimization_ratio:.1f}x</span>
</div>

<div class="filter-bar">
    <label>🔍 筛选:</label>
    <select id="sevFilter" onchange="applyFilter()">
        <option value="all">全部 ({len(matches)})</option>
        <option value="critical">🔴 严重 ({n_critical})</option>
        <option value="high">🟠 高风险 ({n_high})</option>
        <option value="medium">🟡 中等 ({n_medium})</option>
        <option value="low">🟢 低风险 ({n_low})</option>
    </select>
    <label>投票通过数:</label>
    <select id="voteFilter" onchange="applyFilter()">
        <option value="0">全部</option>
        <option value="4">4 种</option>
        <option value="3">≥ 3 种</option>
        <option value="2">≥ 2 种</option>
    </select>
    <label>最低置信度:</label>
    <input type="range" id="confFilter" min="0" max="100" value="0" oninput="updateConfLabel();applyFilter()">
    <span id="confLabel">0%</span>
    <label>搜索:</label>
    <input type="text" id="searchInput" placeholder="文件名/类型/详情..." oninput="applyFilter()" style="min-width:180px;">
    <span class="result-count" id="resultCount">显示 {len(matches)} 对</span>
</div>

<div style="display:flex;justify-content:center;gap:12px;align-items:center;margin-bottom:16px;">
    <button id="prevBtn" class="filter-btn" onclick="changePage(-1)" disabled>← 上一页</button>
    <span id="pageInfo">第 1 页</span>
    <button id="nextBtn" class="filter-btn" onclick="changePage(1)" disabled>下一页 →</button>
    <select id="pageSizeSel" onchange="applyFilter()" style="padding:6px 12px;border:1px solid #ddd;border-radius:6px;">
        <option value="50" selected>50 条/页</option>
        <option value="100">100 条/页</option>
        <option value="200">200 条/页</option>
        <option value="0">全部</option>
    </select>
</div>

<div id="matchList">
"""

    # ----- 为所有匹配生成缩略图 -----
    # 改进1：按图片路径缓存——同一张图出现在多对里时只生成一次
    # 改进2：并行生成（旧版串行，且每对重复生成2张）
    report_dir = os.path.dirname(os.path.abspath(output_path))
    thumb_dir = os.path.join(report_dir, '_thumbnails')
    if os.path.isdir(thumb_dir):
        for old in os.listdir(thumb_dir):
            try: os.remove(os.path.join(thumb_dir, old))
            except: pass
    os.makedirs(thumb_dir, exist_ok=True)

    # 统计每张图出现的次数（用于badge提示审查优先级）
    appear_count = defaultdict(int)
    for m in matches:
        appear_count[m.image1] += 1
        appear_count[m.image2] += 1

    thumb_map: Dict[str, Optional[str]] = {}
    if not no_thumbnails:
        unique_paths = sorted(appear_count.keys())
        print(f"  [·] 生成 {len(unique_paths)} 张唯一缩略预览图（并行）...")
        n_thumb_workers = min(8, max(1, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=n_thumb_workers) as ex:
            future_to_path = {}
            for p in unique_paths:
                thumb_name = hashlib.md5(p.encode('utf-8')).hexdigest()[:12] + '.jpg'
                future = ex.submit(_generate_thumbnail, p, os.path.join(thumb_dir, thumb_name))
                future_to_path[future] = (p, thumb_name)
            thumb_ok = 0
            thumb_fail = 0
            for future in as_completed(future_to_path):
                p, thumb_name = future_to_path[future]
                if future.result():
                    thumb_map[p] = os.path.join('_thumbnails', thumb_name).replace('\\', '/')
                    thumb_ok += 1
                else:
                    thumb_map[p] = None
                    thumb_fail += 1
        print(f"      缩略图: {thumb_ok} 成功, {thumb_fail} 失败")
    else:
        print("  [·] 已跳过缩略图生成 (--no-thumbnails)")

    for i, m in enumerate(matches, 1):
        color = severity_colors.get(m.severity, "#999")
        label = severity_labels.get(m.severity, "未知")
        sim_bar_width = max(1, int(m.similarity * 100))
        sev_order = severity_order.get(m.severity, 9)
        vote_count = len(m.vote_details.split("+")) if m.vote_details else 0
        if "MD5" in m.vote_details:
            vote_count = 4

        fname1 = os.path.basename(m.image1)
        fname2 = os.path.basename(m.image2)
        cnt1 = appear_count.get(m.image1, 0)
        cnt2 = appear_count.get(m.image2, 0)

        rel_thumb1 = thumb_map.get(m.image1)
        rel_thumb2 = thumb_map.get(m.image2)

        conf_pct = int(m.confidence * 100) if m.confidence else 0
        search_text = f"{fname1} {fname2} {m.match_type} {m.details}".lower()

        def _thumb_html(rel, fname, cnt):
            badge = f' <span style="color:#1a73e8">(出现{cnt}次)</span>' if cnt and cnt > 1 else ''
            if rel:
                return f'<label>📷 {fname}{badge}</label><img src="{rel}" alt="{fname}" loading="lazy" onclick="showFullscreen(this)">'
            return f'<label>📷 {fname}{badge}</label><div style="padding:60px 10px;text-align:center;color:#bbb;border:1px dashed #ddd;border-radius:6px;font-size:12px;">{"⚠️ 无预览 (--no-thumbnails)" if no_thumbnails else "⚠️ 无法生成预览<br><span style=\"font-size:11px;\">格式不支持或文件损坏</span>"}</div>'

        html += f"""
<div class="match {m.severity}" data-severity="{m.severity}" data-votes="{vote_count}" data-conf="{conf_pct}" data-search="{search_text}">
    <div class="match-header">
        <span class="match-title">#{i} {m.match_type}</span>
        <span class="sev-badge" style="background:{color}">{label}</span>
    </div>
    <div class="match-body">
        <div class="match-meta">
            <span>相似度: <strong>{m.similarity*100:.1f}%</strong></span>
            <span>置信度: <strong>{conf_pct}%</strong></span>
            <span>投票: <strong>{m.vote_details}</strong></span>
        </div>
        <div class="match-details">{m.details}</div>
        <div class="sim-bar"><div class="sim-bar-fill" style="width:{sim_bar_width}%;background:{color}"></div></div>
        <div class="image-comparison">
            <div class="img-box">
                {_thumb_html(rel_thumb1, fname1, cnt1)}
            </div>
            <div class="vs-badge">VS</div>
            <div class="img-box">
                {_thumb_html(rel_thumb2, fname2, cnt2)}
            </div>
        </div>
    </div>
</div>
"""

    html += f"""
</div>

<div class="footer">
    <p>由 科研图片查重工具(优化版) 生成 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>优化技术参考: <a href="https://github.com/ayumilove/xImageDuplicateChecker">xImageDuplicateChecker</a> (MIT License) · 
    <a href="https://github.com/idealo/imagededup">imagededup</a></p>
</div>

</div>

<!-- 全屏预览 -->
<div class="fullscreen-overlay" id="fullscreenOverlay" onclick="this.classList.remove('show')">
    <img id="fullscreenImg" src="" alt="preview">
</div>

<script>
var shownItems = [];
var currentPage = 1;

function applyFilter() {{
    var sev = document.getElementById('sevFilter').value;
    var minVotes = parseInt(document.getElementById('voteFilter').value);
    var minConf = parseInt(document.getElementById('confFilter').value);
    var q = document.getElementById('searchInput').value.toLowerCase().trim();
    var items = document.querySelectorAll('.match');
    shownItems = [];
    for (var i = 0; i < items.length; i++) {{
        var item = items[i];
        var s = item.getAttribute('data-severity');
        var v = parseInt(item.getAttribute('data-votes'));
        var c = parseInt(item.getAttribute('data-conf'));
        var show = (sev === 'all' || s === sev) && v >= minVotes && c >= minConf;
        if (show && q && item.getAttribute('data-search').indexOf(q) < 0) show = false;
        item.style.display = 'none';
        if (show) shownItems.push(item);
    }}
    currentPage = 1;
    renderPage();
}}

function getPageSize() {{
    var v = parseInt(document.getElementById('pageSizeSel').value);
    return v > 0 ? v : shownItems.length;
}}

function renderPage() {{
    var total = shownItems.length;
    var pageSize = getPageSize();
    var pages = Math.max(1, Math.ceil(total / pageSize));
    if (currentPage > pages) currentPage = pages;
    if (currentPage < 1) currentPage = 1;
    var start = (currentPage - 1) * pageSize;
    var end = Math.min(start + pageSize, total);
    for (var i = 0; i < total; i++) {{
        shownItems[i].style.display = (i >= start && i < end) ? '' : 'none';
    }}
    var shownOnPage = Math.max(0, end - start);
    document.getElementById('resultCount').textContent = total + ' 对' + (pageSize < total ? ' (本页 ' + shownOnPage + ')' : '');
    document.getElementById('pageInfo').textContent = pages <= 1 ? '共 ' + total + ' 条' : '第 ' + currentPage + ' / ' + pages + ' 页';
    document.getElementById('prevBtn').disabled = currentPage <= 1 || pages <= 1;
    document.getElementById('nextBtn').disabled = currentPage >= pages || pages <= 1;
}}

function changePage(delta) {{
    currentPage += delta;
    renderPage();
    window.scrollTo({{top: 0, behavior: 'smooth'}});
}}

function updateConfLabel() {{
    var val = document.getElementById('confFilter').value;
    document.getElementById('confLabel').textContent = val + '%';
}}
function showFullscreen(el) {{
    var overlay = document.getElementById('fullscreenOverlay');
    var img = document.getElementById('fullscreenImg');
    img.src = el.src;  // 用缩略图全屏预览（避开 file:// 限制）
    overlay.classList.add('show');
}}
document.addEventListener('keydown', function(e) {{
    var overlay = document.getElementById('fullscreenOverlay');
    if (overlay.classList.contains('show') && e.key === 'Escape') overlay.classList.remove('show');
}});
</script>

</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  HTML报告已生成: {output_path}")
    print(f"      缩略图: {thumb_ok} 成功, {thumb_fail} 失败（共 {len(matches)*2} 张）")


def generate_csv_report(matches: List[DuplicateMatch], output_path: str):
    """生成CSV报告"""
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['序号', '问题类型', '严重程度', '相似度', '置信度', '投票详情', '图片1', '图片2', '详细信息'])
        for i, m in enumerate(matches, 1):
            writer.writerow([
                i, m.match_type, m.severity,
                f"{m.similarity*100:.1f}%",
                f"{m.confidence*100:.1f}%",
                m.vote_details,
                m.image1, m.image2, m.details
            ])
    print(f"  CSV报告已生成: {output_path}")


def generate_json_report(matches: List[DuplicateMatch], output_path: str):
    """生成JSON报告（机器可读，便于二次处理/CI集成）"""
    data = {
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_matches": len(matches),
        "matches": [
            {
                "index": i,
                "match_type": m.match_type,
                "severity": m.severity,
                "similarity": m.similarity,
                "confidence": m.confidence,
                "vote_details": m.vote_details,
                "image1": m.image1,
                "image2": m.image2,
                "details": m.details,
            }
            for i, m in enumerate(matches, 1)
        ],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  JSON报告已生成: {output_path}")


# ============================================================
# 主扫描流程（优化版）
# ============================================================

def _dhash_quick_check(info1: Dict, info2: Dict, threshold: int = 8) -> bool:
    """极速 dHash 预筛（仅 XOR + 位计数，~0.001ms），用于快速过滤明显不相似的候选对"""
    diff = info1['dhash_int'] ^ info2['dhash_int']
    # 快速位计数（Brian Kernighan 算法）
    count = 0
    while diff:
        diff &= diff - 1
        count += 1
    return count <= threshold


def _hash_quick_check(info1: Dict, info2: Dict,
                      dthresh: int = 8, pthresh: int = 12) -> bool:
    """
    极速哈希预筛（~0.002ms）：dHash 或 pHash 任一接近即放行。

    旧版只用 dHash：dHash 对亮度/对比度偏移天生不鲁棒（实测同图亮度调整
    dHash 差异可达 20/256），会把"亮度/对比度调整"类副本在预筛阶段误杀。
    pHash 对光度变化鲁棒（DCT 低频），与 dHash 互补，二者取或可保召回；
    精确性由后续完整投票 + ORB 确认兜底。
    """
    # dHash 快速检查
    diff = info1['dhash_int'] ^ info2['dhash_int']
    count = 0
    while diff:
        diff &= diff - 1
        count += 1
    if count <= dthresh:
        return True
    # pHash 快速检查
    diff = info1['phash_int'] ^ info2['phash_int']
    count = 0
    while diff:
        diff &= diff - 1
        count += 1
    return count <= pthresh


_ROT_TRANSFORMS = [
    ("旋转90°", lambda im: im.rotate(90, expand=False)),
    ("旋转180°", lambda im: im.rotate(180, expand=False)),
    ("旋转270°", lambda im: im.rotate(270, expand=False)),
    ("水平翻转", lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)),
    ("垂直翻转", lambda im: im.transpose(Image.FLIP_TOP_BOTTOM)),
]


def _check_subimage_fast(info_big: Dict, info_small: Dict) -> Optional[DuplicateMatch]:
    """
    快速子图/裁剪检测：小图是否出现在大图中。

    原版的滑动窗口模板匹配是 O(w*h*scale) 逐像素 NCC，慢到不可用。
    这里把大图降采样到 ~160px，模板按**大图的同一降采样因子**缩放
    （保持相对尺度——若按小图自身最大边降采样，模板会比实际裁剪区域
    大出数倍，永远匹配不上），再做多尺度归一化互相关，单对毫秒级。
    """
    try:
        import cv2
    except ImportError:
        return None

    try:
        from PIL import Image as _PILImage
        big_img = _PILImage.open(info_big['path']).convert('L')
        small_img = _PILImage.open(info_small['path']).convert('L')

        fb = SUBIMAGE_MAX_DIM / max(big_img.size)  # 大图降采样因子
        bw = max(1, int(big_img.size[0] * fb))
        bh = max(1, int(big_img.size[1] * fb))
        big = np.array(big_img.resize((bw, bh), _PILImage.LANCZOS))

        # 模板基准尺寸 = 小图按同一因子缩放（即小图在大图坐标系中的尺寸）
        tpl_w = int(small_img.size[0] * fb)
        tpl_h = int(small_img.size[1] * fb)
        if tpl_w < 12 or tpl_h < 12:
            return None
        if tpl_w >= bw or tpl_h >= bh:
            return None  # 小图不比大图小，不可能是裁剪关系

        small_arr = np.array(small_img)
        # 模板最大边长限制（防止小图过大时模板计算慢）
        max_tpl = 200
        if max(tpl_w, tpl_h) > max_tpl:
            r = max_tpl / max(tpl_w, tpl_h)
            tpl_w, tpl_h = int(tpl_w * r), int(tpl_h * r)

        best_score = -1.0
        best_scale = 1.0
        for scale in SUBIMAGE_SCALES:
            new_w, new_h = int(tpl_w * scale), int(tpl_h * scale)
            if new_w < 12 or new_h < 12 or new_w >= bw or new_h >= bh:
                continue
            tpl = cv2.resize(small_arr, (new_w, new_h))
            res = cv2.matchTemplate(big, tpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best_score = score
                best_scale = scale
            if best_score >= SUBIMAGE_NCC_THRESHOLD:
                break

        if best_score >= SUBIMAGE_NCC_THRESHOLD:
            area_ratio = (info_big['size'][0] * info_big['size'][1]) / \
                         max(1, info_small['size'][0] * info_small['size'][1])
            return DuplicateMatch(
                image1=info_small['path'],
                image2=info_big['path'],
                match_type="疑似子图/裁剪",
                similarity=round(float(best_score), 4),
                details=f"小图出现在大图中, NCC: {best_score:.4f}, 缩放: {best_scale}, 面积比: {area_ratio:.1f}x",
                severity="critical",
                confidence=round(float(best_score) * 0.9, 4),
                vote_details="子图"
            )
    except Exception:
        pass
    return None


def _build_subimage_pairs(images: List[Dict],
                          exact_pairs: Set[Tuple[str, str]],
                          min_std: float,
                          max_pairs: int = SUBIMAGE_MAX_PAIRS) -> List[Tuple[Dict, Dict]]:
    """
    按面积比筛选潜在的"大图-小图"裁剪候选对（纯算术过滤，无IO）。
    按面积升序排序后二分定位，避免 O(n²) 全量比较。
    """
    exact_set = {tuple(sorted(p)) for p in exact_pairs}
    area_sorted = sorted(images, key=lambda im: im['size'][0] * im['size'][1])
    areas = [im['size'][0] * im['size'][1] for im in area_sorted]

    pairs: List[Tuple[Dict, Dict]] = []
    n = len(area_sorted)
    for i in range(n):
        a_i = areas[i]
        if a_i == 0:
            continue
        # 二分找到第一个面积 ≥ 1.7*area_i 的 j
        lo, hi = i + 1, n
        target = a_i * SUBIMAGE_MIN_AREA_RATIO
        while lo < hi:
            mid = (lo + hi) // 2
            if areas[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        for j in range(lo, n):
            im_i, im_j = area_sorted[i], area_sorted[j]
            if im_i['path'] == im_j['path']:
                continue
            if min(im_i.get('gray_std', 255.0), im_j.get('gray_std', 255.0)) < min_std:
                continue
            if _is_same_experiment_channel(im_i['path'], im_j['path']):
                continue
            pair = tuple(sorted([im_i['path'], im_j['path']]))
            if pair in exact_set:
                continue
            # 小图为 im_i（面积小），大图为 im_j
            pairs.append((im_j, im_i))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _build_rotation_candidates(images: List[Dict], hash_size: int,
                               exact_pairs: Set[Tuple[str, str]],
                               max_pairs: int = ROT_MAX_PAIRS) -> List[Tuple[str, str]]:
    """
    构建旋转/翻转增强索引并返回候选对。

    对每张图缓存的等比缩放画布（rot_array，256x256，无需重新读盘）
    施加 5 种几何变换，把变换后的 dHash 高位前缀放入宽松桶；
    同桶的不同图片即旋转候选。
    """
    exact_set = {tuple(sorted(p)) for p in exact_pairs}
    mask = (1 << ROT_AUG_PREFIX_BITS) - 1
    buckets: Dict[int, List[str]] = defaultdict(list)

    for img in images:
        try:
            im = Image.fromarray(img['rot_array'])
        except Exception:
            continue
        for _name, tf in _ROT_TRANSFORMS:
            try:
                h = imagehash.dhash(tf(im), hash_size=hash_size)
                nbits = h.hash.size
                prefix = (int(str(h), 16) >> max(0, nbits - ROT_AUG_PREFIX_BITS)) & mask
                buckets[prefix].append(img['path'])
            except Exception:
                continue

    seen = set()
    candidates: List[Tuple[str, str]] = []
    for paths in buckets.values():
        if len(paths) < 2:
            continue
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                p1, p2 = paths[i], paths[j]
                if p1 == p2:
                    continue
                pair = (min(p1, p2), max(p1, p2))
                if pair in seen:
                    continue
                seen.add(pair)
                if pair in exact_set:
                    continue
                candidates.append(pair)
                if len(candidates) >= max_pairs:
                    return candidates
    return candidates


def find_duplicates_optimized(images: List[Dict],
                              stats: PerformanceStats,
                              pthresh: int = DEFAULT_PHASH_THRESHOLD,
                              sthresh: float = DEFAULT_SSIM_THRESHOLD,
                              hthresh: float = DEFAULT_HIST_THRESHOLD,
                              check_rotations: bool = True,
                              use_lsh: bool = True,
                              min_votes: int = VOTE_MIN_AGREE,
                              orb_confirm: bool = True,
                              orb_strict: bool = False,
                              min_std: float = LOW_INFO_STD,
                              hash_size: int = DEFAULT_HASH_SIZE,
                              check_subimage: bool = True) -> List[DuplicateMatch]:
    """
    执行所有检测（两级过滤 + 并行投票制）
    
    核心改进：
    1. 两级过滤：dHash 极速预筛 → 通过者再做完整投票
    2. 投票制：至少 min_votes 个结构类检测器一致才报告
    3. 尺寸/颜色/低信息量预过滤 + 通道过滤
    4. 旋转检测仅在投票通过后执行
    5. ORB 二次确认边缘候选对（降低误报）
    """
    matches = []
    n = len(images)

    print(f"\n[3/4] 执行图片查重检测 (共 {n} 张)...")

    # ---- 阶段1: MD5精确去重 ----
    md5_groups = defaultdict(list)
    for img in images:
        md5_groups[img['md5']].append(img)

    exact_pairs = set()
    for md5, group in md5_groups.items():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    m = check_exact_duplicate(group[i], group[j])
                    if m:
                        m.confidence = 1.0
                        m.vote_details = "MD5"
                        matches.append(m)
                        exact_pairs.add((group[i]['path'], group[j]['path']))

    print(f"  [✓] 阶段1 完全相同: {len(matches)} 对")
    print(f"  [✓] 阶段1 排除 {len(exact_pairs)} 对（已标记为完全相同）")

    # ---- 阶段2: 构建索引 ----
    print(f"  [·] 阶段2 构建索引...")
    index_start = time.time()
    
    if use_lsh:
        index = LSHIndex()
        index.build(images)
        candidate_pairs = index.get_candidate_pairs()
        index_type = "LSH"
    else:
        index = PrefixIndex()
        index.build(images)
        candidate_pairs = index.get_candidate_pairs()
        index_type = "前缀"
    
    stats.update(index_building_time=time.time() - index_start)
    
    naive_comparisons = n * (n - 1) // 2
    optimization_ratio = naive_comparisons / len(candidate_pairs) if candidate_pairs else 1
    stats.update(candidate_pairs=len(candidate_pairs), optimization_ratio=optimization_ratio)
    
    print(f"  [✓] 阶段2 {index_type}索引构建完成: {len(candidate_pairs)} 候选对 (优化 {optimization_ratio:.1f}x)")

    # 过滤掉完全相同、同实验通道的对
    path_to_info = {img['path']: img for img in images}
    valid_pairs = [
        (p1, p2) for p1, p2 in candidate_pairs
        if p1 in path_to_info and p2 in path_to_info
        and (p1, p2) not in exact_pairs and (p2, p1) not in exact_pairs
        and not _is_same_experiment_channel(p1, p2)
    ]
    n_total = len(valid_pairs)

    # ================================================================
    # 阶段3a: 哈希极速预筛（全部候选对，极快）
    # ================================================================
    print(f"\n  [·] 阶段3a 哈希极速预筛 {n_total} 个候选对 (dHash≤8 或 pHash≤12)...")
    t0 = time.time()
    # 分批并行处理预筛
    import os as _os
    _n_workers = min(8, max(1, (_os.cpu_count() or 4) // 2))
    dhash_batch_size = 50000
    prequal = []
    prequal_lock = threading.Lock()
    
    def _dhash_batch(pairs_batch):
        return [(p1, p2) for p1, p2 in pairs_batch
                if _hash_quick_check(path_to_info[p1], path_to_info[p2])]
    
    with ThreadPoolExecutor(max_workers=_n_workers) as ex:
        futures = []
        for bstart in range(0, n_total, dhash_batch_size):
            batch = valid_pairs[bstart:bstart + dhash_batch_size]
            futures.append(ex.submit(_dhash_batch, batch))
        for f in as_completed(futures):
            prequal.extend(f.result())
    
    dhash_time = time.time() - t0
    print(f"  [✓] 阶段3a 完成: {n_total}→{len(prequal)} 对通过预筛 ({dhash_time:.1f}s)")
    print(f"      过滤掉 {(n_total-len(prequal)):,} 对 ({max(0, (n_total-len(prequal))/max(n_total,1)*100):.1f}%)")

    if not prequal:
        print(f"\n  [✓] 无候选对需要通过完整检测")
        stats.update(comparison_time=0, total_comparisons=n_total)
        return matches

    # ================================================================
    # 阶段3b: 完整投票制检测（仅对预筛通过的对）
    # ================================================================
    print(f"\n  [·] 阶段3b 完整投票制检测 {len(prequal)} 个候选对...")
    print(f"      要求至少 {min_votes} 种检测器一致通过")
    print(f"      使用 {_n_workers} 个并行线程")
    t1 = time.time()
    
    vote_stats = defaultdict(int)
    matches_lock = threading.Lock()
    
    def _full_compare(pair):
        p1, p2 = pair
        m = check_pair_voting(
            path_to_info[p1], path_to_info[p2],
            pthresh=pthresh, sthresh=sthresh, hthresh=hthresh,
            min_votes=min_votes, check_rotations=check_rotations,
            orb_confirm=orb_confirm, orb_strict=orb_strict,
            min_std=min_std
        )
        return m
    
    batch_size = 2000
    processed = 0
    with ThreadPoolExecutor(max_workers=_n_workers) as executor:
        for bstart in range(0, len(prequal), batch_size):
            batch = prequal[bstart:bstart + batch_size]
            for f in as_completed([executor.submit(_full_compare, p) for p in batch]):
                m = f.result()
                if m:
                    with matches_lock:
                        matches.append(m)
                        for name in m.vote_details.split("+"):
                            vote_stats[name] += 1
            processed += len(batch)
            if processed % 5000 == 0 or processed == len(prequal):
                print(f"    进度: {processed}/{len(prequal)} ({processed/max(len(prequal),1)*100:.1f}%)")
    
    comparison_time = time.time() - t1
    stats.update(comparison_time=comparison_time, total_comparisons=n_total)
    
    if vote_stats:
        print(f"      投票统计: {dict(vote_stats)}")
    print(f"  [✓] 阶段3b 完成: {processed} 对比较, 耗时 {comparison_time:.1f}s")
    print(f"      发现 {len(matches)} 对可疑重复")
    print(f"      总有效加速: 从 {n_total:,} 预筛到 {len(prequal):,}, 完整检测仅 {len(prequal):,} 对")

    # ================================================================
    # 阶段3c: 旋转/翻转增强召回
    # （旋转副本哈希差异大，进不了普通索引，需专门的变换增强索引）
    # ================================================================
    if check_rotations:
        print(f"\n  [·] 阶段3c 旋转/翻转增强召回...")
        t2 = time.time()
        rot_candidates = _build_rotation_candidates(images, hash_size, exact_pairs)
        prequal_set = {tuple(sorted(p)) for p in prequal}
        rot_pairs = []
        skipped_lowinfo = 0
        for p in rot_candidates:
            if p in prequal_set:
                continue  # 已进入完整投票，其旋转检测已在 check_pair_voting 内完成
            info1, info2 = path_to_info[p[0]], path_to_info[p[1]]
            if min(info1.get('gray_std', 255.0), info2.get('gray_std', 255.0)) < min_std:
                skipped_lowinfo += 1
                continue
            rot_pairs.append(p)
        print(f"      增强索引候选 {len(rot_candidates)} 对, 需旋转校验 {len(rot_pairs)} 对"
              + (f" (跳过低信息 {skipped_lowinfo} 对)" if skipped_lowinfo else ""))

        rot_found = 0
        if rot_pairs:
            def _rot_check(pair):
                return check_rotation_flip(path_to_info[pair[0]], path_to_info[pair[1]])

            with ThreadPoolExecutor(max_workers=_n_workers) as ex:
                for f in as_completed([ex.submit(_rot_check, p) for p in rot_pairs]):
                    for m in f.result():
                        with matches_lock:
                            matches.append(m)
                            rot_found += 1
        rot_time = time.time() - t2
        print(f"  [✓] 阶段3c 完成: 旋转/翻转匹配 {rot_found} 对 ({rot_time:.1f}s)")

    # ================================================================
    # 阶段3d: 子图/裁剪检测（条带复用类造假）
    # ================================================================
    if check_subimage:
        print(f"\n  [·] 阶段3d 子图/裁剪检测...")
        t3 = time.time()
        # 跳过已被投票/旋转阶段报告过的对，避免同一对重复出现
        reported = set()
        for m in matches:
            reported.add(tuple(sorted([m.image1, m.image2])))
        sub_pairs = _build_subimage_pairs(images, exact_pairs, min_std)
        sub_pairs = [p for p in sub_pairs if tuple(sorted([p[0]['path'], p[1]['path']])) not in reported]
        print(f"      面积比候选 {len(sub_pairs)} 对 (阈值: {SUBIMAGE_MIN_AREA_RATIO}x, 上限: {SUBIMAGE_MAX_PAIRS})")

        sub_found = 0
        if sub_pairs:
            def _sub_check(pair):
                return _check_subimage_fast(pair[0], pair[1])

            with ThreadPoolExecutor(max_workers=_n_workers) as ex:
                for f in as_completed([ex.submit(_sub_check, p) for p in sub_pairs]):
                    m = f.result()
                    if m:
                        with matches_lock:
                            matches.append(m)
                            sub_found += 1
        sub_time = time.time() - t3
        print(f"  [✓] 阶段3d 完成: 子图/裁剪匹配 {sub_found} 对 ({sub_time:.1f}s)")

    return matches


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="科研图片查重工具(优化版) - 借鉴xImageDuplicateChecker的优化技术",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python image_dedup_optimized.py D:\\proj_test\\proj_1
  python image_dedup_optimized.py D:\\proj_test\\proj_1 --workers 8
  python image_dedup_optimized.py D:\\proj_test\\proj_1 --min-votes 3   # 更严格
  python image_dedup_optimized.py D:\\proj_test\\proj_1 --no-lsh
  python image_dedup_optimized.py D:\\proj_test\\proj_1 --min-confidence 0.7
  python image_dedup_optimized.py D:\\proj_test\\proj_1 --orb-strict    # ORB未确认直接剔除
  python image_dedup_optimized.py D:\\proj_test\\proj_1 --no-thumbnails --report report.json
  
优化技术参考: https://github.com/ayumilove/xImageDuplicateChecker (MIT License)
投票制设计参考: https://github.com/idealo/imagededup
        """
    )

    parser.add_argument("directory", help="要扫描的图片目录")
    parser.add_argument("--hash-size", type=int, default=DEFAULT_HASH_SIZE,
                        help=f"哈希大小 (默认: {DEFAULT_HASH_SIZE})")
    parser.add_argument("--threshold", type=int, default=DEFAULT_PHASH_THRESHOLD,
                        help=f"哈希差异阈值，越小越严格 (默认: {DEFAULT_PHASH_THRESHOLD})")
    parser.add_argument("--ssim-threshold", type=float, default=DEFAULT_SSIM_THRESHOLD,
                        help=f"SSIM阈值 (默认: {DEFAULT_SSIM_THRESHOLD})")
    parser.add_argument("--hist-threshold", type=float, default=DEFAULT_HIST_THRESHOLD,
                        help=f"直方图阈值 (默认: {DEFAULT_HIST_THRESHOLD})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"并行工作线程数 (默认: {DEFAULT_WORKERS})")
    parser.add_argument("--report", type=str, default=None,
                        help="输出报告文件路径 (支持 .html / .csv / .json)")
    parser.add_argument("--no-rotation", action="store_true",
                        help="跳过旋转/翻转检测")
    parser.add_argument("--min-votes", type=int, default=VOTE_MIN_AGREE,
                        help=f"投票制最少通过的结构类检测器数(pHash/dHash/SSIM) (默认: {VOTE_MIN_AGREE})")
    parser.add_argument("--no-lsh", action="store_true",
                        help="使用简单前缀索引代替LSH索引")
    parser.add_argument("--min-std", type=float, default=LOW_INFO_STD,
                        help=f"灰度标准差阈值：低于此值的空白/纯色图不参与感知匹配 (默认: {LOW_INFO_STD})")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="只保留置信度≥此值的匹配（0~1，默认0=全部保留）")
    parser.add_argument("--no-thumbnails", action="store_true",
                        help="跳过HTML报告缩略图生成（大报告加速）")
    parser.add_argument("--no-orb-confirm", action="store_true",
                        help="关闭ORB二次确认（边缘候选对不再用特征匹配复核）")
    parser.add_argument("--orb-strict", action="store_true",
                        help="ORB确认失败直接剔除该候选对（默认仅降级为低风险）")
    parser.add_argument("--no-subimage", action="store_true",
                        help="关闭子图/裁剪检测（面积比过滤+模板匹配）")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"错误: 目录不存在: {args.directory}")
        sys.exit(1)

    print("=" * 60)
    print("  🔬 科研图片查重工具 (优化版)")
    print("  优化技术参考: xImageDuplicateChecker (MIT License)")
    print("=" * 60)
    print(f"  扫描目录: {args.directory}")
    print(f"  哈希阈值: {args.threshold}")
    print(f"  SSIM阈值: {args.ssim_threshold}")
    print(f"  直方图阈值: {args.hist_threshold}")
    print(f"  并行线程: {args.workers}")
    print(f"  投票制: 最少通过 {args.min_votes} 种结构类检测器")
    print(f"  索引类型: {'简单前缀' if args.no_lsh else 'LSH'}")
    print(f"  低信息过滤: 标准差 < {args.min_std} 的图跳过感知匹配")
    print(f"  ORB二次确认: {'关闭' if args.no_orb_confirm else ('严格剔除' if args.orb_strict else '降级模式')}")

    # 扫描
    images, stats = scan_directory_parallel(args.directory, args.hash_size, args.workers)

    if len(images) < 2:
        print("\n图片数量不足，无法进行查重。")
        sys.exit(0)

    # 检测
    matches = find_duplicates_optimized(
        images,
        stats,
        pthresh=args.threshold,
        sthresh=args.ssim_threshold,
        hthresh=args.hist_threshold,
        check_rotations=not args.no_rotation,
        use_lsh=not args.no_lsh,
        min_votes=args.min_votes,
        orb_confirm=not args.no_orb_confirm,
        orb_strict=args.orb_strict,
        min_std=args.min_std,
        hash_size=args.hash_size,
        check_subimage=not args.no_subimage,
    )

    # 去重排序
    matches = deduplicate_matches(matches)

    # 按置信度过滤
    if args.min_confidence > 0:
        before = len(matches)
        matches = [m for m in matches if m.confidence >= args.min_confidence]
        if len(matches) != before:
            print(f"  [·] 置信度过滤: {before} → {len(matches)} 对 (≥{args.min_confidence:.2f})")

    # 生成可视化对比图（只对非MD5的前200对生成，避免OpenCV大量警告）
    vis_dir = os.path.join(args.directory, "visualization")
    non_md5 = [m for m in matches if 'MD5' not in m.vote_details]
    if non_md5:
        print(f"\n[4/4] 生成可视化对比图（仅对非MD5匹配的前{min(200, len(non_md5))}对）...")
        # 抑制 OpenCV 警告
        import warnings as _w
        _w.filterwarnings('ignore', category=UserWarning, module='cv2')
        try:
            import os as _os
            _os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
        except:
            pass
        vis_count = 0
        for i, m in enumerate(non_md5[:200]):
            vis_path = create_visualization(m, vis_dir)
            if vis_path:
                m.details += f" [可视化: {os.path.basename(vis_path)}]"
                vis_count += 1
        print(f"  [✓] 生成 {vis_count} 张可视化对比图: {vis_dir}")

    # 输出结果
    print("\n" + "=" * 60)
    print(f"  📊 检测结果: 发现 {len(matches)} 对可疑重复")
    print("=" * 60)

    if not matches:
        print("\n  ✅ 未发现可疑重复图片！")
    else:
        by_severity = defaultdict(list)
        for m in matches:
            by_severity[m.severity].append(m)

        severity_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        severity_names = {"critical": "严重", "high": "高风险", "medium": "中等", "low": "低风险"}

        for sev in ["critical", "high", "medium", "low"]:
            if sev in by_severity:
                print(f"\n  {severity_icons[sev]} {severity_names[sev]} ({len(by_severity[sev])} 对):")
                for m in by_severity[sev][:10]:
                    print(f"    • {m.match_type} (相似度: {m.similarity*100:.1f}%, 置信度: {m.confidence*100:.1f}%)")
                    print(f"      投票: {m.vote_details}")
                    print(f"      {os.path.basename(m.image1)} ↔ {os.path.basename(m.image2)}")
                    print(f"      {m.details}")
                if len(by_severity[sev]) > 10:
                    print(f"    ... 还有 {len(by_severity[sev]) - 10} 对")

    # 生成报告
    if args.report:
        report_path = args.report
    else:
        report_path = os.path.join(args.directory, f"image_dedup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if report_path.endswith('.csv'):
        generate_csv_report(matches, report_path)
    elif report_path.endswith('.json'):
        generate_json_report(matches, report_path)
    else:
        if not report_path.endswith('.html'):
            report_path += '.html'
        generate_html_report(matches, report_path, len(images), args.directory, stats,
                             no_thumbnails=args.no_thumbnails)

    csv_path = report_path.rsplit('.', 1)[0] + '.csv'
    if csv_path != report_path:
        generate_csv_report(matches, csv_path)

    # 打印性能统计
    print(stats.get_summary())
    
    print(f"\n✅ 扫描完成！")


if __name__ == "__main__":
    main()
