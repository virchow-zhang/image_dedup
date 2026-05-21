#!/usr/bin/env python3
"""
科研图片查重工具 - 优化版
========================
借鉴 xImageDuplicateChecker (MIT License) 的优化思路

主要优化：
1. 并行哈希计算：使用ThreadPoolExecutor多线程加速
2. LSH索引：局部敏感哈希加速相似图片检索
3. 哈希前缀分组：按哈希前缀分组，减少比较次数
4. 分层检测：先快速筛选，再精确验证
5. 增强的可视化：精确框出重复区域

用法：
    python image_dedup_optimized.py <图片目录> [选项]

示例：
    python image_dedup_optimized.py D:\\proj_test\\proj_1
    python image_dedup_optimized.py D:\\proj_test\\proj_1 --workers 8
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
DEFAULT_PHASH_THRESHOLD = 5     # 感知哈希差异阈值（越小越严格）
DEFAULT_SSIM_THRESHOLD = 0.85   # SSIM 相似度阈值（越大越严格）
DEFAULT_HIST_THRESHOLD = 0.80   # 直方图相似度阈值
DEFAULT_FEATURE_MIN_MATCH = 10  # 特征匹配最小点数
DEFAULT_SUBIMAGE_THRESHOLD = 0.90  # 子图匹配阈值
DEFAULT_WORKERS = 4             # 并行工作进程数

# LSH索引配置
LSH_NUM_HASH_TABLES = 6        # LSH哈希表数量
LSH_NUM_HASH_FUNCTIONS = 3     # 每个哈希表的哈希函数数量
LSH_INDEX_PREFIX_BITS = 16     # 索引前缀位数


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
        """将哈希对象转换为二进制向量"""
        hash_int = int(str(hash_obj), 16)
        binary = np.array([(hash_int >> i) & 1 for i in range(self.hash_length)], 
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
        # 使用dHash的前缀作为主索引
        if 'dhash_int' in file_info:
            prefix = file_info['dhash_int'] & self.mask
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
# 图片加载与预处理（并行版）
# ============================================================

def load_single_image(filepath: str, hash_size: int = DEFAULT_HASH_SIZE) -> Optional[Dict]:
    """加载单张图片并计算特征（用于并行处理）"""
    try:
        img = Image.open(filepath)
        
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            img.seek(0)
        
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        elif img.mode == 'L':
            img = img.convert('RGB')
        
        size = img.size
        file_size = os.path.getsize(filepath)
        
        # 计算MD5
        with open(filepath, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        
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
    exclude_dirs = {'visualization', 'venv', '.git', '__pycache__', 'node_modules'}
    
    print(f"\n[1/4] 扫描目录: {root_dir}")
    
    # 收集所有图片路径
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        for p in root.rglob(f"*{ext}"):
            parts = p.relative_to(root).parts
            if not any(part in exclude_dirs for part in parts):
                image_paths.append(str(p))
        for p in root.rglob(f"*{ext.upper()}"):
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


def compute_ssim(img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
    """计算结构相似性指数 (SSIM)"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1_gray.astype(np.float64)
    img2 = img2_gray.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim)


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
                        threshold: int = DEFAULT_PHASH_THRESHOLD) -> List[DuplicateMatch]:
    """检测旋转/翻转"""
    results = []

    try:
        img1 = Image.open(info1['path']).convert('L').resize((256, 256))
        img2 = Image.open(info2['path']).convert('L').resize((256, 256))
    except:
        return results

    transforms = {
        "旋转90°": lambda im: im.rotate(90, expand=True),
        "旋转180°": lambda im: im.rotate(180, expand=True),
        "旋转270°": lambda im: im.rotate(270, expand=True),
        "水平翻转": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        "垂直翻转": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    }

    hash2 = imagehash.phash(img2, hash_size=DEFAULT_HASH_SIZE)

    for name, transform_fn in transforms.items():
        try:
            transformed = transform_fn(img1)
            hash_t = imagehash.phash(transformed, hash_size=DEFAULT_HASH_SIZE)
            diff = _hamming_distance(hash_t, hash2)
            if diff <= threshold:
                sim = 1.0 - diff / (hash_t.hash.size)
                results.append(DuplicateMatch(
                    image1=info1['path'],
                    image2=info2['path'],
                    match_type=f"疑似{name}",
                    similarity=round(sim, 4),
                    details=f"变换后哈希差异: {diff}",
                    severity="critical"
                ))
        except:
            continue

    return results


# ============================================================
# 结果去重与排序
# ============================================================

def deduplicate_matches(matches: List[DuplicateMatch]) -> List[DuplicateMatch]:
    """去除重复的匹配结果"""
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

    unique.sort(key=lambda m: (severity_order.get(m.severity, 9), -m.similarity))
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

def generate_html_report(matches: List[DuplicateMatch], output_path: str,
                         total_images: int, scan_dir: str, stats: PerformanceStats):
    """生成HTML报告"""
    severity_colors = {
        "critical": "#ff4444",
        "high": "#ff8800",
        "medium": "#ffcc00",
        "low": "#44aa44"
    }
    severity_labels = {
        "critical": "严重",
        "high": "高",
        "medium": "中",
        "low": "低"
    }

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>科研图片查重报告 - 优化版</title>
<style>
    body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; margin: 20px; background: #f5f5f5; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
    .summary {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .summary h2 {{ margin-top: 0; color: #2196F3; }}
    .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .stat-box {{ background: #f0f7ff; padding: 15px 25px; border-radius: 8px; text-align: center; }}
    .stat-box .number {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
    .stat-box .label {{ color: #666; font-size: 0.9em; }}
    .performance {{ background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 10px 0; }}
    .match {{ background: white; margin: 15px 0; padding: 20px; border-radius: 8px;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #ccc; }}
    .match.critical {{ border-left-color: #ff4444; }}
    .match.high {{ border-left-color: #ff8800; }}
    .match.medium {{ border-left-color: #ffcc00; }}
    .match.low {{ border-left-color: #44aa44; }}
    .match-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .match-type {{ font-weight: bold; font-size: 1.1em; }}
    .severity {{ padding: 3px 10px; border-radius: 12px; color: white; font-size: 0.85em; }}
    .similarity {{ font-size: 1.2em; font-weight: bold; color: #2196F3; }}
    .paths {{ color: #666; font-size: 0.9em; margin: 5px 0; word-break: break-all; }}
    .details {{ color: #888; font-size: 0.85em; }}
    .footer {{ text-align: center; color: #999; margin-top: 30px; padding: 20px; font-size: 0.85em; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<div class="container">
    <h1>🔬 科研图片查重报告 (优化版)</h1>

    <div class="summary">
        <h2>📊 扫描摘要</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="number">{total_images}</div>
                <div class="label">扫描图片数</div>
            </div>
            <div class="stat-box">
                <div class="number">{len(matches)}</div>
                <div class="label">发现可疑对</div>
            </div>
            <div class="stat-box">
                <div class="number">{sum(1 for m in matches if m.severity == 'critical')}</div>
                <div class="label">严重问题</div>
            </div>
            <div class="stat-box">
                <div class="number">{sum(1 for m in matches if m.severity == 'high')}</div>
                <div class="label">高风险</div>
            </div>
        </div>
        <p><strong>扫描目录:</strong> {scan_dir}</p>
        <p><strong>扫描时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="performance">
        <h2>⚡ 性能统计</h2>
        <p><strong>哈希计算耗时:</strong> {stats.hash_calculation_time:.2f}s</p>
        <p><strong>索引构建耗时:</strong> {stats.index_building_time:.2f}s</p>
        <p><strong>比较检测耗时:</strong> {stats.comparison_time:.2f}s</p>
        <p><strong>总比较次数:</strong> {stats.total_comparisons:,}</p>
        <p><strong>候选对数量:</strong> {stats.candidate_pairs:,}</p>
        <p><strong>优化比例:</strong> {stats.optimization_ratio:.1f}x</p>
    </div>

    <h2>🚨 可疑重复详情</h2>
"""

    for i, m in enumerate(matches, 1):
        color = severity_colors.get(m.severity, "#999")
        label = severity_labels.get(m.severity, "未知")
        sim_bar_width = int(m.similarity * 100)

        html += f"""
    <div class="match {m.severity}">
        <div class="match-header">
            <span class="match-type">#{i} {m.match_type}</span>
            <span class="severity" style="background:{color}">{label}</span>
        </div>
        <div class="similarity">相似度: {m.similarity*100:.1f}%</div>
        <div class="paths">
            <div>📷 {m.image1}</div>
            <div>📷 {m.image2}</div>
        </div>
        <div class="details">{m.details}</div>
        <div style="background:#eee; height:8px; border-radius:4px; margin-top:8px;">
            <div style="background:{color}; height:100%; width:{sim_bar_width}%; border-radius:4px;"></div>
        </div>
    </div>
"""

    html += f"""
    <div class="footer">
        <p>由 科研图片查重工具(优化版) 生成 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>优化技术参考: <a href="https://github.com/ayumilove/xImageDuplicateChecker">xImageDuplicateChecker</a> (MIT License)</p>
    </div>
</div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  HTML报告已生成: {output_path}")


def generate_csv_report(matches: List[DuplicateMatch], output_path: str):
    """生成CSV报告"""
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['序号', '问题类型', '严重程度', '相似度', '图片1', '图片2', '详细信息'])
        for i, m in enumerate(matches, 1):
            writer.writerow([
                i, m.match_type, m.severity,
                f"{m.similarity*100:.1f}%",
                m.image1, m.image2, m.details
            ])
    print(f"  CSV报告已生成: {output_path}")


# ============================================================
# 主扫描流程（优化版）
# ============================================================

def find_duplicates_optimized(images: List[Dict],
                              stats: PerformanceStats,
                              pthresh: int = DEFAULT_PHASH_THRESHOLD,
                              sthresh: float = DEFAULT_SSIM_THRESHOLD,
                              hthresh: float = DEFAULT_HIST_THRESHOLD,
                              check_rotations: bool = True,
                              use_lsh: bool = True) -> List[DuplicateMatch]:
    """执行所有检测（优化版：使用索引加速）"""
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
                        matches.append(m)
                        exact_pairs.add((group[i]['path'], group[j]['path']))

    print(f"  [✓] 阶段1 完全相同: {len(matches)} 对")

    # ---- 阶段2: 构建索引 ----
    print(f"  [·] 阶段2 构建索引...")
    start_time = time.time()
    
    if use_lsh:
        # 使用LSH索引
        index = LSHIndex()
        index.build(images)
        candidate_pairs = index.get_candidate_pairs()
        index_type = "LSH"
    else:
        # 使用前缀索引
        index = PrefixIndex()
        index.build(images)
        candidate_pairs = index.get_candidate_pairs()
        index_type = "前缀"
    
    index_time = time.time() - start_time
    stats.update(index_building_time=index_time)
    
    # 计算优化比例
    naive_comparisons = n * (n - 1) // 2
    optimization_ratio = naive_comparisons / len(candidate_pairs) if candidate_pairs else 1
    stats.update(
        candidate_pairs=len(candidate_pairs),
        optimization_ratio=optimization_ratio
    )
    
    print(f"  [✓] 阶段2 {index_type}索引构建完成: {len(candidate_pairs)} 候选对")
    print(f"      优化比例: {optimization_ratio:.1f}x (从 {naive_comparisons:,} 减少到 {len(candidate_pairs):,})")

    # 过滤掉已经是完全相同的对，以及同实验不同通道的对
    candidate_pairs = {
        (p1, p2) for p1, p2 in candidate_pairs
        if (p1, p2) not in exact_pairs and
           (p2, p1) not in exact_pairs and
           not _is_same_experiment_channel(p1, p2)
    }

    # ---- 阶段3: 对候选对做详细检测 ----
    print(f"  [·] 阶段3 详细检测 {len(candidate_pairs)} 个候选对...")
    start_time = time.time()
    
    # 创建路径到图片信息的映射
    path_to_info = {img['path']: img for img in images}
    
    comparison_count = 0
    for p1, p2 in candidate_pairs:
        if p1 not in path_to_info or p2 not in path_to_info:
            continue
        
        img1, img2 = path_to_info[p1], path_to_info[p2]
        comparison_count += 1

        # pHash 检测
        m = check_phash_similarity(img1, img2, pthresh)
        if m:
            matches.append(m)

        # dHash 检测
        m = check_dhash_similarity(img1, img2, pthresh)
        if m and not any(
            mm.match_type == "感知哈希相似(pHash)" and
            ((mm.image1 == img1['path'] and mm.image2 == img2['path']) or
             (mm.image1 == img2['path'] and mm.image2 == img1['path']))
            for mm in matches
        ):
            matches.append(m)

        # SSIM
        m = check_ssim_similarity(img1, img2, sthresh)
        if m:
            matches.append(m)

        # 直方图
        m = check_histogram_similarity(img1, img2, hthresh)
        if m:
            matches.append(m)

        m = check_brightness_contrast(img1, img2)
        if m:
            matches.append(m)

        # 旋转/翻转
        if check_rotations:
            rot_matches = check_rotation_flip(img1, img2, pthresh)
            matches.extend(rot_matches)

        if comparison_count % 500 == 0:
            print(f"    进度: {comparison_count}/{len(candidate_pairs)}")

    comparison_time = time.time() - start_time
    stats.update(
        comparison_time=comparison_time,
        total_comparisons=comparison_count
    )
    
    print(f"  [✓] 阶段3 详细检测完成: {comparison_count} 对比较, 耗时 {comparison_time:.2f}s")

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
  python image_dedup_optimized.py D:\\proj_test\\proj_1 --no-lsh
  
优化技术参考: https://github.com/ayumilove/xImageDuplicateChecker (MIT License)
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
                        help="输出报告文件路径 (支持 .html / .csv)")
    parser.add_argument("--no-rotation", action="store_true",
                        help="跳过旋转/翻转检测")
    parser.add_argument("--no-lsh", action="store_true",
                        help="使用简单前缀索引代替LSH索引")

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
    print(f"  并行线程: {args.workers}")
    print(f"  索引类型: {'简单前缀' if args.no_lsh else 'LSH'}")

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
    )

    # 去重排序
    matches = deduplicate_matches(matches)

    # 生成可视化对比图
    vis_dir = os.path.join(args.directory, "visualization")
    if matches:
        print(f"\n[4/4] 生成可视化对比图...")
        vis_count = 0
        for i, m in enumerate(matches):
            if i % 20 == 0 and i > 0:
                print(f"  进度: {i}/{len(matches)}")
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
                    print(f"    • {m.match_type} (相似度: {m.similarity*100:.1f}%)")
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
    else:
        if not report_path.endswith('.html'):
            report_path += '.html'
        generate_html_report(matches, report_path, len(images), args.directory, stats)

    csv_path = report_path.rsplit('.', 1)[0] + '.csv'
    if csv_path != report_path:
        generate_csv_report(matches, csv_path)

    # 打印性能统计
    print(stats.get_summary())
    
    print(f"\n✅ 扫描完成！")


if __name__ == "__main__":
    main()
