#!/usr/bin/env python3
"""
科研图片查重工具
================
检测科研论文中常见的图片重复/造假手段：
1. 完全复制 / 微小修改复制
2. 旋转 / 翻转
3. 亮度/对比度/曝光度调整
4. 缩放
5. 裁剪（子图重复）
6. 边缘重叠拼接
7. 颜色通道/色调调整
8. 局部区域复制粘贴

用法：
    python image_dedup.py <图片目录> [选项]

示例：
    python image_dedup.py D:\\proj_test\\proj_1
    python image_dedup.py D:\\proj_test\\proj_1 --threshold 5 --report report.html
"""

import os
import sys
import argparse
import hashlib
import json
import csv
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from itertools import combinations

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


@dataclass
class DuplicateMatch:
    """重复匹配结果"""
    image1: str
    image2: str
    match_type: str       # 匹配类型
    similarity: float     # 相似度 (0~1)
    details: str = ""     # 详细说明
    severity: str = "medium"  # low / medium / high / critical


# ============================================================
# 图片加载与预处理
# ============================================================

# ============================================================
# 同实验通道过滤
# ============================================================

# 通道后缀模式
CHANNEL_SUFFIXES = ['_CH1', '_CH2', '_CH3', '_CH4', '_CH5',
                    '_Overlay', '_Bright', '_DIC', '_GFP', '_RFP',
                    '_DAPI', '_Cy3', '_Cy5', '_FITC', '_TRITC']


def _get_experiment_key(filepath: str) -> tuple:
    """
    提取实验标识：(父目录, 基础文件名去掉通道后缀)
    例如: MOUSE1/40-1_02/40_CH2.tif -> ('MOUSE1/40-1_02', '40')
    """
    path = Path(filepath)
    parent = str(path.parent)  # 父目录
    stem = path.stem  # 文件名不含扩展名，如 40_CH2

    # 去掉通道后缀
    base = stem
    for suffix in CHANNEL_SUFFIXES:
        if stem.upper().endswith(suffix.upper()):
            base = stem[:len(stem) - len(suffix)]
            break

    return (parent, base)


def _is_same_experiment_channel(filepath1: str, filepath2: str) -> bool:
    """
    判断两张图是否来自同一样本的不同通道
    同目录 + 同基础名 + 不同通道后缀 -> True
    """
    key1 = _get_experiment_key(filepath1)
    key2 = _get_experiment_key(filepath2)

    # 必须同目录同基础名
    if key1 != key2:
        return False

    # 确认是不同文件（不同通道）
    return filepath1 != filepath2


# ============================================================
# 图片加载与预处理
# ============================================================


def load_image_info(filepath: str, hash_size: int = DEFAULT_HASH_SIZE) -> Optional[ImageInfo]:
    """加载图片并计算各种特征"""
    try:
        img = Image.open(filepath)

        # 处理多帧图片（如某些科研格式），只取第一帧
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            img.seek(0)

        # 转为RGB
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

        # 计算颜色直方图（归一化）
        hist_r = np.histogram(np.array(img)[:,:,0], bins=64, range=(0,256))[0]
        hist_g = np.histogram(np.array(img)[:,:,1], bins=64, range=(0,256))[0]
        hist_b = np.histogram(np.array(img)[:,:,2], bins=64, range=(0,256))[0]
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
        hist = hist / hist.sum()  # 归一化

        # 灰度图数组（用于SSIM等计算）
        gray_array = np.array(gray.resize((256, 256)))  # 统一大小便于比较

        return ImageInfo(
            path=filepath,
            size=size,
            file_size=file_size,
            md5=md5,
            phash=phash,
            dhash=dhash,
            ahash=ahash,
            chash=chash,
            hist=hist,
            gray_array=gray_array,
        )
    except Exception as e:
        print(f"  [警告] 无法加载图片: {filepath} - {e}")
        return None


# ============================================================
# 检测方法 1: 精确重复 (MD5)
# ============================================================

def check_exact_duplicate(info1: ImageInfo, info2: ImageInfo) -> Optional[DuplicateMatch]:
    """检测完全相同的文件"""
    if info1.md5 == info2.md5:
        return DuplicateMatch(
            image1=info1.path,
            image2=info2.path,
            match_type="完全相同",
            similarity=1.0,
            details=f"MD5一致: {info1.md5}",
            severity="critical"
        )
    return None


# ============================================================
# 检测方法 2: 感知哈希 (pHash) - 检测微小修改
# ============================================================

def check_phash_similarity(info1: ImageInfo, info2: ImageInfo,
                           threshold: int = DEFAULT_PHASH_THRESHOLD) -> Optional[DuplicateMatch]:
    """检测感知哈希相似度（对亮度、对比度、轻微修改鲁棒）"""
    diff = info1.phash - info2.phash
    if diff <= threshold:
        sim = 1.0 - diff / (info1.phash.hash.size)
        return DuplicateMatch(
            image1=info1.path,
            image2=info2.path,
            match_type="感知哈希相似(pHash)",
            similarity=round(sim, 4),
            details=f"哈希差异: {diff}/{info1.phash.hash.size}",
            severity="critical" if diff <= 2 else "high"
        )
    return None


# ============================================================
# 检测方法 3: 差异哈希 (dHash) - 检测渐变变化
# ============================================================

def check_dhash_similarity(info1: ImageInfo, info2: ImageInfo,
                           threshold: int = DEFAULT_PHASH_THRESHOLD) -> Optional[DuplicateMatch]:
    """检测差异哈希相似度（对渐变、边缘变化敏感）"""
    diff = info1.dhash - info2.dhash
    if diff <= threshold:
        sim = 1.0 - diff / (info1.dhash.hash.size)
        return DuplicateMatch(
            image1=info1.path,
            image2=info2.path,
            match_type="差异哈希相似(dHash)",
            similarity=round(sim, 4),
            details=f"哈希差异: {diff}/{info1.dhash.hash.size}",
            severity="critical" if diff <= 2 else "high"
        )
    return None


# ============================================================
# 检测方法 4: 颜色直方图 - 检测曝光度/亮度调整
# ============================================================

def check_histogram_similarity(info1: ImageInfo, info2: ImageInfo,
                               threshold: float = DEFAULT_HIST_THRESHOLD) -> Optional[DuplicateMatch]:
    """
    检测颜色直方图相似度
    对曝光度、亮度、对比度调整非常敏感
    即使图片内容不同，如果进行了相同的曝光调整，直方图也会相似
    """
    # 使用相关性比较
    hist1 = info1.hist - info1.hist.mean()
    hist2 = info2.hist - info2.hist.mean()

    correlation = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2) + 1e-10)

    # 使用巴氏距离
    bhattacharyya = np.sum(np.sqrt(info1.hist * info2.hist))

    avg_sim = (max(0, correlation) + bhattacharyya) / 2

    if avg_sim >= threshold:
        details = f"直方图相关性: {correlation:.3f}, 巴氏系数: {bhattacharyya:.3f}"
        if correlation > 0.95:
            details += " [疑似曝光度/亮度统一调整]"
        return DuplicateMatch(
            image1=info1.path,
            image2=info2.path,
            match_type="直方图相似(疑似曝光调整)",
            similarity=round(avg_sim, 4),
            details=details,
            severity="high" if avg_sim > 0.95 else "medium"
        )
    return None


# ============================================================
# 检测方法 5: SSIM - 结构相似性
# ============================================================

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


def check_ssim_similarity(info1: ImageInfo, info2: ImageInfo,
                          threshold: float = DEFAULT_SSIM_THRESHOLD) -> Optional[DuplicateMatch]:
    """检测结构相似性"""
    ssim = compute_ssim(info1.gray_array, info2.gray_array)

    if ssim >= threshold:
        return DuplicateMatch(
            image1=info1.path,
            image2=info2.path,
            match_type="结构相似(SSIM)",
            similarity=round(ssim, 4),
            details=f"SSIM值: {ssim:.4f}",
            severity="critical" if ssim > 0.95 else "high"
        )
    return None


# ============================================================
# 检测方法 6: 旋转/翻转检测
# ============================================================

def check_rotation_flip(info1: ImageInfo, info2: ImageInfo,
                        threshold: int = DEFAULT_PHASH_THRESHOLD) -> list[DuplicateMatch]:
    """
    检测旋转（90/180/270度）和翻转
    通过比较各种变换后的哈希值
    """
    results = []

    try:
        img1 = Image.open(info1.path).convert('L').resize((256, 256))
        img2 = Image.open(info2.path).convert('L').resize((256, 256))
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
            diff = hash_t - hash2
            if diff <= threshold:
                sim = 1.0 - diff / (hash_t.hash.size)
                results.append(DuplicateMatch(
                    image1=info1.path,
                    image2=info2.path,
                    match_type=f"疑似{name}",
                    similarity=round(sim, 4),
                    details=f"变换后哈希差异: {diff}",
                    severity="critical"
                ))
        except:
            continue

    return results


# ============================================================
# 检测方法 7: 缩放检测
# ============================================================

def check_scale(info1: ImageInfo, info2: ImageInfo,
                threshold: float = DEFAULT_SSIM_THRESHOLD) -> Optional[DuplicateMatch]:
    """
    检测缩放关系
    将两张图片缩放到相同大小后比较
    """
    if abs(info1.size[0] / info1.size[1] - info2.size[0] / info2.size[1]) > 0.1:
        return None  # 宽高比差异太大，不太可能是缩放

    try:
        img1 = Image.open(info1.path).convert('L').resize((256, 256))
        img2 = Image.open(info2.path).convert('L').resize((256, 256))
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        ssim = compute_ssim(arr1, arr2)

        if ssim >= threshold:
            size_ratio = (info1.size[0] * info1.size[1]) / (info2.size[0] * info2.size[1])
            return DuplicateMatch(
                image1=info1.path,
                image2=info2.path,
                match_type="疑似缩放",
                similarity=round(ssim, 4),
                details=f"尺寸比: {info1.size} vs {info2.size}, 面积比: {size_ratio:.2f}",
                severity="high"
            )
    except:
        pass
    return None


# ============================================================
# 检测方法 8: 子图/裁剪检测
# ============================================================

def check_subimage(info1: ImageInfo, info2: ImageInfo,
                   threshold: float = DEFAULT_SUBIMAGE_THRESHOLD) -> Optional[DuplicateMatch]:
    """
    检测一张图是否是另一张图的裁剪/子图
    使用模板匹配方法
    """
    try:
        # 确保img1是较大的图
        if info1.file_size < info2.file_size:
            info1, info2 = info2, info1

        img1 = Image.open(info1.path).convert('L')
        img2 = Image.open(info2.path).convert('L')

        w1, h1 = img1.size
        w2, h2 = img2.size

        if w2 >= w1 or h2 >= h1:
            return None  # img2不比img1小，不可能是子图

        # 多尺度检测
        for scale in [1.0, 0.8, 0.6, 0.4]:
            new_w, new_h = int(w2 * scale), int(h2 * scale)
            if new_w < 20 or new_h < 20:
                continue

            img2_scaled = img2.resize((new_w, new_h))

            # 使用滑动窗口做模板匹配
            arr1 = np.array(img1, dtype=np.float64)
            arr2 = np.array(img2_scaled, dtype=np.float64)

            # 简化的归一化互相关
            step = max(1, min(w1, h1) // 50)  # 步长

            best_score = 0
            best_pos = (0, 0)

            for y in range(0, h1 - new_h + 1, step):
                for x in range(0, w1 - new_w + 1, step):
                    patch = arr1[y:y+new_h, x:x+new_w]
                    if patch.shape != arr2.shape:
                        continue

                    # 归一化互相关
                    p_norm = (patch - patch.mean()) / (patch.std() + 1e-10)
                    t_norm = (arr2 - arr2.mean()) / (arr2.std() + 1e-10)
                    ncc = np.mean(p_norm * t_norm)

                    if ncc > best_score:
                        best_score = ncc
                        best_pos = (x, y)

            if best_score >= threshold:
                return DuplicateMatch(
                    image1=info1.path,
                    image2=info2.path,
                    match_type="疑似子图/裁剪",
                    similarity=round(float(best_score), 4),
                    details=f"在原图位置({best_pos[0]},{best_pos[1]})处发现匹配, 缩放比例:{scale}",
                    severity="critical"
                )
    except Exception as e:
        pass
    return None


# ============================================================
# 检测方法 9: 边缘重叠检测
# ============================================================

def check_edge_overlap(info1: ImageInfo, info2: ImageInfo,
                       threshold: float = 0.7) -> Optional[DuplicateMatch]:
    """
    检测两张图片的边缘是否重叠（拼接造假）
    比较一张图的右/下边缘与另一张图的左/上边缘
    """
    try:
        img1 = Image.open(info1.path).convert('L')
        img2 = Image.open(info2.path).convert('L')

        # 统一高度进行水平拼接检测
        h = min(img1.size[1], img2.size[1])
        if h < 20:
            return None

        img1_r = img1.resize((int(img1.size[0] * h / img1.size[1]), h))
        img2_r = img2.resize((int(img2.size[0] * h / img2.size[1]), h))

        arr1 = np.array(img1_r, dtype=np.float64)
        arr2 = np.array(img2_r, dtype=np.float64)

        # 检测 img1 右边缘 与 img2 左边缘
        strip_width = min(20, arr1.shape[1] // 4, arr2.shape[1] // 4)
        if strip_width < 3:
            return None

        right_strip = arr1[:, -strip_width:]
        left_strip = arr2[:, :strip_width]

        # 归一化互相关
        r_norm = (right_strip - right_strip.mean()) / (right_strip.std() + 1e-10)
        l_norm = (left_strip - left_strip.mean()) / (left_strip.std() + 1e-10)
        h_corr = np.mean(r_norm * l_norm)

        # 统一宽度进行垂直拼接检测
        w = min(img1.size[0], img2.size[0])
        if w >= 20:
            img1_r2 = img1.resize((w, int(img1.size[1] * w / img1.size[0])))
            img2_r2 = img2.resize((w, int(img2.size[1] * w / img2.size[0])))

            arr1v = np.array(img1_r2, dtype=np.float64)
            arr2v = np.array(img2_r2, dtype=np.float64)

            strip_h = min(20, arr1v.shape[0] // 4, arr2v.shape[0] // 4)
            if strip_h >= 3:
                bottom_strip = arr1v[-strip_h:, :]
                top_strip = arr2v[:strip_h, :]

                b_norm = (bottom_strip - bottom_strip.mean()) / (bottom_strip.std() + 1e-10)
                t_norm = (top_strip - top_strip.mean()) / (top_strip.std() + 1e-10)
                v_corr = np.mean(b_norm * t_norm)
            else:
                v_corr = 0
        else:
            v_corr = 0

        max_corr = max(h_corr, v_corr)
        direction = "水平" if h_corr >= v_corr else "垂直"

        if max_corr >= threshold:
            return DuplicateMatch(
                image1=info1.path,
                image2=info2.path,
                match_type=f"疑似{direction}边缘重叠/拼接",
                similarity=round(float(max_corr), 4),
                details=f"边缘相关性: 水平={h_corr:.3f}, 垂直={v_corr:.3f}",
                severity="high"
            )
    except:
        pass
    return None


# ============================================================
# 检测方法 10: ORB特征匹配 - 检测旋转/缩放/裁剪
# ============================================================

def check_orb_features(info1: ImageInfo, info2: ImageInfo,
                       min_matches: int = DEFAULT_FEATURE_MIN_MATCH) -> Optional[DuplicateMatch]:
    """
    使用ORB特征点匹配
    对旋转、缩放、裁剪、轻微修改都有较好的鲁棒性
    """
    try:
        import cv2
    except ImportError:
        return None  # 没有OpenCV则跳过

    try:
        img1 = cv2.imread(info1.path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(info2.path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return None

        # 创建ORB检测器
        orb = cv2.ORB_create(nfeatures=500)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return None

        # BF匹配器
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # 过滤好的匹配
        good_matches = [m for m in matches if m.distance < 50]

        if len(good_matches) >= min_matches:
            sim = min(1.0, len(good_matches) / max(len(kp1), len(kp2)))
            return DuplicateMatch(
                image1=info1.path,
                image2=info2.path,
                match_type="特征点匹配(ORB)",
                similarity=round(sim, 4),
                details=f"匹配点数: {len(good_matches)}/{len(matches)}, 关键点: {len(kp1)} vs {len(kp2)}",
                severity="high" if sim > 0.3 else "medium"
            )
    except:
        pass
    return None


# ============================================================
# 检测方法 11: 亮度/对比度调整检测
# ============================================================

def check_brightness_contrast(info1: ImageInfo, info2: ImageInfo,
                              threshold: float = 0.90) -> Optional[DuplicateMatch]:
    """
    专门检测亮度和对比度调整
    通过分析图片的统计特性
    """
    try:
        arr1 = info1.gray_array.astype(np.float64)
        arr2 = info2.gray_array.astype(np.float64)

        mean1, std1 = arr1.mean(), arr1.std()
        mean2, std2 = arr2.mean(), arr2.std()

        # 亮度差异
        brightness_ratio = min(mean1, mean2) / (max(mean1, mean2) + 1e-10)

        # 对比度差异
        contrast_ratio = min(std1, std2) / (max(std1, std2) + 1e-10)

        # 结构保持度（排除亮度对比度影响）
        arr1_norm = (arr1 - mean1) / (std1 + 1e-10)
        arr2_norm = (arr2 - mean2) / (std2 + 1e-10)
        structure_sim = np.mean(arr1_norm * arr2_norm)
        structure_sim = max(0, structure_sim)

        # 如果结构高度相似但亮度/对比度不同，说明可能进行了调整
        if structure_sim >= threshold and (brightness_ratio < 0.9 or contrast_ratio < 0.9):
            return DuplicateMatch(
                image1=info1.path,
                image2=info2.path,
                match_type="疑似亮度/对比度调整",
                similarity=round(structure_sim, 4),
                details=f"结构相似度:{structure_sim:.3f}, 亮度比:{brightness_ratio:.3f}, 对比度比:{contrast_ratio:.3f}",
                severity="high"
            )
    except:
        pass
    return None


# ============================================================
# 检测方法 12: 局部区域复制检测
# ============================================================

def check_internal_duplicate(image_path: str, block_size: int = 64,
                             threshold: float = 0.95) -> list[DuplicateMatch]:
    """
    检测单张图片内部是否存在区域复制粘贴
    将图片分块，比较各块之间的相似度
    """
    results = []
    try:
        img = Image.open(image_path).convert('L')
        arr = np.array(img, dtype=np.float64)
        h, w = arr.shape

        if h < block_size * 2 or w < block_size * 2:
            return results

        # 采样一些块进行比较
        step = block_size // 2
        blocks = []
        positions = []

        for y in range(0, h - block_size, step):
            for x in range(0, w - block_size, step):
                block = arr[y:y+block_size, x:x+block_size]
                blocks.append(block)
                positions.append((x, y))

        # 随机采样比较（避免O(n^2)复杂度）
        import random
        n_blocks = len(blocks)
        if n_blocks > 200:
            indices = random.sample(range(n_blocks), 200)
        else:
            indices = list(range(n_blocks))

        compared = set()
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                xi, yi = positions[idx_i]
                xj, yj = positions[idx_j]

                # 跳过相邻块
                if abs(xi - xj) < block_size and abs(yi - yj) < block_size:
                    continue

                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                if pair in compared:
                    continue
                compared.add(pair)

                b1 = blocks[idx_i]
                b2 = blocks[idx_j]

                # 归一化互相关
                n1 = (b1 - b1.mean()) / (b1.std() + 1e-10)
                n2 = (b2 - b2.mean()) / (b2.std() + 1e-10)
                corr = np.mean(n1 * n2)

                if corr >= threshold:
                    results.append(DuplicateMatch(
                        image1=image_path,
                        image2=image_path,
                        match_type="内部区域复制",
                        similarity=round(float(corr), 4),
                        details=f"区域({xi},{yi})与({xj},{yj})高度相似",
                        severity="critical"
                    ))
                    if len(results) >= 3:
                        return results
    except:
        pass
    return results


# ============================================================
# 可视化：框出相似部位
# ============================================================


def _find_matching_region(img1_gray: np.ndarray, img2_gray: np.ndarray, match_type: str = ""):
    """
    用多种方法找两张图之间的对应区域
    返回: (x1, y1, w1, h1, x2, y2, w2, h2) 或 None
    """
    try:
        import cv2
    except ImportError:
        return None

    try:
        h1, w1 = img1_gray.shape[:2]
        h2, w2 = img2_gray.shape[:2]

        # 方法1: 对于非常相似的图片，使用差异图找不同区域
        # 统一到相同大小进行比较
        target_h = min(h1, h2, 600)
        scale1 = target_h / h1
        scale2 = target_h / h2
        new_w1 = int(w1 * scale1)
        new_w2 = int(w2 * scale2)
        
        img1_resized = cv2.resize(img1_gray, (new_w1, target_h))
        img2_resized = cv2.resize(img2_gray, (new_w2, target_h))
        
        # 计算差异图
        # 确保两张图大小相同再做差
        img2_for_diff = cv2.resize(img2_resized, (new_w1, target_h))
        diff = cv2.absdiff(img1_resized, img2_for_diff)
        
        # 二值化差异图
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去噪
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        # 找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours) > 0:
            # 找最大的轮廓（主要差异区域）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 如果差异区域足够大（超过图片面积的1%）
            if area > (new_w1 * target_h * 0.01):
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 转换回原始坐标
                x1_orig = int(x / scale1)
                y1_orig = int(y / scale1)
                w1_orig = int(w / scale1)
                h1_orig = int(h / scale1)
                
                # 对于第二张图，找对应的位置
                x2_orig = int(x / scale2 * new_w2 / new_w1)
                y2_orig = int(y / scale2)
                w2_orig = int(w / scale2 * new_w2 / new_w1)
                h2_orig = int(h / scale2)
                
                # 确保坐标在图片范围内
                x1_orig = max(0, min(x1_orig, w1 - 10))
                y1_orig = max(0, min(y1_orig, h1 - 10))
                x2_orig = max(0, min(x2_orig, w2 - 10))
                y2_orig = max(0, min(y2_orig, h2 - 10))
                
                w1_orig = min(w1_orig, w1 - x1_orig)
                h1_orig = min(h1_orig, h1 - y1_orig)
                w2_orig = min(w2_orig, w2 - x2_orig)
                h2_orig = min(h2_orig, h2 - y2_orig)
                
                if w1_orig > 10 and h1_orig > 10 and w2_orig > 10 and h2_orig > 10:
                    return (x1_orig, y1_orig, w1_orig, h1_orig, x2_orig, y2_orig, w2_orig, h2_orig)

        # 方法2: 使用特征匹配（对有旋转/变换的图片有效）
        max_dim = 800
        scale1_feat = min(1.0, max_dim / max(w1, h1))
        scale2_feat = min(1.0, max_dim / max(w2, h2))

        img1_small = cv2.resize(img1_gray, (int(w1*scale1_feat), int(h1*scale1_feat)))
        img2_small = cv2.resize(img2_gray, (int(w2*scale2_feat), int(h2*scale2_feat)))

        # ORB特征检测
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(img1_small, None)
        kp2, des2 = orb.detectAndCompute(img2_small, None)

        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
            # 特征匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)

            # Lowe's ratio test
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) >= 4:
                # 提取匹配点坐标
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # 用RANSAC找单应矩阵
                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                if H is not None:
                    inliers = mask.ravel().sum()
                    if inliers >= 4:
                        # 计算匹配区域
                        h1s, w1s = img1_small.shape[:2]
                        corners1 = np.float32([[0,0], [w1s,0], [w1s,h1s], [0,h1s]]).reshape(-1, 1, 2)
                        projected = cv2.perspectiveTransform(corners1, H)

                        # 计算包围盒
                        proj_orig = projected / scale2_feat
                        x2 = int(max(0, proj_orig[:, 0, 0].min()))
                        y2 = int(max(0, proj_orig[:, 0, 1].min()))
                        x2_end = int(min(w2, proj_orig[:, 0, 0].max()))
                        y2_end = int(min(h2, proj_orig[:, 0, 1].max()))
                        w2_box = x2_end - x2
                        h2_box = y2_end - y2

                        if w2_box > 10 and h2_box > 10:
                            # 在img1中找对应的区域（匹配点的包围盒）
                            pts1_orig = pts1 / scale1_feat
                            x1 = int(max(0, pts1_orig[:, 0, 0].min()))
                            y1 = int(max(0, pts1_orig[:, 0, 1].min()))
                            x1_end = int(min(w1, pts1_orig[:, 0, 0].max()))
                            y1_end = int(min(h1, pts1_orig[:, 0, 1].max()))
                            w1_box = x1_end - x1
                            h1_box = y1_end - y1
                            
                            if w1_box > 10 and h1_box > 10:
                                return (x1, y1, w1_box, h1_box, x2, y2, w2_box, h2_box)

        # 方法3: 对于完全相同的图片，返回None让调用者决定如何处理
        return None

    except Exception:
        return None


def _find_template_region(img1_gray: np.ndarray, img2_gray: np.ndarray):
    """
    用模板匹配找img2在img1中的位置（适用于裁剪/子图情况）
    返回: (x, y, w, h) 在img1中的位置，或 None
    """
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    # 确保img1是大图
    if w2 > w1 or h2 > h1:
        img1_gray, img2_gray = img2_gray, img1_gray
        h1, w1 = img1_gray.shape[:2]
        h2, w2 = img2_gray.shape[:2]

    if w2 > w1 or h2 > h1:
        return None

    # 多尺度模板匹配
    best_val = -1
    best_loc = None
    best_scale = 1.0

    for scale in [1.0, 0.9, 0.8, 0.7, 0.6]:
        new_w, new_h = int(w2 * scale), int(h2 * scale)
        if new_w < 20 or new_h < 20 or new_w > w1 or new_h > h1:
            continue

        template = cv2.resize(img2_gray, (new_w, new_h)) if 'cv2' in dir() else None
        if template is None:
            # 用PIL缩放
            from PIL import Image
            template = np.array(Image.fromarray(img2_gray).resize((new_w, new_h)))

        # 归一化互相关
        t = template.astype(np.float64)
        t_norm = (t - t.mean()) / (t.std() + 1e-10)

        step = max(1, min(w1, h1) // 100)
        for y in range(0, h1 - new_h + 1, step):
            for x in range(0, w1 - new_w + 1, step):
                patch = img1_gray[y:y+new_h, x:x+new_w].astype(np.float64)
                p_norm = (patch - patch.mean()) / (patch.std() + 1e-10)
                score = np.mean(p_norm * t_norm)

                if score > best_val:
                    best_val = score
                    best_loc = (x, y, new_w, new_h)
                    best_scale = scale

    if best_val > 0.7 and best_loc:
        return best_loc
    return None


def _extract_source_info(filepath: str) -> dict:
    """
    从文件路径提取来源信息
    例如: /mnt/d/proj_test/proj_1/MOUSE1/40-1_08/40_CH2.tif
    返回: {"group": "MOUSE1", "subfolder": "40-1_08", "filename": "40_CH2.tif", "channel": "CH2"}
    """
    path = Path(filepath)
    parts = path.parts

    info = {
        "group": "",
        "subfolder": "",
        "filename": path.name,
        "channel": ""
    }

    # 提取通道
    stem = path.stem.upper()
    for ch in ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'OVERLAY', 'BRIGHT', 'DIC', 'GFP', 'RFP', 'DAPI']:
        if ch in stem:
            info["channel"] = ch
            break

    # 从路径中提取 MOUSE 组和子文件夹
    # 路径格式: .../proj_1/MOUSE1/40-1_08/40_CH2.tif
    for i, part in enumerate(parts):
        if part.upper().startswith('MOUSE') or part.upper().startswith('NOUSE'):
            info["group"] = part
            # 下一个部分是子文件夹
            if i + 1 < len(parts) - 1:  # -1 因为最后是文件名
                info["subfolder"] = parts[i + 1]
            break
        elif part.lower().startswith('mouse') or part.lower().startswith('nouse'):
            info["group"] = part
            if i + 1 < len(parts) - 1:
                info["subfolder"] = parts[i + 1]
            break

    return info


def _draw_text_with_bg(img, text, pos, font, font_scale, text_color, bg_color, thickness=2):
    """绘制带背景的文字，提高可读性"""
    import cv2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # 画背景矩形
    cv2.rectangle(img, (x-2, y-th-4), (x+tw+4, y+baseline+2), bg_color, -1)
    # 画文字
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    return th + baseline + 6


def _parse_internal_duplicate_coords(details: str):
    """
    从内部区域复制的details中提取坐标
    格式: '区域(x1,y1)与(x2,y2)高度相似'
    返回: [(x1, y1), (x2, y2)] 或 None
    """
    import re
    pattern = r'区域\((\d+),(\d+)\)与\((\d+),(\d+)\)'
    match = re.search(pattern, details)
    if match:
        x1, y1, x2, y2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        return [(x1, y1), (x2, y2)]
    return None


def _parse_subimage_coords(details: str):
    """
    从子图/裁剪的details中提取坐标
    格式: '在原图位置(x,y)处发现匹配, 缩放比例:scale'
    返回: (x, y) 或 None
    """
    import re
    pattern = r'在原图位置\((\d+),(\d+)\)'
    match = re.search(pattern, details)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def _find_difference_region(img1_gray: np.ndarray, img2_gray: np.ndarray):
    """
    通过差异图找到两张相似图片的不同区域
    返回: (x, y, w, h) 在统一坐标系下的区域，或 None
    """
    try:
        import cv2
        # 统一大小
        h, w = 400, 600
        img1 = cv2.resize(img1_gray, (w, h))
        img2 = cv2.resize(img2_gray, (w, h))
        
        # 计算差异
        diff = cv2.absdiff(img1, img2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((7,7), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 合并所有轮廓的包围盒
            all_points = np.vstack(contours)
            x, y, bw, bh = cv2.boundingRect(all_points)
            if bw > 20 and bh > 20:
                return (x, y, bw, bh, w, h)  # 返回区域和参考尺寸
    except:
        pass
    return None


def create_visualization(match: DuplicateMatch, output_dir: str) -> Optional[str]:
    """
    为一对相似图片创建可视化对比图
    用红框标出相似区域，左右并排显示，标注来源信息
    """
    try:
        import cv2
    except ImportError:
        return _create_visualization_pil(match, output_dir)

    try:
        img1 = cv2.imread(match.image1)
        img2 = cv2.imread(match.image2)

        if img1 is None or img2 is None:
            return None

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 提取来源信息
        src1 = _extract_source_info(match.image1)
        src2 = _extract_source_info(match.image2)

        # 统一显示高度
        target_h = 450
        scale1 = target_h / img1.shape[0]
        scale2 = target_h / img2.shape[0]
        disp1 = cv2.resize(img1, (int(img1.shape[1] * scale1), target_h))
        disp2 = cv2.resize(img2, (int(img2.shape[1] * scale2), target_h))
        
        # 根据匹配类型选择框选策略
        region = None
        block_size = 64  # 默认块大小，用于内部区域复制
        
        # 1. 内部区域复制：从details提取坐标
        if match.match_type == "内部区域复制":
            coords = _parse_internal_duplicate_coords(match.details)
            if coords:
                (x1, y1), (x2, y2) = coords
                # 两个区域都在同一张图上，需要分别框出
                sx1, sy1 = int(x1 * scale1), int(y1 * scale1)
                cv2.rectangle(disp1, (sx1, sy1), (sx1 + int(block_size * scale1), sy1 + int(block_size * scale1)), (0, 0, 255), 3)
                sx2, sy2 = int(x2 * scale1), int(y2 * scale1)
                cv2.rectangle(disp1, (sx2, sy2), (sx2 + int(block_size * scale1), sy2 + int(block_size * scale1)), (0, 255, 0), 3)
                # 第二张图也画相同的框（因为是同一张图）
                cv2.rectangle(disp2, (sx1, sy1), (sx1 + int(block_size * scale1), sy1 + int(block_size * scale1)), (0, 0, 255), 3)
                cv2.rectangle(disp2, (sx2, sy2), (sx2 + int(block_size * scale1), sy2 + int(block_size * scale1)), (0, 255, 0), 3)
        
        # 2. 子图/裁剪：找子图在原图中的位置
        elif "子图" in match.match_type or "裁剪" in match.match_type:
            subimage_pos = _parse_subimage_coords(match.details)
            if subimage_pos:
                # 使用模板匹配找精确位置
                template_region = _find_template_region(gray1, gray2)
                if template_region:
                    x, y, w, h = template_region
                    # 确定哪张是大图
                    if img1.shape[0] * img1.shape[1] >= img2.shape[0] * img2.shape[1]:
                        sx, sy = int(x * scale1), int(y * scale1)
                        sw, sh = int(w * scale1), int(h * scale1)
                        cv2.rectangle(disp1, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 3)
                        cv2.rectangle(disp2, (5, 5), (disp2.shape[1]-5, disp2.shape[0]-5), (0, 0, 255), 3)
                    else:
                        sx, sy = int(x * scale2), int(y * scale2)
                        sw, sh = int(w * scale2), int(h * scale2)
                        cv2.rectangle(disp1, (5, 5), (disp1.shape[1]-5, disp1.shape[0]-5), (0, 0, 255), 3)
                        cv2.rectangle(disp2, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 3)
        
        # 3. 其他类型：使用特征匹配或差异图
        else:
            # 先尝试特征匹配
            region = _find_matching_region(gray1, gray2, match.match_type)
            
            if region:
                x1, y1, w1, h1, x2, y2, w2, h2 = region
                sx1, sy1 = int(x1 * scale1), int(y1 * scale1)
                sw1, sh1 = int(w1 * scale1), int(h1 * scale1)
                cv2.rectangle(disp1, (sx1, sy1), (sx1+sw1, sy1+sh1), (0, 0, 255), 3)

                sx2, sy2 = int(x2 * scale2), int(y2 * scale2)
                sw2, sh2 = int(w2 * scale2), int(h2 * scale2)
                cv2.rectangle(disp2, (sx2, sy2), (sx2+sw2, sy2+sh2), (0, 0, 255), 3)
            else:
                # 如果特征匹配失败，尝试差异图方法
                diff_region = _find_difference_region(gray1, gray2)
                if diff_region:
                    x, y, w, h, ref_w, ref_h = diff_region
                    # 将坐标转换到显示图上
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
                    # 最后的后备方案：画整图框
                    cv2.rectangle(disp1, (5, 5), (disp1.shape[1]-5, disp1.shape[0]-5), (0, 0, 255), 3)
                    cv2.rectangle(disp2, (5, 5), (disp2.shape[1]-5, disp2.shape[0]-5), (0, 0, 255), 3)

        # ---- 布局参数 ----
        header_h = 80      # 顶部来源信息区高度
        gap = 15           # 两图之间间距
        footer_h = 70      # 底部相似度信息区高度

        img_w1 = disp1.shape[1]
        img_w2 = disp2.shape[1]
        total_w = img_w1 + gap + img_w2
        total_h = header_h + target_h + footer_h

        # 创建画布 (深灰色背景)
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)

        # 放置图片
        canvas[header_h:header_h+target_h, :img_w1] = disp1
        canvas[header_h:header_h+target_h, img_w1+gap:] = disp2

        # ---- 绘制顶部来源信息 ----
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_small = cv2.FONT_HERSHEY_PLAIN

        # 左侧来源
        x1_base = 10
        # 组名用大字醒目显示
        group_color = (0, 200, 255) if src1["group"] != src2["group"] else (100, 255, 100)
        cv2.putText(canvas, src1["group"], (x1_base, 30), font, 1.0, group_color, 2)
        # 子文件夹
        cv2.putText(canvas, src1["subfolder"], (x1_base, 55), font, 0.65, (200, 200, 200), 1)
        # 文件名 + 通道
        file_text = src1["filename"]
        if src1["channel"]:
            file_text += f"  [{src1['channel']}]"
        cv2.putText(canvas, file_text, (x1_base, 73), font, 0.5, (160, 160, 160), 1)

        # 右侧来源
        x2_base = img_w1 + gap + 10
        cv2.putText(canvas, src2["group"], (x2_base, 30), font, 1.0, group_color, 2)
        cv2.putText(canvas, src2["subfolder"], (x2_base, 55), font, 0.65, (200, 200, 200), 1)
        file_text2 = src2["filename"]
        if src2["channel"]:
            file_text2 += f"  [{src2['channel']}]"
        cv2.putText(canvas, file_text2, (x2_base, 73), font, 0.5, (160, 160, 160), 1)

        # 中间画一个 "VS" 或相似度箭头
        mid_x = img_w1 + gap // 2
        cv2.putText(canvas, "VS", (mid_x - 15, 40), font, 0.7, (0, 0, 255), 2)

        # ---- 分割线 ----
        cv2.line(canvas, (0, header_h-2), (total_w, header_h-2), (80, 80, 80), 1)
        cv2.line(canvas, (0, header_h+target_h), (total_w, header_h+target_h), (80, 80, 80), 1)

        # ---- 绘制底部信息 ----
        footer_y = header_h + target_h + 5

        # 相似度
        sim_text = f"Similarity: {match.similarity*100:.1f}%"
        sim_color = (0, 0, 255) if match.severity == "critical" else (0, 140, 255) if match.severity == "high" else (0, 255, 255)
        cv2.putText(canvas, sim_text, (10, footer_y + 25), font, 0.7, sim_color, 2)

        # 检测类型
        type_text = f"Type: {match.match_type}"
        cv2.putText(canvas, type_text, (10, footer_y + 50), font, 0.5, (180, 180, 180), 1)

        # 严重程度标签
        sev_labels = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
        sev_colors = {"critical": (0, 0, 200), "high": (0, 100, 200), "medium": (0, 180, 180), "low": (0, 150, 0)}
        sev_label = sev_labels.get(match.severity, "UNKNOWN")
        sev_color = sev_colors.get(match.severity, (100, 100, 100))
        # 画一个标签框
        (tw, th), _ = cv2.getTextSize(sev_label, font, 0.55, 1)
        label_x = total_w - tw - 20
        cv2.rectangle(canvas, (label_x-5, footer_y+10), (label_x+tw+10, footer_y+10+th+10), sev_color, -1)
        cv2.putText(canvas, sev_label, (label_x, footer_y+10+th+5), font, 0.55, (255, 255, 255), 1)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        idx = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')]) + 1
        sev_tag = match.severity.upper()[:3]
        # 文件名加上来源组信息
        g1 = src1["group"] or "UNK"
        g2 = src2["group"] or "UNK"
        filename = f"{idx:03d}_{sev_tag}_{match.similarity*100:.0f}pct_{g1}_vs_{g2}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return output_path

    except Exception as e:
        return None


def _create_visualization_pil(match: DuplicateMatch, output_dir: str) -> Optional[str]:
    """用PIL创建简化版可视化（无OpenCV时的后备方案）"""
    try:
        from PIL import ImageDraw, ImageFont

        img1 = Image.open(match.image1).convert('RGB')
        img2 = Image.open(match.image2).convert('RGB')

        # 提取来源信息
        src1 = _extract_source_info(match.image1)
        src2 = _extract_source_info(match.image2)

        # 统一高度
        target_h = 450
        scale1 = target_h / img1.size[1]
        scale2 = target_h / img2.size[1]
        new_w1 = int(img1.size[0] * scale1)
        new_w2 = int(img2.size[0] * scale2)

        img1_disp = img1.resize((new_w1, target_h))
        img2_disp = img2.resize((new_w2, target_h))

        # 画红框
        draw1 = ImageDraw.Draw(img1_disp)
        draw2 = ImageDraw.Draw(img2_disp)
        draw1.rectangle([5, 5, new_w1-5, target_h-5], outline='red', width=3)
        draw2.rectangle([5, 5, new_w2-5, target_h-5], outline='red', width=3)

        # 布局
        header_h = 80
        gap = 15
        footer_h = 70
        total_w = new_w1 + gap + new_w2
        total_h = header_h + target_h + footer_h

        canvas = Image.new('RGB', (total_w, total_h), (40, 40, 40))
        canvas.paste(img1_disp, (0, header_h))
        canvas.paste(img2_disp, (new_w1 + gap, header_h))

        draw = ImageDraw.Draw(canvas)
        try:
            font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except:
            font_big = ImageFont.load_default()
            font_med = font_big
            font_small = font_big

        # 顶部来源信息
        group_color = (0, 200, 255) if src1["group"] != src2["group"] else (100, 255, 100)

        # 左侧
        draw.text((10, 5), src1["group"], fill=group_color, font=font_big)
        draw.text((10, 35), src1["subfolder"], fill=(200, 200, 200), font=font_med)
        file_text1 = src1["filename"]
        if src1["channel"]:
            file_text1 += f"  [{src1['channel']}]"
        draw.text((10, 58), file_text1, fill=(160, 160, 160), font=font_small)

        # 右侧
        x2 = new_w1 + gap + 10
        draw.text((x2, 5), src2["group"], fill=group_color, font=font_big)
        draw.text((x2, 35), src2["subfolder"], fill=(200, 200, 200), font=font_med)
        file_text2 = src2["filename"]
        if src2["channel"]:
            file_text2 += f"  [{src2['channel']}]"
        draw.text((x2, 58), file_text2, fill=(160, 160, 160), font=font_small)

        # VS
        mid_x = new_w1 + gap // 2
        draw.text((mid_x - 10, 10), "VS", fill=(0, 0, 255), font=font_med)

        # 底部信息
        footer_y = header_h + target_h + 5
        sim_color = (0, 0, 255) if match.severity == "critical" else (0, 140, 255)
        draw.text((10, footer_y + 10), f"Similarity: {match.similarity*100:.1f}%", fill=sim_color, font=font_med)
        draw.text((10, footer_y + 35), f"Type: {match.match_type}", fill=(180, 180, 180), font=font_small)

        # 严重程度标签
        sev_labels = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM"}
        sev_colors = {"critical": (200, 0, 0), "high": (200, 100, 0), "medium": (180, 180, 0)}
        sev_label = sev_labels.get(match.severity, "UNKNOWN")
        sev_color = sev_colors.get(match.severity, (100, 100, 100))
        tw = font_med.getlength(sev_label)
        label_x = total_w - int(tw) - 20
        draw.rectangle([label_x-5, footer_y+5, label_x+int(tw)+10, footer_y+30], fill=sev_color)
        draw.text((label_x, footer_y+7), sev_label, fill=(255, 255, 255), font=font_med)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        idx = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')]) + 1
        sev_tag = match.severity.upper()[:3]
        g1 = src1["group"] or "UNK"
        g2 = src2["group"] or "UNK"
        filename = f"{idx:03d}_{sev_tag}_{match.similarity*100:.0f}pct_{g1}_vs_{g2}.jpg"
        output_path = os.path.join(output_dir, filename)
        canvas.save(output_path, quality=90)

        return output_path
    except:
        return None


# ============================================================
# 主扫描流程
# ============================================================

def scan_directory(root_dir: str, hash_size: int = DEFAULT_HASH_SIZE,
                   workers: int = DEFAULT_WORKERS) -> list[ImageInfo]:
    """递归扫描目录，加载所有图片（排除visualization目录）"""
    root = Path(root_dir)
    image_paths = []

    # 需要排除的目录
    exclude_dirs = {'visualization', 'venv', '.git', '__pycache__', 'node_modules'}

    print(f"\n[1/3] 扫描目录: {root_dir}")
    for ext in IMAGE_EXTENSIONS:
        for p in root.rglob(f"*{ext}"):
            # 检查路径中是否包含需要排除的目录
            parts = p.relative_to(root).parts
            if not any(part in exclude_dirs for part in parts):
                image_paths.append(p)
        for p in root.rglob(f"*{ext.upper()}"):
            parts = p.relative_to(root).parts
            if not any(part in exclude_dirs for part in parts):
                image_paths.append(p)

    # 去重
    image_paths = sorted(set(str(p) for p in image_paths))
    print(f"  找到 {len(image_paths)} 张图片\n")

    if not image_paths:
        print("  未找到任何图片文件！")
        return []

    print(f"[2/3] 加载图片并计算特征...")
    images = []
    failed = 0

    for i, path in enumerate(image_paths):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  进度: {i+1}/{len(image_paths)}")
        info = load_image_info(path, hash_size)
        if info:
            images.append(info)
        else:
            failed += 1

    print(f"  成功加载: {len(images)}, 失败: {failed}\n")
    return images


def _hamming_distance(h1, h2):
    """计算两个哈希之间的汉明距离"""
    return h1 - h2


def find_duplicates(images: list[ImageInfo],
                    pthresh: int = DEFAULT_PHASH_THRESHOLD,
                    sthresh: float = DEFAULT_SSIM_THRESHOLD,
                    hthresh: float = DEFAULT_HIST_THRESHOLD,
                    check_rotations: bool = True,
                    check_subimages: bool = True,
                    check_edges: bool = True,
                    check_internal: bool = True) -> list[DuplicateMatch]:
    """执行所有检测（优化版：哈希分桶减少比较次数）"""
    matches = []
    n = len(images)

    print(f"[3/3] 执行图片查重检测 (共 {n} 张)...")

    # ---- 阶段1: MD5精确去重 ----
    md5_groups = defaultdict(list)
    for img in images:
        md5_groups[img.md5].append(img)

    exact_pairs = set()
    for md5, group in md5_groups.items():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    m = check_exact_duplicate(group[i], group[j])
                    if m:
                        matches.append(m)
                        exact_pairs.add((group[i].path, group[j].path))

    print(f"  [✓] 阶段1 完全相同: {len(matches)} 对")

    # ---- 阶段2: 哈希分桶找候选对 ----
    # 用pHash分桶：将哈希划分为多个子段，只要有一个子段匹配就作为候选
    print(f"  [·] 阶段2 哈希分桶找候选对...")

    # 直接用pHash的汉明距离筛选候选
    candidate_pairs = set()

    # 将哈希转为整数方便比较
    def hash_to_int(h):
        return int(str(h), 16)

    hash_list = []
    for idx, img in enumerate(images):
        h = hash_to_int(img.phash)
        hash_list.append((h, idx))

    # 使用多段哈希索引加速
    # 将64位哈希分成4段，每段16位，用作桶的key
    n_segments = 4
    segment_size = 64 // n_segments
    segment_masks = [(1 << segment_size) - 1 for _ in range(n_segments)]

    # 对每个段建立倒排索引
    for seg_idx in range(n_segments):
        buckets = defaultdict(list)
        mask = segment_masks[seg_idx]
        shift = seg_idx * segment_size

        for h, idx in hash_list:
            key = (h >> shift) & mask
            buckets[key].append(idx)

        # 同一桶内的图片可能是相似的
        for key, indices in buckets.items():
            if len(indices) > 1 and len(indices) < 50:  # 桶太大说明不具区分度
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        pair = (min(indices[i], indices[j]), max(indices[i], indices[j]))
                        candidate_pairs.add(pair)

    # 也用dHash分桶补充
    for seg_idx in range(n_segments):
        buckets = defaultdict(list)
        mask = segment_masks[seg_idx]
        shift = seg_idx * segment_size

        for idx, img in enumerate(images):
            h = hash_to_int(img.dhash)
            key = (h >> shift) & mask
            buckets[key].append(idx)

        for key, indices in buckets.items():
            if len(indices) > 1 and len(indices) < 50:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        pair = (min(indices[i], indices[j]), max(indices[i], indices[j]))
                        candidate_pairs.add(pair)

    # 过滤掉已经是完全相同的对，以及同实验不同通道的对
    candidate_pairs = {
        (i, j) for i, j in candidate_pairs
        if (images[i].path, images[j].path) not in exact_pairs and
           (images[j].path, images[i].path) not in exact_pairs and
           not _is_same_experiment_channel(images[i].path, images[j].path)
    }

    print(f"  [✓] 候选对数量: {len(candidate_pairs)} (vs 全量 {n*(n-1)//2})")

    # ---- 阶段3: 对候选对做详细检测 ----
    print(f"  [·] 阶段3 详细检测候选对...")

    # 用pHash距离做第一轮精确过滤
    final_candidates = []
    for i, j in candidate_pairs:
        p_dist = _hamming_distance(images[i].phash, images[j].phash)
        d_dist = _hamming_distance(images[i].dhash, images[j].dhash)
        min_dist = min(p_dist, d_dist)

        if min_dist <= pthresh + 3:  # 留一些余量给旋转等检测
            final_candidates.append((i, j, min_dist))

    # 按距离排序，优先处理最相似的
    final_candidates.sort(key=lambda x: x[2])

    print(f"  [✓] 需要详细检测: {len(final_candidates)} 对")

    # 预加载所有图片的pHash用于旋转检测
    # （只对距离较近的候选对做旋转检测）

    for idx, (i, j, dist) in enumerate(final_candidates):
        if (idx + 1) % 200 == 0:
            print(f"    进度: {idx+1}/{len(final_candidates)}")

        img1, img2 = images[i], images[j]

        # pHash 检测
        m = check_phash_similarity(img1, img2, pthresh)
        if m:
            matches.append(m)
            # 如果非常相似(pHash距离<=2)，直接跳过后续检测
            if dist <= 2:
                continue

        # dHash 检测
        m = check_dhash_similarity(img1, img2, pthresh)
        if m and not any(
            mm.match_type == "感知哈希相似(pHash)" and
            ((mm.image1 == img1.path and mm.image2 == img2.path) or
             (mm.image1 == img2.path and mm.image2 == img1.path))
            for mm in matches
        ):
            matches.append(m)

        # SSIM（只对哈希距离中等的做，太近的已经检测到了，太远的没必要）
        if 3 <= dist <= pthresh + 5:
            m = check_ssim_similarity(img1, img2, sthresh)
            if m:
                matches.append(m)

        # 直方图（检测曝光调整）
        if dist <= pthresh + 3:
            m = check_histogram_similarity(img1, img2, hthresh)
            if m:
                matches.append(m)

            m = check_brightness_contrast(img1, img2)
            if m:
                matches.append(m)

        # 旋转/翻转（只对哈希距离较远但仍有可能相似的做）
        if check_rotations and pthresh < dist <= pthresh + 8:
            rot_matches = check_rotation_flip(img1, img2, pthresh)
            matches.extend(rot_matches)

        # 缩放
        if 3 <= dist <= pthresh + 5:
            m = check_scale(img1, img2, sthresh)
            if m:
                matches.append(m)

        # ORB特征（只对哈希不太相似但可能有变换的做）
        if pthresh < dist <= pthresh + 10:
            m = check_orb_features(img1, img2)
            if m:
                matches.append(m)

        # 子图检测（仅对尺寸差异明显的）
        if check_subimages and dist <= pthresh + 3:
            area_ratio = (img1.size[0] * img1.size[1]) / (img2.size[0] * img2.size[1] + 1)
            if 1.5 < area_ratio < 10 or 0.1 < area_ratio < 0.67:
                m = check_subimage(img1, img2)
                if m:
                    matches.append(m)

        # 边缘重叠
        if check_edges and dist <= pthresh + 3:
            m = check_edge_overlap(img1, img2)
            if m:
                matches.append(m)

    print(f"  [✓] 阶段3 详细检测完成")

    # ---- 阶段4: 内部区域复制检测 ----
    if check_internal:
        print(f"  [·] 阶段4 检测图片内部区域复制...")
        internal_count = 0
        for img in images:
            internal = check_internal_duplicate(img.path)
            matches.extend(internal)
            internal_count += len(internal)
        print(f"  [✓] 内部复制检测完成: 发现 {internal_count} 处")

    return matches


# ============================================================
# 结果去重与排序
# ============================================================

def deduplicate_matches(matches: list[DuplicateMatch]) -> list[DuplicateMatch]:
    """
    去除重复的匹配结果
    对于同一对图片，只保留相似度最高的匹配结果
    """
    # 用字典存储每对图片的最佳匹配
    best_matches = {}

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    for m in matches:
        # 跳过自己和自己配对的情况
        if m.image1 == m.image2:
            continue
        
        # 规范化key（确保img1,img2顺序一致）
        pair_key = tuple(sorted([m.image1, m.image2]))

        # 如果这对图片还没有记录，或者当前匹配的相似度更高，则更新
        if pair_key not in best_matches or m.similarity > best_matches[pair_key].similarity:
            best_matches[pair_key] = m

    # 转换为列表并按严重程度排序
    unique = list(best_matches.values())
    unique.sort(key=lambda m: (severity_order.get(m.severity, 9), -m.similarity))
    return unique


# ============================================================
# 报告生成
# ============================================================

def generate_html_report(matches: list[DuplicateMatch], output_path: str,
                         total_images: int, scan_dir: str):
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
<title>科研图片查重报告</title>
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
    <h1>🔬 科研图片查重报告</h1>

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

    # 按类型统计
    type_counts = defaultdict(int)
    for m in matches:
        type_counts[m.match_type] += 1

    html += """
    <div class="summary">
        <h2>📈 问题类型统计</h2>
        <table>
            <tr><th>问题类型</th><th>数量</th><th>占比</th></tr>
"""
    for mtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(matches) if matches else 0
        html += f"            <tr><td>{mtype}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n"

    html += f"""
        </table>
    </div>

    <div class="footer">
        <p>由 科研图片查重工具 生成 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  HTML报告已生成: {output_path}")


def generate_csv_report(matches: list[DuplicateMatch], output_path: str):
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


def generate_json_report(matches: list[DuplicateMatch], output_path: str,
                         total_images: int, scan_dir: str):
    """生成JSON报告"""
    report = {
        "scan_info": {
            "directory": scan_dir,
            "total_images": total_images,
            "total_matches": len(matches),
            "scan_time": datetime.now().isoformat(),
        },
        "matches": [
            {
                "index": i,
                "match_type": m.match_type,
                "severity": m.severity,
                "similarity": m.similarity,
                "image1": m.image1,
                "image2": m.image2,
                "details": m.details,
            }
            for i, m in enumerate(matches, 1)
        ]
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  JSON报告已生成: {output_path}")


# ============================================================
# 命令行入口
# ============================================================

def main():
    """
    科研图片查重工具 - 扫描当前目录
    双击运行即可自动查重
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="科研图片查重工具 - 检测图片重复、旋转、翻转、曝光调整、裁剪、拼接等",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  图片查重工具.exe                          # 扫描当前目录
  图片查重工具.exe D:\\proj_test\\proj_1     # 扫描指定目录
  图片查重工具.exe --threshold 3 --report report.html
        """
    )

    parser.add_argument("directory", nargs='?', default=None,
                        help="要扫描的图片目录 (默认: 当前目录)")
    parser.add_argument("--hash-size", type=int, default=DEFAULT_HASH_SIZE,
                        help=f"哈希大小 (默认: {DEFAULT_HASH_SIZE})")
    parser.add_argument("--threshold", type=int, default=DEFAULT_PHASH_THRESHOLD,
                        help=f"哈希差异阈值，越小越严格 (默认: {DEFAULT_PHASH_THRESHOLD})")
    parser.add_argument("--ssim-threshold", type=float, default=DEFAULT_SSIM_THRESHOLD,
                        help=f"SSIM阈值 (默认: {DEFAULT_SSIM_THRESHOLD})")
    parser.add_argument("--hist-threshold", type=float, default=DEFAULT_HIST_THRESHOLD,
                        help=f"直方图阈值 (默认: {DEFAULT_HIST_THRESHOLD})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"并行工作进程数 (默认: {DEFAULT_WORKERS})")
    parser.add_argument("--report", type=str, default=None,
                        help="输出报告文件路径 (支持 .html / .csv / .json)")
    parser.add_argument("--no-rotation", action="store_true",
                        help="跳过旋转/翻转检测")
    parser.add_argument("--no-subimage", action="store_true",
                        help="跳过子图/裁剪检测")
    parser.add_argument("--no-edge", action="store_true",
                        help="跳过边缘重叠检测")
    parser.add_argument("--no-internal", action="store_true",
                        help="跳过内部区域复制检测")
    parser.add_argument("--no-orb", action="store_true",
                        help="跳过ORB特征匹配")

    args = parser.parse_args()

    # 如果没有指定目录，使用当前目录
    if args.directory is None:
        args.directory = os.getcwd()
    
    if not os.path.isdir(args.directory):
        print(f"错误: 目录不存在: {args.directory}")
        sys.exit(1)

    print("=" * 60)
    print("  🔬 科研图片查重工具")
    print("=" * 60)
    print(f"  扫描目录: {args.directory}")
    print(f"  哈希阈值: {args.threshold}")
    print(f"  SSIM阈值: {args.ssim_threshold}")
    print(f"  直方图阈值: {args.hist_threshold}")

    # 扫描
    images = scan_directory(args.directory, args.hash_size, args.workers)

    if len(images) < 2:
        print("\n图片数量不足，无法进行查重。")
        sys.exit(0)

    # 检测
    matches = find_duplicates(
        images,
        pthresh=args.threshold,
        sthresh=args.ssim_threshold,
        hthresh=args.hist_threshold,
        check_rotations=not args.no_rotation,
        check_subimages=not args.no_subimage,
        check_edges=not args.no_edge,
        check_internal=not args.no_internal,
    )

    # 去重排序
    matches = deduplicate_matches(matches)

    # ---- 生成可视化对比图 ----
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
        # 按严重程度分组输出
        by_severity = defaultdict(list)
        for m in matches:
            by_severity[m.severity].append(m)

        severity_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        severity_names = {"critical": "严重", "high": "高风险", "medium": "中等", "low": "低风险"}

        for sev in ["critical", "high", "medium", "low"]:
            if sev in by_severity:
                print(f"\n  {severity_icons[sev]} {severity_names[sev]} ({len(by_severity[sev])} 对):")
                for m in by_severity[sev][:10]:  # 每类最多显示10条
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

    # 根据扩展名生成对应格式
    if report_path.endswith('.csv'):
        generate_csv_report(matches, report_path)
    elif report_path.endswith('.json'):
        generate_json_report(matches, report_path, len(images), args.directory)
    else:
        if not report_path.endswith('.html'):
            report_path += '.html'
        generate_html_report(matches, report_path, len(images), args.directory)

    # 总是生成一份CSV
    csv_path = report_path.rsplit('.', 1)[0] + '.csv'
    if csv_path != report_path:
        generate_csv_report(matches, csv_path)

    print(f"\n✅ 扫描完成！")


if __name__ == "__main__":
    main()
