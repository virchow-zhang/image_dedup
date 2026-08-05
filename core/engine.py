import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.loader import scan_images, load_image_info, ImageInfo
from core.grouping import extract_group_info, compute_pair_relation
from core.index import HashIndex, imagehash_to_bits, rotate_hash_bits

from detectors.base import MatchResult
from detectors.exact import ExactDetector
from detectors.perceptual import PerceptualDetector
from detectors.structural import StructuralDetector
from detectors.histogram import HistogramDetector
from detectors.transform import TransformDetector
from detectors.edge_overlap import EdgeOverlapDetector
from detectors.subimage import SubimageDetector
from detectors.feature_match import FeatureMatchDetector
from detectors.cmfd import CmfdDetector

_SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# 主证据优先级: 几何定位 > 精确 > 局部 > 哈希
_TYPE_PRIORITY = {
    "特征匹配": 0, "完全相同": 1, "内部区域复制": 2, "疑似子图": 3,
    "感知哈希": 4, "结构相似": 5, "疑似旋转": 6, "疑似翻转": 6,
    "疑似缩放": 7, "边缘": 8, "直方图": 9, "AI嵌入": 10,
}


def _pair_key(p1: str, p2: str) -> tuple:
    if p1 == p2:
        return p1
    return tuple(sorted([p1, p2]))


def _load_all(directory: str, files: list[str], config: dict) -> list[ImageInfo]:
    scan_cfg = config.get("scan", {})
    det_cfg = config.get("detection", {})
    grouping_cfg = config.get("grouping", {})
    hash_size = det_cfg.get("phash_hash_size", 16)
    grouping_patterns = grouping_cfg.get("patterns", [])
    workers = max(1, scan_cfg.get("max_workers", 8))

    infos = []
    n = len(files)
    if n == 0:
        return infos

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(load_image_info, f, directory, hash_size) for f in files]
        for i, fut in enumerate(as_completed(futures), 1):
            info = fut.result()
            if info is not None:
                info.group_info = extract_group_info(info.path, grouping_patterns)
                infos.append(info)
            if i % 100 == 0 or i == n:
                print(f"  Loading: {i}/{n}", end="\r")
    print(f"  Loading: {n}/{n}  ({time.time() - t0:.1f}s, {len(infos)} 张成功)")
    return infos


def _exact_pairs(infos: list[ImageInfo]) -> tuple[list[MatchResult], set]:
    """MD5 精确去重, 返回 (匹配结果, 精确重复对集合)。"""
    groups = defaultdict(list)
    for info in infos:
        groups[info.md5].append(info)

    matches = []
    exact_pairs = set()
    det = ExactDetector({})
    for md5, group in groups.items():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    r = det.compare(group[i], group[j])
                    if r:
                        matches.append(r)
                        exact_pairs.add(_pair_key(group[i].path, group[j].path))
    return matches, exact_pairs


def _build_hash_sources(infos: list[ImageInfo], det_cfg: dict, with_rot: bool):
    """构建各哈希源的位矩阵, 返回 [(source_name, ids, bits, is_variant)]。

    ids 与 bits 行一一对应, 值域为 infos 的下标。
    任一行为 None 时整个源丢弃, 保证对齐。
    """
    import numpy as np
    from PIL import Image
    import imagehash

    valid = [i for i, info in enumerate(infos) if info.phash is not None]
    sources = []

    def bits_of(fn):
        rows = [fn(infos[i]) for i in valid]
        if any(r is None for r in rows):
            return None
        return np.stack(rows)

    ph = bits_of(lambda info: imagehash_to_bits(info.phash))
    dh = bits_of(lambda info: imagehash_to_bits(info.dhash))
    wh = bits_of(lambda info: imagehash_to_bits(info.whash) if info.whash else None)

    if ph is not None:
        sources.append(("phash", valid, ph, False, None))
    if dh is not None:
        sources.append(("dhash", valid, dh, False, None))
    if wh is not None:
        sources.append(("whash", valid, wh, False, None))

    if with_rot:
        hash_fn = lambda im: imagehash.phash(im, hash_size=det_cfg.get("phash_hash_size", 16))
        for kind in ("rot90", "rot180", "rot270", "flipH", "flipV",
                     "rot15", "rot345"):
            rows = []
            for i in valid:
                try:
                    rows.append(rotate_hash_bits(infos[i].gray_256, kind, hash_fn))
                except Exception:
                    rows.append(None)
            if any(r is None for r in rows):
                continue
            sources.append((f"phash_{kind}", valid, np.stack(rows), True, ph))
    return sources


def _candidate_pairs(infos: list[ImageInfo], det_cfg: dict, cfg: dict):
    """返回候选对集合 (infos 下标对)。"""
    index_cfg = cfg.get("index", {})
    threshold = det_cfg.get("phash_threshold", 3)
    # 候选阈值可以比验证阈值宽: 用更宽的网召回 (裁剪/自由旋转/局部粘贴
    # 会让哈希距离显著变大), 具体判定交给验证器。MIH 保证需 < 16。
    cand_threshold = min(15, max(threshold, index_cfg.get("candidate_threshold", threshold + 10)))
    n_tables = index_cfg.get("mih_tables", 16)
    with_rot = index_cfg.get("rotation_variants", True)
    if det_cfg.get("rotation_enabled", True) is False:
        with_rot = False

    pairs = set()
    t0 = time.time()

    # 原生 C++ 内核优先 (可插拔, 失败自动回退 Python MIH)
    used_native = False
    if index_cfg.get("native_core", True):
        from core.native import core_available, native_hash_candidates, has_unsupported_formats
        if core_available():
            root_dir = os.path.dirname(infos[0].path)
            if not has_unsupported_formats(root_dir):
                native_pairs = native_hash_candidates(root_dir, cand_threshold)
                if native_pairs is not None:
                    path_set = {i.path for i in infos}
                    path_idx = {i.path: k for k, i in enumerate(infos)}
                    for a, b in native_pairs:
                        if a in path_idx and b in path_idx:
                            pairs.add((path_idx[a], path_idx[b]))
                    used_native = True
                    print(f"  [索引] 原生内核 dedup_core.exe: "
                          f"{len(pairs)} 哈希候选对")

    if not used_native:
        sources = _build_hash_sources(infos, det_cfg, with_rot)
        for name, ids, bits, is_variant, plain_bits in sources:
            idx = HashIndex(n_tables=n_tables)
            if is_variant:
                # 变换哈希与前 N 行, 原始哈希放后 N 行, 同一索引:
                # 命中即 phash(变换(A)) 与 phash(B) 的距离 <= threshold
                n = len(ids)
                idx.add(ids, bits)
                idx.add(ids, plain_bits)
                found = idx.candidates(cand_threshold)
                full = list(ids) + list(ids)
                mapped = set()
                for i, j in found:
                    a, b = full[i], full[j]
                    if a != b:
                        mapped.add((a, b) if a < b else (b, a))
            else:
                idx.add(ids, bits)
                found = idx.candidates(cand_threshold)
                mapped = {(ids[i], ids[j]) for i, j in found}
            pairs |= mapped
            print(f"  [索引] {name}: {idx.size} 行 -> {len(mapped)} 候选对")

    # BOW 层: ORB 词袋倒排索引, 覆盖自由旋转/裁剪/拼接等局部复用
    if index_cfg.get("bow_enabled", True):
        t1 = time.time()
        from core.bow import candidate_pairs as bow_pairs
        bow = bow_pairs(infos, min_shared=index_cfg.get("bow_min_shared", 8))
        pairs |= bow
        print(f"  [索引] bow: 共享ORB特征>=8 -> {len(bow)} 候选对 "
              f"({time.time() - t1:.2f}s)")

    print(f"  [索引] 候选构建耗时 {time.time() - t0:.2f}s, 共 {len(pairs)} 对")

    # 过滤精确重复对 + 同 FOV 不同通道对
    keep = set()
    for i, j in pairs:
        r = compute_pair_relation(infos[i].group_info, infos[j].group_info)
        if r == "SAME_FOV_DIFF_CH":
            continue
        keep.add((i, j))
    return keep


def _verify_pairs(infos: list[ImageInfo], pairs: set, det_cfg: dict, cfg: dict,
                  workers: int = 8):
    """对候选对执行全部验证器 (并行, OpenCV 释放 GIL)。"""
    dets = [
        PerceptualDetector(det_cfg),
        StructuralDetector(det_cfg),
        TransformDetector(det_cfg),
        EdgeOverlapDetector(det_cfg),
        FeatureMatchDetector(det_cfg),
    ]
    if det_cfg.get("subimage_enabled", True):
        dets.append(SubimageDetector(det_cfg))
    if det_cfg.get("hist_enabled", False):
        dets.append(HistogramDetector(det_cfg))

    def work(item):
        i, j = item
        relation = compute_pair_relation(infos[i].group_info, infos[j].group_info)
        out = []
        for det in dets:
            try:
                r = det.compare(infos[i], infos[j], relation)
                if r is not None:
                    out.append(r)
            except Exception:
                continue
        return out

    matches = []
    pair_list = sorted(pairs)
    n = len(pair_list)
    t0 = time.time()
    n_workers = max(1, workers) if n > 200 else 1
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for k, res in enumerate(ex.map(work, pair_list), 1):
                matches.extend(res)
                if k % 2000 == 0 or k == n:
                    pct = k * 100 // max(1, n)
                    print(f"  [验证] {pct}% ({k:,}/{n:,}) 候选对, "
                          f"已发现 {len(matches)} 条匹配", end="\r")
    else:
        for k, (i, j) in enumerate(pair_list, 1):
            matches.extend(work((i, j)))
            if k % 2000 == 0 or k == n:
                pct = k * 100 // max(1, n)
                print(f"  [验证] {pct}% ({k:,}/{n:,}) 候选对, "
                      f"已发现 {len(matches)} 条匹配", end="\r")
    print()
    print(f"  [验证] 候选对验证耗时 {time.time() - t0:.2f}s "
          f"({n_workers} 线程)")
    return matches


def _run_cmfd(infos: list[ImageInfo], det_cfg: dict, cfg: dict):
    """单图内部区域复制检测。"""
    if not det_cfg.get("cmfd_enabled", True):
        return []
    det = CmfdDetector(det_cfg)
    matches = []
    t0 = time.time()
    for k, info in enumerate(infos, 1):
        r = det.detect(info)
        if r is not None:
            matches.append(r)
        if k % 100 == 0 or k == len(infos):
            print(f"  [CMFD] {k}/{len(infos)}", end="\r")
    print()
    print(f"  [CMFD] 内部区域复制检测耗时 {time.time() - t0:.2f}s, "
          f"发现 {len(matches)} 条")
    return matches


def _run_ai_layer(infos: list[ImageInfo], det_cfg: dict, cfg: dict):
    """AI 嵌入层: 提取特征 + 余弦候选。

    嵌入相似仅用于候选召回 (召回后由 SIFT/哈希等验证器判定),
    不直接作为报告证据 —— 同类科研图像的整体嵌入余弦普遍偏高,
    绝对阈值无法可靠区分, 避免污染报告。
    """
    ai_cfg = cfg.get("ai", {})
    model_path = ai_cfg.get("model_path", "models/mobilenetv2-7.onnx")
    if not os.path.isabs(model_path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base, model_path)
    cosine_threshold = ai_cfg.get("cosine_threshold", 0.90)
    top_k = ai_cfg.get("top_k", 20)

    from detectors.embedding import EmbeddingModel, ensure_model
    path = ensure_model(model_path)
    if path is None:
        print("  [AI] 模型下载失败, 跳过 AI 检测层")
        return [], set()

    print("  [AI] 提取嵌入向量...")
    t0 = time.time()
    model = EmbeddingModel(path)
    emb = model.embed(infos)
    print(f"  [AI] 嵌入耗时 {time.time() - t0:.2f}s, 维度 {emb.shape[1]}")

    pairs = model.candidate_pairs(emb, cosine_threshold, top_k)
    print(f"  [AI] 发现 {len(pairs)} 对嵌入候选 (仅用于候选召回)")
    return [], pairs


def _pick_primary(ms: list[MatchResult]) -> MatchResult:
    def key(m):
        p = _TYPE_PRIORITY.get(m.match_type, 20)
        if m.severity == "critical":
            p -= 50
        return (_SEV_RANK.get(m.severity, 9), p, -m.similarity)
    return min(ms, key=key)


def _fuse_matches(matches: list[MatchResult]) -> list[MatchResult]:
    """按对融合: 每对保留一条主证据, 其余进入证据链。"""
    groups = defaultdict(list)
    for m in matches:
        groups[_pair_key(m.image1, m.image2)].append(m)

    out = []
    for key, ms in groups.items():
        primary = _pick_primary(ms)
        primary.evidence = [
            {"type": m.match_type, "severity": m.severity,
             "similarity": m.similarity, "details": m.details}
            for m in ms if m is not primary
        ]
        if primary.evidence:
            chain = " | ".join(
                f"{e['type']}({e['similarity']:.2f})" for e in primary.evidence[:4])
            primary.details = f"{primary.details} | 其他证据: {chain}"
        out.append(primary)

    out.sort(key=lambda m: (_SEV_RANK.get(m.severity, 9), -m.similarity))
    return out


def run_detection(directory: str, config: dict
                  ) -> tuple[list[ImageInfo], list[MatchResult], list[MatchResult]]:
    scan_cfg = config.get("scan", {})
    det_cfg = config.get("detection", {})
    ai_cfg = config.get("ai", {})

    extensions = set(scan_cfg.get(
        "extensions", ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']))
    directory = os.path.abspath(directory)

    print(f"Scanning: {directory}")
    t_total = time.time()
    files = scan_images(directory, extensions)
    n = len(files)
    print(f"  Found {n} images")

    print("Loading images and computing features...")
    infos = _load_all(directory, files, config)
    if len(infos) < 2:
        return infos, [], []

    # ---- 阶段 1: 精确重复 ----
    exact_matches, exact_pairs = _exact_pairs(infos)
    print(f"  [阶段1] 完全相同: {len(exact_matches)} 对")

    # ---- 阶段 2: 候选生成 ----
    print("Building index and generating candidates...")
    t0 = time.time()
    candidates = _candidate_pairs(infos, det_cfg, config)
    ai_matches = []
    if ai_cfg.get("enabled", False):
        m_ai, p_ai = _run_ai_layer(infos, det_cfg, config)
        ai_matches = m_ai
        candidates |= p_ai
    # 去掉精确重复对
    candidates = {p for p in candidates if _pair_key(
        infos[p[0]].path, infos[p[1]].path) not in exact_pairs}
    print(f"  [阶段2] 候选对: {len(candidates):,} "
          f"(朴素全量: {len(infos) * (len(infos) - 1) // 2:,}, "
          f"过滤率 {100 * (1 - len(candidates) / max(1, len(infos) * (len(infos) - 1) // 2)):.1f}%)")

    # ---- 阶段 3: 候选验证 ----
    print("Verifying candidates...")
    workers = max(1, scan_cfg.get("max_workers", 8))
    matches = _verify_pairs(infos, candidates, det_cfg, config, workers=workers)

    # ---- 阶段 4: 单图内部区域复制 ----
    cmfd_matches = _run_cmfd(infos, det_cfg, config)

    all_matches = exact_matches + ai_matches + matches + cmfd_matches
    fused = _fuse_matches(all_matches)

    sift_only = [m for m in fused if "特征匹配" in m.match_type]
    other = [m for m in fused if m not in sift_only]

    print(f"\nTotal: {len(fused)} matches "
          f"(exact={len(exact_matches)}, hash/ssim={len(other) - len(cmfd_matches)}, "
          f"sift={len(sift_only)}, cmfd={len(cmfd_matches)}, "
          f"ai={len(ai_matches)})")
    print(f"总耗时 {time.time() - t_total:.1f}s")
    return infos, fused, sift_only
