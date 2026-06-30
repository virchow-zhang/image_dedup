import os
import time
from itertools import combinations
from collections import Counter

from core.loader import scan_images, load_image_info, ImageInfo
from core.grouping import extract_group_info, compute_pair_relation

from detectors.base import MatchResult
from detectors.exact import ExactDetector
from detectors.perceptual import PerceptualDetector
from detectors.structural import StructuralDetector
from detectors.histogram import HistogramDetector
from detectors.transform import TransformDetector
from detectors.feature_match import FeatureMatchDetector
from detectors.edge_overlap import EdgeOverlapDetector
from detectors.subimage import SubimageDetector
from detectors.fov_overlap import FovOverlapDetector, FovOverlapMatch


def _build_fast_dets(cfg: dict):
    dets = [
        ExactDetector(cfg),
        PerceptualDetector(cfg),
        StructuralDetector(cfg),
        TransformDetector(cfg),
        EdgeOverlapDetector(cfg),
    ]
    return dets


def _pair_key(p1: str, p2: str) -> tuple:
    return tuple(sorted([p1, p2]))


def run_detection(directory: str, config: dict) -> tuple[list[ImageInfo], list[MatchResult]]:
    scan_cfg = config.get("scan", {})
    det_cfg = config.get("detection", {})
    grouping_cfg = config.get("grouping", {})

    extensions = set(scan_cfg.get("extensions", ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']))
    hash_size = det_cfg.get("phash_hash_size", 16)
    use_orb = det_cfg.get("orb_enabled", True)
    orb_max_features = det_cfg.get("orb_max_features", 2000)
    grouping_patterns = grouping_cfg.get("patterns", [])

    directory = os.path.abspath(directory)
    print(f"Scanning: {directory}")
    files = scan_images(directory, extensions)
    n = len(files)
    print(f"  Found {n} images")

    print("Loading images and computing features...")
    t0 = time.time()
    infos = []
    for i, f in enumerate(files):
        info = load_image_info(f, directory, hash_size=hash_size,
                                compute_orb=use_orb, orb_max_features=orb_max_features)
        if info is not None:
            gi = extract_group_info(info.path, grouping_patterns)
            info.group_info = gi
            infos.append(info)
        if (i + 1) % 100 == 0:
            print(f"  Loading: {i + 1}/{n}", end='\r')
    print(f"  Loading: {n}/{n}")
    print(f"  Loaded {len(infos)} images in {time.time() - t0:.1f}s")

    total_pairs = len(infos) * (len(infos) - 1) // 2
    print(f"Comparing {total_pairs} pairs (6 detectors)...")
    t0 = time.time()

    fast_dets = _build_fast_dets(det_cfg)
    all_matches = []
    report_interval = max(1, total_pairs // 20)

    for idx, (info_a, info_b) in enumerate(combinations(infos, 2)):
        relation = compute_pair_relation(info_a.group_info, info_b.group_info)
        for det in fast_dets:
            try:
                r = det.compare(info_a, info_b, relation)
                if r is not None:
                    all_matches.append(r)
            except Exception:
                continue
        if (idx + 1) % report_interval == 0 or idx == total_pairs - 1:
            pct = (idx + 1) * 100 // total_pairs
            print(f"  Progress: {pct}% ({idx + 1:,}/{total_pairs:,})", end='\r')
    print()
    elapsed = time.time() - t0
    rate = total_pairs / elapsed if elapsed > 0 else 0
    print(f"  Done in {elapsed:.1f}s ({rate:.0f} pairs/s)")

    unique = _dedup_matches(all_matches)
    unique.sort(key=lambda m: (_sev_rank(m.severity), -m.similarity))
    print(f"  Found {len(unique)} unique matches")
    _print_summary(unique)

    if use_orb and unique:
        orb_det = FeatureMatchDetector(det_cfg)
        candidates = [m for m in unique
                      if m.severity in ("critical", "high")
                      and not m.is_cross_channel
                      and m.match_type != "完全相同（MD5）"
                      and m.match_type != "特征点匹配（ORB）"]

        max_orb = min(len(candidates), 300)
        candidates = candidates[:max_orb]
        if candidates:
            print(f"Running ORB verification on {len(candidates)} candidates...")
            info_map = {info.path: info for info in infos}
            orb_matches = []
            for i, m in enumerate(candidates):
                info_a = info_map.get(m.image1)
                info_b = info_map.get(m.image2)
                if info_a is None or info_b is None:
                    continue
                relation = compute_pair_relation(info_a.group_info, info_b.group_info)
                try:
                    r = orb_det.compare(info_a, info_b, relation)
                    if r is not None:
                        orb_matches.append(r)
                except Exception:
                    continue
                if (i + 1) % 20 == 0 or i == len(candidates) - 1:
                    print(f"  ORB: {i + 1}/{len(candidates)}", end='\r')
            print()

            if orb_matches:
                print(f"  ORB confirmed {len(orb_matches)} matches")
                seen_keys = {(m.match_type,) for m in unique}
                for m in orb_matches:
                    k = (m.match_type,)
                    if k not in seen_keys:
                        unique.append(m)

                unique.sort(key=lambda m: (_sev_rank(m.severity), -m.similarity))

    fov_matches = []
    fov_cfg = det_cfg.get("fov_overlap", {})
    if fov_cfg.get("enabled", True):
        fov_det = FovOverlapDetector(det_cfg)
        same_ch_pairs = []
        info_map = {info.path: info for info in infos}
        for info_a, info_b in combinations(infos, 2):
            gi1, gi2 = info_a.group_info, info_b.group_info
            if gi1.get("channel") and gi2.get("channel") and gi1["channel"] == gi2["channel"]:
                same_ch_pairs.append((info_a, info_b))
        print(f"Running FOV overlap detection on {len(same_ch_pairs)} same-channel pairs...")
        for i, (info_a, info_b) in enumerate(same_ch_pairs):
            relation = compute_pair_relation(info_a.group_info, info_b.group_info)
            try:
                r = fov_det.compare(info_a, info_b, relation)
                if r is not None:
                    fov_matches.append(r)
            except Exception:
                continue
            if (i + 1) % 5000 == 0 or i == len(same_ch_pairs) - 1:
                pct = (i + 1) * 100 // max(len(same_ch_pairs), 1)
                print(f"  FOV overlap: {pct}% ({i + 1}/{len(same_ch_pairs)})", end='\r')
        print()
        if fov_matches:
            print(f"  Found {len(fov_matches)} FOV overlap warnings")

    print(f"\nTotal: {len(unique)} matches, {len(fov_matches)} FOV overlaps")
    _print_summary(unique)
    return infos, unique, fov_matches


def _sev_rank(sev: str) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(sev, 99)


def _dedup_matches(matches: list[MatchResult]) -> list[MatchResult]:
    seen = set()
    out = []
    for m in matches:
        k = (_pair_key(m.image1, m.image2), m.match_type)
        if k not in seen:
            seen.add(k)
            out.append(m)
    return out


def _print_summary(matches: list[MatchResult]):
    sev_c = Counter(m.severity for m in matches)
    type_c = Counter(m.match_type for m in matches)
    cross_c = sum(1 for m in matches if m.is_cross_channel)
    if matches:
        print(f"  Severity: critical={sev_c.get('critical',0)} high={sev_c.get('high',0)} medium={sev_c.get('medium',0)}")
        top_types = type_c.most_common(5)
        print(f"  Types: {', '.join(f'{t}={c}' for t,c in top_types)}")
        print(f"  Cross-channel: {cross_c}")
