import os
import time
from itertools import combinations

from core.loader import scan_images, load_image_info, ImageInfo
from core.grouping import extract_group_info, compute_pair_relation

from detectors.base import MatchResult
from detectors.exact import ExactDetector
from detectors.perceptual import PerceptualDetector
from detectors.structural import StructuralDetector
from detectors.histogram import HistogramDetector
from detectors.transform import TransformDetector
from detectors.edge_overlap import EdgeOverlapDetector
from detectors.subimage import SubimageDetector
from detectors.feature_match import FeatureMatchDetector


def _build_fast_dets(cfg: dict):
    return [
        ExactDetector(cfg),
        PerceptualDetector(cfg),
        StructuralDetector(cfg),
        TransformDetector(cfg),
        EdgeOverlapDetector(cfg),
    ]


def _pair_key(p1: str, p2: str) -> tuple:
    return tuple(sorted([p1, p2]))


def run_detection(directory: str, config: dict
                  ) -> tuple[list[ImageInfo], list[MatchResult], list[MatchResult]]:
    scan_cfg = config.get("scan", {})
    det_cfg = config.get("detection", {})
    grouping_cfg = config.get("grouping", {})

    extensions = set(scan_cfg.get("extensions", ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']))
    hash_size = det_cfg.get("phash_hash_size", 16)
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
        info = load_image_info(f, directory, hash_size=hash_size)
        if info is not None:
            gi = extract_group_info(info.path, grouping_patterns)
            info.group_info = gi
            infos.append(info)
        if (i + 1) % 100 == 0:
            print(f"  Loading: {i + 1}/{n}", end='\r')
    print(f"  Loading: {n}/{n}")
    print(f"  Loaded {len(infos)} images in {time.time() - t0:.1f}s")

    total_pairs = len(infos) * (len(infos) - 1) // 2
    print(f"Comparing {total_pairs} pairs (fast detectors)...")
    t0 = time.time()

    fast_dets = _build_fast_dets(det_cfg)
    primary_matches = []
    report_interval = max(1, total_pairs // 20)

    for idx, (info_a, info_b) in enumerate(combinations(infos, 2)):
        relation = compute_pair_relation(info_a.group_info, info_b.group_info)
        for det in fast_dets:
            try:
                r = det.compare(info_a, info_b, relation)
                if r is not None:
                    primary_matches.append(r)
            except Exception:
                continue
        if (idx + 1) % report_interval == 0 or idx == total_pairs - 1:
            pct = (idx + 1) * 100 // total_pairs
            print(f"  Progress: {pct}% ({idx + 1:,}/{total_pairs:,})", end='\r')
    print()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    unique = _dedup_matches(primary_matches)
    _print_summary(unique)

    sift_matches = []
    use_sift = det_cfg.get("sift_enabled", True)
    if use_sift:
        sift_det = FeatureMatchDetector(det_cfg)
        all_pairs = list(combinations(infos, 2))
        print(f"Running SIFT verification on all {len(all_pairs)} pairs...")
        t1 = time.time()
        for i, (info_a, info_b) in enumerate(all_pairs):
            relation = compute_pair_relation(info_a.group_info, info_b.group_info)
            if relation == "SAME_FOV_DIFF_CH":
                continue
            try:
                r = sift_det.compare(info_a, info_b, relation)
                if r is not None:
                    sift_matches.append(r)
            except Exception:
                continue
            if (i + 1) % 8000 == 0 or i == len(all_pairs) - 1:
                pct = (i + 1) * 100 // len(all_pairs)
                print(f"  SIFT: {pct}% ({i + 1:,}/{len(all_pairs):,})", end='\r')
        print()
        print(f"  SIFT done in {time.time() - t1:.1f}s")
        print(f"  Found {len(sift_matches)} SIFT matches")

    if sift_matches:
        unique.extend(sift_matches)
        unique.sort(key=lambda m: (_sev_rank(m.severity), -m.similarity))
        unique = _dedup_matches(unique)

    edge_matches = [m for m in unique if "边缘" in m.match_type]
    sift_matches_only = [m for m in unique if "特征匹配" in m.match_type]
    other_matches = [m for m in unique if m not in edge_matches and m not in sift_matches_only]

    all_reported = other_matches + sift_matches_only + edge_matches

    print(f"\nTotal: {len(all_reported)} matches "
          f"(hash/ssim={len(other_matches)}, "
          f"sift={len(sift_matches_only)}, "
          f"edge={len(edge_matches)})")
    return infos, all_reported, sift_matches_only


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
    from collections import Counter
    sev_c = Counter(m.severity for m in matches)
    type_c = Counter(m.match_type for m in matches)
    cross_c = sum(1 for m in matches if m.is_cross_channel)
    if matches:
        print(f"  Severity: critical={sev_c.get('critical',0)} "
              f"high={sev_c.get('high',0)} medium={sev_c.get('medium',0)}")
        top = type_c.most_common(5)
        print(f"  Types: {', '.join(f'{t}={c}' for t,c in top)}")
        print(f"  Cross-channel: {cross_c}")
