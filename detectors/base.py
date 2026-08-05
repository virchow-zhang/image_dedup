from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchResult:
    image1: str
    image2: str
    match_type: str
    similarity: float
    severity: str
    details: str = ""
    is_cross_channel: bool = False
    match_points1: list = None
    match_points2: list = None

    transform_type: str = ""
    rotation_deg: float = 0.0
    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    edge_concentration: float = 0.0
    edge_continuity: float = 0.0
    inlier_count: int = 0

    # 内部区域复制: 复制源区域与粘贴区域 (x, y, w, h), 原图像素坐标
    region1: tuple = None
    region2: tuple = None

    # 融合后的证据链: [{type, severity, similarity, details}, ...]
    evidence: list = field(default_factory=list)


class BaseDetector:
    name = "base"

    def __init__(self, config: dict):
        self.config = config

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        raise NotImplementedError
