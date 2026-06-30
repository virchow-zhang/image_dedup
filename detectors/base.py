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


class BaseDetector:
    name = "base"

    def __init__(self, config: dict):
        self.config = config

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        raise NotImplementedError
