from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchResult:
    image1: str
    image2: str
    match_type: str
    similarity: float
    severity: str  # low / medium / high / critical
    details: str = ""
    is_cross_channel: bool = False
    match_points1: list = None
    match_points2: list = None


class BaseDetector:
    name = "base"

    def __init__(self, config: dict):
        self.config = config

    def compare(self, info1, info2, relation: str = "UNKNOWN") -> Optional[MatchResult]:
        raise NotImplementedError
