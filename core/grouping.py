import re
import os


def extract_group_info(filepath: str, patterns: list[dict] = None) -> dict:
    if patterns is None:
        patterns = [
            {
                "name": "subdir_fov",
                "fov_regex": r"(?P<group>[^/]+)/(?P<fov>\d+_\d+)/",
                "channel_regex": r"_(?P<channel>CH\d+|Overlay)\.",
            }
        ]

    result = {
        "experiment": "",
        "group": "",
        "fov": "",
        "channel": "",
        "is_overlay": False,
        "base_name": "",
        "has_group_info": False,
        "group_key": "",
        "fov_key": "",
    }

    rel_path = filepath.replace(os.sep, '/')
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]

    for pat in patterns:
        fov_match = re.search(pat["fov_regex"], rel_path)
        ch_match = re.search(pat["channel_regex"], filename)

        if fov_match:
            result["group"] = fov_match.group("group")
            result["fov"] = fov_match.group("fov")
            result["has_group_info"] = True

        if ch_match:
            result["channel"] = ch_match.group("channel")
            result["is_overlay"] = result["channel"].upper() == "OVERLAY"

        if result["has_group_info"]:
            break

    result["base_name"] = stem
    result["group_key"] = result["group"]
    result["fov_key"] = f"{result['group']}/{result['fov']}" if result["has_group_info"] else ""

    return result


def compute_pair_relation(gi1: dict, gi2: dict) -> str:
    if not gi1["has_group_info"] or not gi2["has_group_info"]:
        return "UNKNOWN"

    same_group = gi1["group"] == gi2["group"]
    same_fov = gi1["fov"] and gi2["fov"] and gi1["fov"] == gi2["fov"]
    same_ch = gi1["channel"] and gi2["channel"] and gi1["channel"] == gi2["channel"]

    if same_group and same_fov and not same_ch:
        return "SAME_FOV_DIFF_CH"
    return "DIFF_FOV"
