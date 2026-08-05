"""C++ 原生内核 (dedup_core.exe) 的 Python 壳。

可插拔设计: exe 存在且可运行 -> 用原生内核做哈希候选生成;
否则自动回退纯 Python 实现, 功能等价。
"""
import os
import shutil
import subprocess
import tempfile


def find_core_exe() -> str:
    """定位 dedup_core.exe (项目根目录 或 PATH)。"""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = [os.path.join(here, "dedup_core.exe")]
    found = shutil.which("dedup_core.exe")
    if found:
        cand.append(found)
    for p in cand:
        if p and os.path.exists(p):
            return p
    return ""


def _libs_dir() -> str:
    """依赖 DLL 目录 (与 exe 同级的 libs/, 供子进程搜索)。"""
    exe = find_core_exe()
    if not exe:
        return ""
    here = os.path.dirname(exe)
    libs = os.path.join(here, "libs")
    return libs if os.path.isdir(libs) else ""


def _run_env():
    """子进程环境: 把 libs/ 前置到 PATH。"""
    env = dict(os.environ)
    libs = _libs_dir()
    if libs:
        env["PATH"] = libs + os.pathsep + env.get("PATH", "")
    return env


def core_available() -> bool:
    """探测内核可用性 (exe 存在且能运行)。"""
    exe = find_core_exe()
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "hash"], capture_output=True, timeout=10,
                           env=_run_env())
        return r.returncode in (0, 2)
    except Exception:
        return False


def native_hash_candidates(directory: str, threshold: int = 15):
    """用原生内核生成哈希候选对, 返回 {(path1, path2)} 或 None (失败)。

    仅当目录不含无法用 OpenCV 解码的格式 (SVS/NDPI 等) 时使用。
    """
    exe = find_core_exe()
    if not exe:
        return None
    env = _run_env()
    with tempfile.TemporaryDirectory(prefix="dedup_core_") as td:
        bin_path = os.path.join(td, "hashes.bin")
        pairs_path = os.path.join(td, "pairs.txt")
        try:
            r1 = subprocess.run([exe, "hash", "--dir", directory,
                                 "--out", bin_path],
                                capture_output=True, timeout=1800, env=env)
            if r1.returncode != 0 or not os.path.exists(bin_path):
                return None
            r2 = subprocess.run([exe, "mih", "--in", bin_path,
                                 "--threshold", str(threshold),
                                 "--out", pairs_path],
                                capture_output=True, timeout=1800, env=env)
            if r2.returncode != 0 or not os.path.exists(pairs_path):
                return None
            pairs = set()
            with open(pairs_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 2 and parts[0] != parts[1]:
                        a, b = parts
                        pairs.add((a, b) if a < b else (b, a))
            return pairs
        except Exception:
            return None


def has_unsupported_formats(directory: str) -> bool:
    """目录中是否有 OpenCV 无法解码的科研格式 (SVS/NDPI/VSI)。"""
    for root, _dirs, files in os.walk(directory):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".svs", ".ndpi", ".vsi", ".czi", ".zvi", ".qptiff"):
                return True
    return False
