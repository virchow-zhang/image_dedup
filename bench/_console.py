"""控制台输出编码修复：Windows 默认 GBK 无法输出 ✓/✗/emoji。"""

import sys


def fix_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
