#!/usr/bin/env python3
"""
科研图片查重工具 v4 —— 入口
============================
与 v2 的本质区别：v2 只回答「不同文件之间是否重复」；v3 还会回答
「同一张组图内部，panel 之间是否重叠/复用/复制粘贴」。

    python image_dedup_v4.py <目录>
    python image_dedup_v4.py <目录> --mode intra
    python image_dedup_v4.py <目录> --report report.html --json result.json

旧版保留在 legacy/ 下，可直接对照运行。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dedup.cli import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
