#!/usr/bin/env python3
"""带 faulthandler 的基准运行包装：N 秒后 dump 所有线程栈，用于定位卡死点。"""
import faulthandler
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _console import fix_console_encoding  # noqa: E402

fix_console_encoding()

dump_path = open(Path(__file__).parent / 'results' / '_hang_trace.txt', 'w', encoding='utf-8')
faulthandler.dump_traceback_later(90, repeat=True, exit=False, file=dump_path)

sys.argv = ['run_bench.py'] + sys.argv[1:]
runpy.run_path(str(Path(__file__).parent / 'run_bench.py'), run_name='__main__')
