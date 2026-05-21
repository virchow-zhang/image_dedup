#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打包图片查重工具为独立可执行文件

运行方式:
  python build_exe.py

打包完成后，可执行文件在 dist 目录下
"""

import os
import sys
import subprocess
import shutil

def main():
    print("=" * 60)
    print("  🔬 科研图片查重工具 - 打包程序")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "image_dedup.py")
    
    if not os.path.exists(script_path):
        print(f"错误: 找不到 {script_path}")
        sys.exit(1)
    
    # 步骤1: 安装PyInstaller
    print("\n[1/4] 检查并安装PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                   capture_output=True, check=False)
    print("  ✓ PyInstaller已就绪")
    
    # 步骤2: 清理旧的构建文件
    print("\n[2/4] 清理旧的构建文件...")
    for folder in ['build', 'dist', '__pycache__']:
        folder_path = os.path.join(script_dir, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"  已删除: {folder}/")
    
    spec_file = os.path.join(script_dir, "图片查重工具.spec")
    if os.path.exists(spec_file):
        os.remove(spec_file)
        print(f"  已删除: 图片查重工具.spec")
    
    # 步骤3: 打包
    print("\n[3/4] 开始打包 (这可能需要几分钟)...")
    
    # 获取依赖列表
    requirements_file = os.path.join(script_dir, "requirements.txt")
    hidden_imports = []
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    pkg = line.split('>=')[0].split('==')[0].split('<=')[0].strip()
                    hidden_imports.extend(["--hidden-import", pkg])
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                         # 打包成单个exe
        "--console",                         # 保留控制台窗口
        "--name", "图片查重工具",              # exe文件名
        "--clean",                           # 清理临时文件
        "--noconfirm",                       # 不询问确认
        "--add-data", f"{os.path.join(script_dir, 'requirements.txt')};.",  # 附带requirements
    ] + hidden_imports + [script_path]
    
    print(f"  执行命令: pyinstaller --onefile --console --name 图片查重工具 image_dedup.py")
    
    result = subprocess.run(cmd, cwd=script_dir, capture_output=False)
    
    if result.returncode == 0:
        print("\n[4/4] 打包完成！")
        exe_path = os.path.join(script_dir, "dist", "图片查重工具.exe")
        
        if os.path.exists(exe_path):
            file_size = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"""
{'=' * 60}
  ✅ 打包成功！
{'=' * 60}

  可执行文件: dist/图片查重工具.exe ({file_size:.1f} MB)

  使用方法:
  ─────────────────────────────────────────────────
  方法1: 双击运行
    将 图片查重工具.exe 复制到包含图片的目录
    双击运行，自动扫描当前目录的所有图片

  方法2: 命令行运行
    图片查重工具.exe                    # 扫描当前目录
    图片查重工具.exe D:\\photos          # 扫描指定目录
    图片查重工具.exe --threshold 3       # 调整敏感度
    图片查重工具.exe --help              # 查看帮助
  ─────────────────────────────────────────────────

  输出结果:
    - visualization/   可视化对比图目录
    - report.html      HTML报告
    - report.csv       CSV报告
""")
        else:
            print(f"\n  警告: 可执行文件未找到: {exe_path}")
    else:
        print("\n❌ 打包失败，请检查错误信息。")
        print("\n常见问题:")
        print("  1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("  2. 确保有足够的磁盘空间")
        print("  3. 检查是否有杀毒软件阻止")

if __name__ == "__main__":
    main()
