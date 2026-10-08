@echo off
chcp 65001 >nul
title 科研图片查重工具 v4

echo ============================================================
echo   🔬 科研图片查重工具 v4
echo   跨文件查重 + 图内 panel 复用检测
echo ============================================================
echo.

set "CURRENT_DIR=%~dp0"
set "CURRENT_DIR=%CURRENT_DIR:~0,-1%"

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
python -c "import cv2, numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo [2/3] 首次运行，正在安装依赖（需要几分钟）...
    echo.
    pip install -r "%CURRENT_DIR%\requirements.txt"
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败！请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo   依赖安装完成！
) else (
    echo   依赖已就绪
)

echo.
echo [3/3] 开始扫描目录: %CURRENT_DIR%
echo        （既查不同文件之间的重复，也查同一张组图内部的 panel 复用）
echo.
echo ────────────────────────────────────────────────────────────
echo.

python "%CURRENT_DIR%\image_dedup_v4.py" "%CURRENT_DIR%" --report "%CURRENT_DIR%\report.html"

echo.
echo ────────────────────────────────────────────────────────────
echo.
echo   ✅ 扫描完成！报告: report.html
echo.

if exist "%CURRENT_DIR%\report.html" (
    start "" "%CURRENT_DIR%\report.html"
)

echo.
echo 按任意键退出...
pause >nul
