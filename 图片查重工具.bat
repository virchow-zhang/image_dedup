@echo off
chcp 65001 >nul
title 科研图片查重工具

echo ============================================================
echo   🔬 科研图片查重工具
echo ============================================================
echo.

REM 获取当前目录
set "CURRENT_DIR=%~dp0"
set "CURRENT_DIR=%CURRENT_DIR:~0,-1%"

REM 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM 检查依赖是否已安装
echo [1/3] 检查依赖...
python -c "import imagehash, numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo [2/3] 首次运行，正在安装依赖（需要几分钟）...
    echo.
    pip install -r "%CURRENT_DIR%\requirements.txt"
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败！
        echo 请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
    echo   依赖安装完成！
) else (
    echo   依赖已就绪
)

echo.
echo [3/3] 开始扫描当前目录: %CURRENT_DIR%
echo.
echo ────────────────────────────────────────────────────────────
echo.

REM 清理旧的可视化结果（可选，取消注释下一行）
REM rmdir /s /q "%CURRENT_DIR%\visualization" 2>nul

REM 运行查重工具
python "%CURRENT_DIR%\image_dedup.py" "%CURRENT_DIR%"

echo.
echo ────────────────────────────────────────────────────────────
echo.
echo   ✅ 扫描完成！请查看以下文件:
echo.
echo   📁 visualization\    可视化对比图目录
echo   📄 *report*.html     HTML报告（推荐，浏览器打开）
echo   📄 *report*.csv      CSV报告（可用Excel打开）
echo.

REM 自动打开HTML报告
for %%f in ("%CURRENT_DIR%\*report*.html") do (
    echo   正在打开报告: %%f
    start "" "%%f"
    goto :found
)
:found

echo.
echo 按任意键退出...
pause >nul
