@echo off
rem 构建 dedup_core.exe (C++17 + OpenCV), 使用 conda 环境的 OpenCV/CMake/Ninja
rem 用法: scripts\build_core.bat
setlocal
set ENV=C:\Users\zhang\anaconda3\envs\dedup_build
set SCRIPT_DIR=%~dp0
set SRC=%SCRIPT_DIR%..\cpp
set BUILD=%SRC%\build

if not exist "%ENV%\Library\include\opencv4\opencv2\opencv.hpp" (
  echo [ERROR] 未找到 OpenCV 开发头文件, 请先:
  echo   conda create -y -n dedup_build -c conda-forge opencv cmake ninja
  exit /b 1
)

call "%ENV%\Scripts\activate.bat" dedup_build
if errorlevel 1 exit /b 1

cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DOpenCV_DIR="%ENV%\Library\lib\cmake\opencv4" ^
  -DCMAKE_PREFIX_PATH="%ENV%\Library"
if errorlevel 1 exit /b 1

cmake --build "%BUILD%"

if exist "%BUILD%\dedup_core.exe" (
  copy /y "%BUILD%\dedup_core.exe" "%SRC%\..\dedup_core.exe" >nul
  echo [OK] 内核已生成: %SRC%\..\dedup_core.exe
)
endlocal
