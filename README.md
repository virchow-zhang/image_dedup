# 🔬 科研图片查重工具

一个用于检测科研论文中图片重复/造假的工具，支持多种检测算法。

## 快速开始

### 方法一：双击运行（最简单）⭐

1. **首次使用**：双击 `图片查重工具.bat`
   - 自动检查并安装依赖
   - 自动扫描当前目录的所有图片
   - 自动打开HTML报告

2. **后续使用**：直接双击 `图片查重工具.bat`

### 方法二：打包成EXE（推荐分发）

```bash
python build_exe.py
```

打包完成后，`dist/图片查重工具.exe` 可以：
- 复制到任何包含图片的目录
- 双击直接运行，无需Python环境
- 分发给其他人使用

### 方法三：命令行运行

```bash
# 扫描当前目录
python image_dedup.py

# 扫描指定目录
python image_dedup.py D:\photos

# 调整参数
python image_dedup.py D:\photos --threshold 3 --workers 8

# 查看帮助
python image_dedup.py --help
```

## 检测功能

| 检测类型 | 说明 | 严重程度 |
|---------|------|---------|
| 完全相同 | MD5哈希完全一致 | 🔴 严重 |
| 感知哈希相似(pHash) | 微小修改、亮度调整 | 🔴 严重 / 🟠 高 |
| 差异哈希相似(dHash) | 渐变、边缘变化 | 🔴 严重 / 🟠 高 |
| 结构相似(SSIM) | 整体结构相似 | 🔴 严重 / 🟠 高 |
| 直方图相似 | 曝光/亮度调整 | 🟠 高 / 🟡 中 |
| 疑似旋转/翻转 | 90°/180°/270°旋转、水平/垂直翻转 | 🔴 严重 |
| 疑似缩放 | 分辨率调整 | 🟠 高 |
| 疑似子图/裁剪 | 局部截取 | 🔴 严重 |
| 边缘重叠/拼接 | 拼接造假 | 🟠 高 |
| 亮度/对比度调整 | 明暗变化 | 🟠 高 |
| 内部区域复制 | 同一图片内复制粘贴 | 🔴 严重 |

## 输出结果

运行后会在目录下生成：

```
your_directory/
├── visualization/          # 可视化对比图目录
│   ├── 001_CRI_100pct_MOUSE1_vs_MOUSE2.jpg
│   ├── 002_HIG_95pct_MOUSE1_vs_MOUSE3.jpg
│   └── ...
├── report.html             # HTML报告（推荐，浏览器打开）
├── report.csv              # CSV报告（可用Excel打开）
└── image_dedup_report_*.html  # 带时间戳的报告备份
```

## 可视化图说明

可视化图中：
- **红色框**：标出疑似重复的部位
- **绿色框**：仅在「内部区域复制」检测中出现，标示同一张图内被复制的第二个区域

## 命令行参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `directory` | 要扫描的目录 | 当前目录 |
| `--threshold N` | 哈希差异阈值（越小越严格） | 5 |
| `--ssim-threshold F` | SSIM阈值（越大越严格） | 0.85 |
| `--hist-threshold F` | 直方图阈值 | 0.80 |
| `--workers N` | 并行工作进程数 | 4 |
| `--report PATH` | 报告输出路径 | 自动生成 |
| `--no-rotation` | 跳过旋转/翻转检测 | - |
| `--no-subimage` | 跳过子图/裁剪检测 | - |
| `--no-edge` | 跳过边缘重叠检测 | - |
| `--no-internal` | 跳过内部区域复制检测 | - |
| `--hash-size N` | 哈希大小（越大越精确但越慢） | 16 |

## 使用示例

```bash
# 严格模式（检测更多细节）
python image_dedup.py --threshold 3

# 宽松模式（减少误报）
python image_dedup.py --threshold 8

# 只检测完全相同和高度相似
python image_dedup.py --threshold 2 --ssim-threshold 0.95

# 快速模式（跳过耗时的检测）
python image_dedup.py --no-rotation --no-subimage --no-internal

# 输出到指定文件
python image_dedup.py --report result.html
```

## 支持的图片格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- GIF (.gif)
- WebP (.webp)
- 科研格式 (.svs, .ndpi, .vsi)

## 环境要求

- Python 3.8+
- 依赖包：见 `requirements.txt`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 许可证

MIT License

## 致谢

- 优化技术参考: [xImageDuplicateChecker](https://github.com/ayumilove/xImageDuplicateChecker) (MIT License)
