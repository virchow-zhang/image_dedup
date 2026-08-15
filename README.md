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

### 方法三：命令行运行（推荐用优化版）

```bash
# 扫描当前目录
python image_dedup.py

# 扫描指定目录（优化版，速度更快、误报更低）
python image_dedup_optimized.py D:\photos

# 调整参数
python image_dedup_optimized.py D:\photos --min-confidence 0.7 --min-votes 3

# 查看帮助
python image_dedup_optimized.py --help
```

## 优化版（image_dedup_optimized.py）

在 `image_dedup.py` 基础上做了三方面增强，**推荐使用**：

### 1. 降低误报率
- **窗口化 SSIM（内容加权）**：替代全局标量SSIM。科研图大片背景相似但内容不同的两张图，
  旧算法 SSIM 会虚高到 0.9+（实测0.96），现在按局部方差加权，仅内容区域主导分数
- **低信息量图过滤**：空白/纯色图（灰度std过低）只走MD5精确匹配，消除
  "两张任意空白图互报 critical" 的经典误报
- **投票去相关**：pHash/dHash 同源高度相关、直方图纯统计信号，三者互不独立。
  现改为 pHash/dHash/SSIM 三个结构类检测器投票（`--min-votes`），直方图降为参考票
- **亮度归一化SSIM**：专捕"同结构不同曝光"（亮度/对比度/曝光调整）的副本，实测区分度 ~1.0 vs <0
- **ORB二次确认**：恰好最低票数通过的边缘候选对，用ORB特征匹配独立复核，
  确认失败默认降级（`--orb-strict` 直接剔除）
- **旋转检测重写**：原 whash-64bit 对深色稀疏条带图无区分度（真实旋转与随机图对
  差异区间重叠，误报极高）；改用变换后 phash-256bit 比较（真旋转≤10、随机对≥90，
  阈值24两侧余量充足），并用**等比缩放画布**修复非方形图旋转比较失真的问题

### 2. 补回漏掉的检测能力
- **子图/裁剪检测（快速版）**：原版 O(w·h·scale) 滑动窗口慢到不可用被删除；
  现用"面积比过滤 + 降采样 + 多尺度 matchTemplate"，毫秒级，条带复用类造假可检出
- **旋转/翻转增强召回**：旋转副本哈希差异巨大、进不了普通索引，旋转检测形同虚设；
  现为每张图额外索引 5 种几何变换哈希，旋转副本重新可召回

### 3. 交互性能
- 目录单次遍历（原实现按扩展名重复遍历 32 次）、MD5 分块流式、超大图（SVS/NDPI）
  draft 无损降采样解码，防内存爆炸
- 缩略图按图片路径缓存 + 并行生成（原实现每对重复生成、串行）
- HTML 报告新增：文件名搜索、分页（50/100/200条）、"出现N次"badge（同一张图
  出现在多对里时提示优先审查）、Esc 关闭全屏
- 新增 `--min-confidence`（置信度过滤）、`--no-thumbnails`（大报告加速）、
  JSON 报告输出（`--report x.json`，便于二次处理）

```bash
python image_dedup_optimized.py D:\photos --no-thumbnails --report result.json
python image_dedup_optimized.py D:\photos --min-confidence 0.8 --no-rotation
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
