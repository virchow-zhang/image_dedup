# 🔬 Image Dedup — 科研图片查重工具

递归扫描目录下所有图片，全量两两比对，检测科研论文中常见的图片重复 / 篡改。

**支持的图片类型**：免疫荧光（多通道）、白光照片、WB 图片、HE 染色、电镜图等。

## 检测管线

```
全量两两比对 O(n²)
│
├── 快速检测器（全部对）             ← 高召回粗筛
│   ├── MD5                       完全相同
│   ├── pHash / dHash / aHash     亮度/对比度微调、JPEG 重压缩
│   ├── SSIM                      结构相似性
│   ├── 5 种变换 + pHash          旋转 90°/180°/270°、水平/垂直翻转
│   └── Canny 边缘 + 边缘重合率   凝胶电泳条带拼接、显微镜视野拼接
│
├── SIFT + RANSAC（全部对）        ← 精匹配，单引擎覆盖所有几何变换
│   ├── CLAHE 预处理 → SIFT 提取 → FLANN 匹配 → Lowe ratio test
│   ├── RANSAC 单应矩阵估计
│   └── 几何分析输出:
│       ├── 变换类型（平移 / 旋转 / 缩放）
│       ├── 旋转角度、缩放比
│       ├── 位移向量 (dx, dy)
│       ├── 边缘集中度（内点是否集中在同一边缘）
│       └── 边缘连续性（内点在边缘上是否连续分布）
│
└── 子图/裁剪检测（全部对）         多尺度模板匹配
```

**关键设计**：SIFT + RANSAC 是唯一的特征匹配引擎，自然覆盖平移（视野重叠）、旋转、缩放、局部复制等所有场景，无需独立的 ORB / FOV 检测器。

## 输出

- **交互式 HTML 报告** — 零外部依赖，每条匹配附带几何标签（旋转角度、缩放比、边缘集中度）
- **CSV 导出** — 可用 Excel 直接打开
- **红线圈点注释图** — SIFT 特征匹配点 + 红色连线 + 橙色边缘条带高亮

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 扫描当前目录
python image_dedup.py

# 3. 扫描指定目录
python image_dedup.py -d D:\IF_images\20260629

# 4. 严格模式
python image_dedup.py -d D:\IF_images --strict

# 5. 快速模式（跳过 SIFT/子图/边缘）
python image_dedup.py -d D:\IF_images --quick
```

## 多通道免疫荧光（IF）支持

当图片遵循 `{组名}/{FOV编号}/{文件名}_CH{通道号}.tif` 的目录结构时，工具自动识别同一视野（FOV）的不同通道，自适应调整阈值：

| 比对类型 | 示例 | 行为 |
|---------|------|------|
| **同 FOV·不同通道** | `D1/40_01/40X_CH2.tif` vs `D1/40_01/40X_CH4.tif` | 阈值自动收紧（SSIM 0.98→0.999），SIFT 跳过 |
| **不同 FOV·同通道** | `D1/40_01/40X_CH2.tif` vs `D1/40_02/40X_CH2.tif` | 标准阈值 |
| **不同 FOV·不同通道** | `D1/40_01/40X_CH2.tif` vs `PE1/40_01/40X_CH2.tif` | 标准阈值 |
| **扁平目录** | 所有文件在同一目录 | 全量标准比对 |

分组信息仅用于阈值决策，**不断开任何比对**。可在 `config.yaml` 中自定义正则表达式适配不同命名规则。

## 几何标签说明

SIFT 匹配结果在报告中附带几何标签：

| 标签 | 含义 | 示例场景 |
|------|------|----------|
| `平移` | 纯平移，内点集中在边缘 | **视野误重叠**（载物台移动不足） |
| `旋转` | 存在旋转角度 | 旋转复制图片 |
| `缩放` | 缩放比 ≠ 1 | 放大/缩小后复用 |
| `边缘xx%` | 边缘集中度百分比 | 匹配点集中在同一边缘条带 |
| `连续xx%` | 边缘连续性百分比 | 匹配点在边缘上均匀分布 |

## 命令行参数

```
python image_dedup.py [选项]

选项：
  -d, --directory TEXT   待扫描目录（默认：当前目录）
  -c, --config FILE      配置文件路径（默认：./config.yaml）
  -o, --output DIR       报告输出目录
  --quick                快速模式（跳过 SIFT/子图/边缘）
  --strict               严格模式（降低所有阈值）
  --no-sift              跳过 SIFT 特征匹配
  --no-rotation          跳过旋转/翻转检测
  --no-subimage          跳过子图/裁剪检测
  --no-edge              跳过边缘重叠检测
  --threads N            并行线程数
  --threshold N          pHash 阈值覆盖
  --ssim F               SSIM 阈值覆盖
  -q, --quiet            静默模式
  --version              版本信息
```

## 配置文件 `config.yaml`

```yaml
scan:
  directory: null
  recursive: true
  extensions: [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"]

grouping:
  patterns:
    - name: "subdir_fov"
      fov_regex: "(?P<group>[^/]+)/(?P<fov>\\d+_\\d+)/"
      channel_regex: "_(?P<channel>CH\\d+|Overlay)\\."
  cross_channel:
    phash_boost: 0.6
    ssim_boost: 1.1

detection:
  phash_threshold: 3
  phash_hash_size: 16
  ssim_threshold: 0.98
  hist_threshold: 0.9999
  sift_enabled: true
  sift_min_inliers: 10
  edge_threshold: 0.55
  subimage_threshold: 0.85

report:
  output_dir: "dedup_report"
  clean_output: true
  thumbnail_height: 500
```

## 报告结构

```
dedup_output/
├── report/
│   ├── index.html            # 交互式 HTML 报告
│   ├── comparisons.csv       # CSV（Excel 可打开）
│   └── vis/                  # 注释对比图
│       ├── 0001_CRI_....jpg
│       ├── 0002_CRI_....jpg
│       └── ...
└── config.yaml               # 本次运行的配置快照
```

### HTML 报告功能

- **汇总看板**：图片总数、比对对数、严重程度、检测类型分布
- **过滤器**：按严重程度（CRITICAL/HIGH/MEDIUM）、跨通道比对
- **对比卡片**：每对匹配一张卡片，顶部显示文件名、严重程度徽章、相似度、几何标签
- **注释图**：左 A + 右 B，红色连线连接 SIFT 匹配点，橙色条带高亮边缘集中区域
- **详情折叠**：展开查看变换类型、旋转角、缩放比、位移向量、内点数
- **Lightbox**：点击注释图放大查看

### 注释图标记

| 标记 | 含义 |
|------|------|
| 🟥 红色方框 | 匹配/差异区域 |
| 🔴 红色实线 | SIFT 特征点匹配连线 |
| ⭕ 红色圆点 | 特征匹配点 |
| 🟧 橙色条带 | 边缘集中高亮（匹配点密集在边缘） |
| 🟡 黄色端点 | 边缘对齐参考点 |

## 测试结果（559 张 IF 荧光图）

| 阶段 | 时间 | 结果 |
|------|------|------|
| 图片加载 + 特征计算 | 24s | 559 张 (1920×1440, 16位 LZW TIFF) |
| Fast detectors (5个) | 6 min | 320 匹配 |
| SIFT + RANSAC (全部 155k 对) | 90s | 167 匹配（含几何参数） |
| 总计 | ~7.5 min | 487 匹配 |

## 依赖

- Python ≥ 3.9
- Pillow, numpy, opencv-python (≥ 4.8), imagehash, PyYAML, scikit-learn
- 对于 LZW 压缩的 TIFF：`pip install imagecodecs`

## 开源协议

MIT License
