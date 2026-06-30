# 🔬 Image Dedup — 科研图片查重工具

自动递归扫描目录下的所有图片，进行全量两两比对，检测科研论文中常见的图片重复/造假行为。

**支持的图片类型**：免疫荧光（多通道）、白光照片、WB 图片、HE 染色、电镜图等。

## 检测能力

| 类型 | 检测手段 | 用途 |
|------|---------|------|
| 完全相同 | MD5 + 文件大小 | 完全复制的文件 |
| 微小修改 / 亮度调整 | pHash + dHash + aHash | 对比度/曝光度微调、JPEG 重压缩 |
| 结构相似 | SSIM | 整体画面结构一致（旋转/缩放/平移后的裁剪） |
| 曝光度/全局调整 | 直方图相关性 + 巴氏系数 | 统一的亮度/对比度调整 |
| 旋转 / 翻转 | 5 种变换 + pHash | 90°/180°/270° 旋转、水平/垂直镜像 |
| 特征匹配 | ORB + FLANN + RANSAC | 局部修改、拼接、旋转+缩放组合 |
| 边缘重叠 | Canny 边缘 + 边缘重合率 | 凝胶电泳条带拼接、显微镜视野拼接 |
| 子图/裁剪 | 多尺度模板匹配 | 一张图截取局部作为另一张图 |

## 输出

- **交互式 HTML 报告** — 零外部依赖，支持严重程度/类型/跨通道筛选，每对含注释对比图
- **CSV 导出** — 可用 Excel 直接打开
- **红线圈点注释图** — 特征匹配点 + 红色连接线 + 区域框

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 扫描当前目录
python image_dedup.py

# 3. 扫描指定目录
python image_dedup.py -d D:\IF_images\20260629

# 4. 严格模式（降低阈值，捕获更细微的重复）
python image_dedup.py -d D:\IF_images --strict

# 5. 快速模式（跳过 ORB/子图/边缘等耗时检测）
python image_dedup.py -d D:\IF_images --quick
```

## 多通道免疫荧光（IF）支持

当图片遵循 `{组名}/{FOV\_编号}/{文件名}\_CH{通道号}.tif` 的目录结构时，工具自动识别同一视野（FOV）的不同通道，并自适应调整阈值：

| 比对类型 | 示例 | 行为 |
|---------|------|------|
| **同 FOV·不同通道** | `D1/40_01/40X_CH2.tif` vs `D1/40_01/40X_CH4.tif` | 阈值自动收紧（SSIM 0.98→0.999） |
| **不同 FOV·同通道** | `D1/40_01/40X_CH2.tif` vs `D1/40_02/40X_CH2.tif` | 标准阈值 |
| **不同 FOV·不同通道** | `D1/40_01/40X_CH2.tif` vs `PE1/40_01/40X_CH2.tif` | 标准阈值 |
| **扁平目录** | 所有文件在同一目录 | 全量标准比对，分组不生效 |

分组信息仅用于阈值决策，**不断开任何比对组合**。对于命名不规范的目录，可在 `config.yaml` 中自定义正则表达式。

## 命令行参数

```
python image_dedup.py [选项]

选项：
  -d, --directory TEXT   待扫描目录（默认：当前目录）
  -c, --config FILE      配置文件路径（默认：./config.yaml）
  -o, --output DIR       报告输出目录
  --quick                快速模式（跳过 ORB/子图/边缘）
  --strict               严格模式（降低所有阈值）
  --no-orb               跳过 ORB 特征匹配
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
  directory: null          # 待扫描目录（CLI -d 覆盖）
  recursive: true
  extensions: [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"]

grouping:
  patterns:
    - name: "subdir_fov"
      fov_regex: "(?P<group>[^/]+)/(?P<fov>\\d+_\\d+)/"
      channel_regex: "_(?P<channel>CH\\d+|Overlay)\\."
  cross_channel:
    phash_boost: 0.6      # 同FOV跨通道时 pHash 阈值 × 0.6
    ssim_boost: 1.1       # 同FOV跨通道时 SSIM 阈值 × 1.1

detection:
  phash_threshold: 3
  phash_hash_size: 16
  ssim_threshold: 0.98
  hist_threshold: 0.9999
  orb_enabled: true
  orb_min_matches: 10
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

- **汇总看板**：图片总数、比对对数、严重程度分布
- **过滤器**：按严重程度（CRITICAL/HIGH/MEDIUM）、匹配类型、跨通道比对 筛选
- **对比卡片**：每对匹配一张卡片，顶部显示文件名、严重程度徽章、相似度
- **注释图**：左侧原图 A + 右侧原图 B，红框标注匹配区域，红色连线连接 ORB 特征点
- **详情折叠**：点击展开完整文件路径、哈希值、匹配细节
- **Lightbox**：点击注释图放大查看

## 注释图说明

注释图使用不同颜色和标记表示不同的检测结果：

| 标记 | 含义 |
|------|------|
| 🟥 红色方框 | 匹配/差异区域 |
| 🔴 红色实线 | ORB 特征点匹配连线 |
| ⭕ 红色圆点 | 特征匹配点（重要性从高到低编号 1-5） |
| 🟧 橙色条带 | 边缘重叠检测区域 |
| 🟡 黄色端点 | 边缘对齐参考点 |

## 依赖

- Python ≥ 3.9
- Pillow, numpy, opencv-python, imagehash, PyYAML
- 对于 LZW 压缩的 TIFF：`pip install imagecodecs`

## 许可证

MIT License
