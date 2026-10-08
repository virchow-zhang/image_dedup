# 🔬 科研图片查重工具 v3

用于检测科研论文中图片重复/复用的工具。**v3 的核心新增能力是检查同一张组图内部
panel 之间的重叠、复用与复制粘贴** —— 这正是 SCI 组图投稿中最容易被忽略、
也最常出问题的一类。

> 与 v2 的完整对比（含实测数据）见 **[COMPARISON.md](COMPARISON.md)**。
> 旧版保留在 `legacy/`，可直接对照运行。

---

## 快速开始

### 方法一：双击运行 ⭐

双击 `图片查重工具.bat` —— 自动检查依赖、扫描当前目录、生成并打开交互式报告。

### 方法二：命令行

```bash
# 同时做「跨文件查重」和「图内 panel 复用检测」（默认）
python image_dedup_v3.py D:\my_paper_figures

# 只查同一张组图内部的 panel 复用
python image_dedup_v3.py D:\my_paper_figures --mode intra

# 只查不同文件之间的重复
python image_dedup_v3.py D:\my_paper_figures --mode cross

# 指定报告 / JSON 输出
python image_dedup_v3.py D:\figs --report out.html --json out.json
```

---

## 两类检测任务

| | 跨文件查重 `--mode cross` | 图内复用检测 `--mode intra` |
|---|---|---|
| 回答的问题 | 同一张图是否在多个文件里重复出现 | 同一张组图里，两个 panel 是否同源 |
| 典型场景 | 一图多用、换名重复投稿 | 组图拼接时误用同一张源图、panel 局部重叠、图内复制粘贴 |
| 方法 | 多变换感知哈希索引 + 结构检测器投票 | 组图 XY-cut 切分 + SIFT 自匹配 + 迭代 RANSAC + NCC 验证 |
| 输出 | 文件对 + 相似度 + 置信度 | **区域坐标对** + 变换类型 + 区域归属 panel |

### 能识别的变换

精确复制、JPEG 重压缩、亮度/对比度/gamma 调整、加噪、缩放、裁剪、
**旋转 90°/180°、水平/垂直镜像**、panel 部分重叠、panel 内复制粘贴。

---

## 交互式报告

运行后生成自包含的 `report.html`（图片内嵌，无外部依赖，双击即可离线打开）：

- **SVG 覆盖层标注**：A 区（蓝实线）/ B 区（橙虚线）/ 自动切分出的 panel 边界（灰点线），
  缩放时框线保持锐利
- **滚轮缩放 + 拖拽平移**：可直接放大到原始分辨率查看细节
- **原始分辨率裁剪对照**：A/B 两处内容并排显示
- **闪烁对比**：同一位置交替显示 A/B 内容 —— 同源时几乎看不出切换，是最直观的判据
- 按文件名搜索、按严重程度过滤

---

## 命令行参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `directory` | 要扫描的目录 | 当前目录 |
| `--mode` | `cross` / `intra` / `both` | `both` |
| `--report PATH` | HTML 报告输出路径 | `<目录>/image_dedup_v3_report.html` |
| `--json PATH` | JSON 结果输出路径 | 不输出 |
| `--workers N` | 并行线程数 | CPU 核数（上限 16） |
| `--hash-threshold N` | pHash 差异阈值（256 位），越小越严格 | 12 |
| `--min-votes N` | 跨文件：至少几种结构检测器通过 | 2 |
| `--min-inliers N` | 图内：RANSAC 最少内点数 | 14 |
| `--ncc-threshold F` | 图内：归一化互相关验证阈值 | 0.55 |
| `--no-thumbnails` | 报告不内嵌图片（文件更小） | - |

---

## 性能

| 规模 | v2 总耗时 | v3 总耗时 | v2 峰值内存 | v3 峰值内存 |
|---|---|---|---|---|
| 120 张 | 6.8 s | **4.5 s** | 265 MB | **79.6 MB** |
| 1200 张 | 24.2 s | **3.2 s** | 410 MB | **85.5 MB** |

检测阶段（不含载入）在 1200 张时从 14.07 s 降到 **0.98 s**（14×）。
加速来自把「逐对 Python 调用」改成「numpy 分块矩阵运算」，而非换语言；
评估过程见 COMPARISON.md 第 4 节。

---

## 基准测试

```bash
python bench/gen_dataset.py --out bench/data --clean          # 生成带真值的数据集
python bench/run_bench.py --detector v3 --out bench/results/v3.json
python bench/show_result.py bench/results/v3.json             # 看明细
python bench/compare.py --results <旧.json> <新.json>          # 多维对比表
```

数据集的每个 case 都注入了**已知的**变换类型与**精确的区域坐标**，
因此既能算准确率/召回，也能算定位 IoU。

---

## 环境要求

- Python 3.8+
- `pip install -r requirements.txt`（opencv-python / numpy / Pillow）

## 支持的图片格式

JPEG、PNG、BMP、TIFF、GIF、WebP，以及科研格式 `.svs` / `.ndpi` / `.vsi`。

## 目录结构

```
image_dedup_v3.py        # 入口
dedup/
  crossfile.py           # 跨文件查重（向量化哈希 + 投票）
  features.py            # 图内区域匹配核心（SIFT + 迭代 RANSAC + 验证）
  panels.py              # 组图 panel 切分（递归 XY-cut）
  report.py              # 交互式 HTML 报告
  cli.py                 # 命令行
bench/                   # 基准数据集生成 / 评测 / 对比
legacy/v1, legacy/v2     # 旧版备份（可直接运行对照）
```

## 许可证

MIT License

## 致谢

- 优化思路参考 [xImageDuplicateChecker](https://github.com/ayumilove/xImageDuplicateChecker) (MIT)
- 图内复制检测方法参考 Amerini 等 *A SIFT-based forensic method for copy-move attack detection*
