# 🔬 Image Dedup — 科研图片查重工具 v3

递归扫描目录下所有图片，检测科研论文中常见的图片重复 / 篡改：
完全复制、旋转/翻转、亮度调整、缩放、裁剪（子图）、边缘拼接、
**内部区域复制粘贴（CMFD）**、跨图拼接等。

**支持图片**：免疫荧光（多通道 TIF）、白光照片、WB、HE 染色、电镜图、JPG/PNG/BMP/WebP。

## 架构：三层流水线（替代 v2 的全量 O(n²) 比对）

```
[层0] 并行加载      16bit TIF 归一化 → 256 灰度 + pHash/dHash/aHash/wHash
                    + ORB 词袋键 + (懒加载) 1024px SIFT 特征
[层1] 候选生成      只生成"可能重复"的对, 典型过滤率 98.8% (1050张 -> 6605 对)
                    ├─ MIH 多索引哈希    pHash/dHash/aHash + 7 种变换变体
                    ├─ BOW ORB 倒排索引  自由旋转 / 裁剪 / 局部拼接
                    └─ (--ai) MobileNetV2 嵌入向量 余弦 top-k
[层2] 验证+判定      仅对候选对运行全部验证器 (8 线程并行):
                    pHash/dHash/aHash · SSIM · 5+2 种变换 · Canny 边缘重叠
                    · SIFT+FLANN+RANSAC (任意角度/缩放/局部复用, 含覆盖率判据)
                    · 多尺度 matchTemplate (子图/裁剪)
                    + 单图 CMFD (SIFT 自匹配+位移聚类, 输出源/粘贴区域坐标)
                    每对融合为一条主证据 + 证据链
```

## 性能（合成基准 1050 张）

| 指标 | v2 | v3 |
|------|-----|-----|
| 全量 SIFT O(n²) | 全部对 | 仅候选对 |
| 总耗时 (1050 张) | 小时级 | **~2 分钟** |
| 14 类变换召回率 | — | **100%** (980/980) |
| 基底间误报 | — | **0** (0/2415) |

## 原生 C++ 内核（可选加速）

`dedup_core.exe`（C++17 + OpenCV，独立 exe + Python 壳 subprocess 调用）：

- `hash` 批量感知哈希 · `mih` MIH 候选对 · `cmfd` 内部复制 · `template` 多尺度模板匹配
- **可插拔**：exe 存在即自动使用哈希候选层，缺失/失败自动回退纯 Python 实现
- 构建：`scripts\build_core.bat`（需 VS Build Tools 2022 + conda 环境 `dedup_build`）
- 分发：`dedup_core.exe` + `libs\`（60 个依赖 DLL，脚本 `scripts\deps.py` 自动裁剪）

## 快速开始

```bash
pip install -r requirements.txt
python image_dedup.py -d D:\IF_images\20260629        # 基础扫描
python image_dedup.py -d D:\IF_images --strict        # 严格模式 (阈值更严)
python image_dedup.py -d D:\IF_images --ai            # AI 嵌入层 (自动下载模型)
python image_dedup.py -d D:\IF_images --quick         # 快速模式
```

## 命令行参数

```
  -d, --directory TEXT   待扫描目录（默认：当前目录）
  -c, --config FILE      配置文件路径
  -o, --output DIR       报告输出目录
  --quick                快速模式（跳过旋转/子图/CMFD/AI）
  --strict               严格模式（降低所有阈值）
  --ai / --no-ai         启用/禁用 AI 嵌入候选层（模型自动下载, 约 14MB）
  --no-cmfd              跳过内部区域复制检测
  --no-orb / --no-sift   跳过 SIFT 特征匹配验证
  --no-rotation          跳过旋转/翻转检测
  --no-subimage          跳过于图/裁剪检测
  --no-edge              跳过边缘重叠检测
  --threads N            并行加载/验证线程数
  --threshold N          pHash 阈值覆盖
  --ssim F               SSIM 阈值覆盖
  -q, --quiet            静默模式
  --version              版本信息
```

## 多通道免疫荧光（IF）支持

`{组名}/{FOV编号}/{文件名}_CH{通道号}.tif` 结构下自动识别同 FOV 不同通道：
同 FOV·异通道对自动排除（属正常成像），其余全量比对。可用 `config.yaml` 自定义正则。

## 检测功能与标记

| 检测 | 算法 | 可视化 |
|------|------|--------|
| 完全相同 | MD5 | — |
| 微小修改/亮度 | pHash/dHash/aHash/SSIM | 差异高亮框 |
| 旋转/翻转 | 7 种变换哈希 + SIFT RANSAC 角度 | 匹配线 + 几何标签 |
| 缩放 | SIFT 单应矩阵尺度 | 几何标签 |
| 裁剪/子图 | 多尺度 matchTemplate 金字塔 | 红框定位 |
| 边缘拼接 | Canny 边缘带 NCC + SIFT 边缘集中 | 橙色条带 |
| 内部区域复制 | SIFT 自匹配 + 位移聚类 | 🔴 源区域 / 🟢 粘贴区域 |
| 跨图拼接 | BOW 候选 + SIFT RANSAC | 匹配线 |
| 重后期处理近重复 | (--ai) 嵌入向量候选 | — |

## 报告

`dedup_report/report/index.html` — 交互式 HTML（严重度筛选、证据链、Lightbox 放大）+ `comparisons.csv`。

## 测试与评估

```bash
python tests/gen_synthetic.py --out tests/data/syn1 --baselines 30   # 生成 14 类变换测试集
python tests/eval.py --data tests/data/syn1                          # 逐类型召回率 + 误报
```

## 依赖

- Python ≥ 3.9：Pillow, numpy, opencv-python, imagehash, PyYAML, tifffile
- 可选：onnxruntime + onnx（--ai 层）；VS Build Tools + conda `opencv`（C++ 内核）

## 开源协议

MIT License
