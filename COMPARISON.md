# v2 → v3 → v4 升级与整合对比报告

> 结论：仓库里存在**两条互不相关的开发线**（`main` 与 `master`，无共同祖先），
> 版本号还互相错位。本报告先查清版本关系，再用**同一套评分标准**把三条实现
> 放在一起实测，最后把两边各自的长处整合为 **v4**。
> 所有数字都来自仓库内 `bench/` 的可复现基准。

---

## 1. 版本关系：确实存在问题

```
main    (有共同祖先？否)
  e0d41a3  2026-05-21  喝咖啡的兔子   Initial commit: 科研图片查重工具
  bb5d5bd  2026-08-15  virchow-zhang  v2: 大幅降低误报率并提升交互性能
  e02a929  2026-10-08  virchow-zhang  v3: 新增图内 panel 复用检测 …（本次）

master  (独立历史，与 main 无共同祖先)
  d53e9bf  2026-07-01  v2.0: scientific image duplication detection tool
  4847520  2026-07-01  Add comprehensive README
  86876ed  2026-07-01  Add FOV overlap detector (SIFT+RANSAC+DBSCAN)
  18259d4  2026-07-01  Merge ORB+FOV into unified SIFT+RANSAC pipeline
  a6cd119  2026-07-01  Rewrite README for unified SIFT+RANSAC pipeline
  168b9f5  2026-08-06  v3.0: three-tier pipeline with MIH/BOW candidates, SIFT
                       verification, CMFD and optional AI embeddings
```

三个具体问题：

1. **两条历史无共同祖先** —— `git merge-base main origin/master` 直接报错。
   同一份仓库里是两个平行宇宙。
2. **版本号错位**：master 的「v3.0」是 8-06，main 的「v2」是 8-15。
   按时间，master 的 v3 反而**早于** main 的 v2；按编号却更高。
   只按 tag/分支名判断「谁更新」必然出错。
3. **能力互补但互不知情**：master 有 C++ 内核 / CMFD / 子图 / 边缘拼接，
   main 有区域级定位与交互报告。两边各补了对方一半。

`master` 已打归档 tag `archive/master-v3-2026-08`（指向 `168b9f5`），
本地可用 `git worktree add ../image_dedup_master archive/master-v3-2026-08` 取出。

---

## 2. 同台实测：三条实现跑同一套基准

基准见第 4 节。为了公平，master 那条线通过 `bench/run_bench.py --detector v3m`
接入同一评分器（pair 级 P/R/F1 + 区域级 IoU），而不是各说各话。

### A. 120 张

| 指标 | v2 | main-v3 | **master-v3.0** | 整合版 v4 |
|---|---|---|---|---|
| 跨文件 P | 1.000 | 1.000 | **0.156** | **1.000** |
| 跨文件 R | 0.542 | 0.875 | 0.875 | **0.958** |
| 跨文件 F1 | 0.703 | 0.933 | **0.264** | **0.979** |
| TP / FP / FN | 13/0/11 | 21/0/3 | 21/**114**/3 | **23/0/1** |
| 图内复用 区域F1 | ❌无能力 | 0.722 | **0.000** | 0.722 |
| 图内复用 定位IoU | — | 0.786 | **0.000** | 0.786 |
| 总耗时 | 6.8s | 3.7s | 7.8s | 4.4s |
| 检测耗时 | 5.53s | 0.04s | 7.41s | 0.10s |
| 峰值内存 | 265MB | 58.5MB | 329MB | 80.7MB |

**最有意思的一点**：master 与 main 的 v3 **漏报完全相同**（都是 TP=21 / FN=3，
同样是那三张 crop60 / crop80 / brightness），但 master 多了 **114 个误报**。

### B. 1200 张（跨文件）

| 指标 | v2 | main-v3 | 整合版 v4 |
|---|---|---|---|
| 精确率 P | 0.562 | 1.000 | **1.000** |
| 召回率 R | 0.410 | 0.825 | **0.955** |
| F1 | 0.474 | 0.904 | **0.977** |
| TP / FP / FN | 82/64/118 | 165/0/35 | **191/0/9** |
| 检测耗时 | 14.07s | 0.98s | 2.46s |
| 峰值内存 | 410MB | 85.5MB | 89.0MB |

---

## 3. master 那条线为什么误报这么多

不是"实现差"，是一个**具体且可定位的技术原因**。

把 114 个误报按目录归类后发现：**88 个是跨 case 的**（完全不同的实验之间），
且样例几乎全部是 `scatter_*` 和 `barplot_*`：

```
scatter_cond0_1.jpg  <->  scatter_cond1_3.jpg      （两张不同的散点图）
barplot_cond1_5.jpg  <->  barplot_cond2_1_rep.jpg  （两张不同的柱状图）
fig_010.png          <->  fig_016.png              （两张不同的组图）
```

原因：这类图有**长直线**（趋势线、坐标轴）。SIFT 会沿直线提取出大量方向一致
的特征，RANSAC 能轻易拟合出一个自洽的仿射变换；而长直线本身没有判别力，
于是两两之间都能"匹配上"。master 的 SIFT 验证层缺一道**退化检测**。

main 线的 v3 里正好有这道检查（`dedup/features.py::_degenerate_line`）——
用内点协方差的最小/最大特征值之比判定"内点是否退化成一条线"。
加上它之后，v3 在同一套数据上的误报是 **0**。

> 这条经验是通用的：**内点数多不等于匹配可信**，还要看内点在二维上是否铺开。

另外，master 的 CMFD 虽然 README 写着"输出源/粘贴区域坐标"，实测
`region1` / `region2` **全部为 None**，20 张组图里只产出 2 条、且无坐标，
所以区域级评分是 0/20。

---

## 4. 从 master 整合进 main 的东西

| 能力 | 来源 | 整合方式 |
|---|---|---|
| **子图/裁剪检测** | `detectors/subimage.py` | 移植为 `dedup/subimage.py`，修掉可扩展性问题（见下） |
| **边缘重叠/拼接检测** | `detectors/edge_overlap.py` | 移植为 `dedup/edge_overlap.py`，加了宽高比守卫 |
| **16bit TIF 支持** | `core/loader.py` 的思路 | 并入 `crossfile._load_gray`，百分位拉伸替代 8bit 截断 |
| 原生 C++ 内核 | `cpp/dedup_core.cpp` | **保留在归档 tag 中未启用**，理由见第 6 节 |

### 整合时必须修的一个可扩展性问题

master 的 `subimage` 候选是**按面积比枚举**的，本质是 O(n²)：

```
1200 张图 → 面积比 ≥1.5 的候选对 = 60034 个
```

任何上限（我一开始取 2000）都只能覆盖前 3%，**绝大多数真裁剪根本轮不到**，
所以能力形同虚设 —— 实测 1200 张时召回毫无提升。

改法：**子窗口哈希**。给每张图预先算好 6 个子窗口的 pHash；若 B 是 A 的裁剪，
则 B 的整图哈希应当接近 A 的**某个子窗口哈希**。于是候选生成的规模与 n 近似线性：

| 1200 张 | 面积比枚举 | 子窗口哈希 |
|---|---|---|
| 候选数 | 60,034 → 截断 2000 | **数百** |
| 召回 | 0.825 | **0.955** |
| 检测耗时 | 19.4 s | **2.46 s** |

顺带也修了另一个坑：每个候选对都重新读盘 + 缩放，仅靠缓存就把 120 张的
检测从 22.5s 压到 3.6s；再叠加"粗层不过就早退"，最终降到 0.10s。

---

## 5. C# / GPU 的评估结论（更新版）

最初的判断（不值得换语言）在看过 master 之后需要**部分修正**：

- master 确实写了 `cpp/dedup_core.cpp`（517 行，C++17 + OpenCV），
  提供 `hash` / `mih` / `cmfd` / `template` 四个子命令，exe 存在即自动启用、
  缺失自动回退纯 Python。**这是一个合理的架构**，不是过度设计。
- 但实测下来，**它想加速的那部分已经不是瓶颈**：整合版 v4 在 120 张图上
  检测层只要 **0.10s**，1200 张 **2.46s**，其中绝大头是子图模板匹配。
- 真正的重活（JPEG 解码、SIFT、`warpAffine`、`matchTemplate`）本来就跑在
  OpenCV 的 C++ 里，Python 只是调度层。

**修正后的结论**：
1. **现阶段不必启用 C++ 内核** —— 瓶颈不在那里，引入 exe 分发（60 个 DLL）
   与构建链（VS Build Tools + conda）的代价换不来可测收益。
2. 保留它是**合理的未来选项**：如果日后要处理 TB 级 WSI 全切片目录，
   "解码 + 哈希 + CMFD"这一段会成为新瓶颈，届时按 master 已有的接口接回去即可。
3. **GPU（V100）仍然不必要**：逐对比较的是几百像素的小图，
   PCIe 传输开销大于计算收益。

---

## 6. 基准方法

合成基准（`bench/gen_dataset.py`，固定种子可复现）：

- **跨文件集** `bench/data/cross/`：24 case × 5 张。每 case 注入 1 种已知变换，
  同 case 其余图为同风格难负样本。覆盖 15 种变换与 3 种命名风格
  （含"同基名换通道后缀"这一 v2 会整对跳过的盲区）。
- **图内集** `bench/data/intra/`：20 张 2×3 组图，各注入 1 处已知复用，
  覆盖 10 类关系，真值给出**像素级区域坐标**。
- **规模集** `bench/data_scale/`：200 case × 6 张 = 1200 张。

评分：跨文件按 pair 级 P/R/F1；图内**额外**按区域级 P/R/F1 + 定位 IoU（阈值 0.5）
—— 「找到」不算数，「框对位置」才算命中。

> 公平性说明：本基准的负样本由**同一生成器换随机种子**产生（如两张不同的 blot），
> 属于刻意加难的设置。master 那条线在自己 README 里报告的是
> 「基底间误报 0 (0/2415)」——在它自己的数据上表现良好。
> 第 3 节分析的"长直线退化"是真实存在且可复现的失效模式，
> 但**绝对数字不应跨数据集直接外推**。

---

## 7. 复现方式

```bash
# 取出 master 那条线做对照
git worktree add ../image_dedup_master archive/master-v3-2026-08

# 生成基准（固定种子）
python bench/gen_dataset.py --out bench/data --clean
python bench/gen_dataset.py --out bench/data_scale --cross-cases 200 --cross-negatives 4 --intra-figures 0

# 四条实现同台
python bench/run_bench.py --detector v2  --out bench/results/v2_baseline.json
python bench/run_bench.py --detector v3  --out bench/results/v3pure.json   # main-v3 原貌
python bench/run_bench.py --detector v4  --out bench/results/v4_120.json   # 整合版
python bench/run_bench.py --detector v3m --out bench/results/v3m.json      # master-v3.0

python bench/compare.py --title "四线对比" --results bench/results/v2_baseline.json \
    bench/results/v3pure.json bench/results/v3m.json bench/results/v4_120.json
python bench/analyze_result.py bench/results/v4_120.json   # 误报/漏报明细
```

---

## 8. 已知短板

1. **整图 180° 对称**：RANSAC 会拟合出全局旋转模型，内点撒满整图，
   区域包围盒退化成整图后被自映射过滤器丢掉 —— 图内旋转 180° 召回 0/2。
   补救路径（`salvage_self_maps`）已实现但默认关闭（实测得不偿失）。
2. **brightness 类仍有 1 例漏报**（120 张）——低信息量图上的亮度偏移。
3. **基准是合成的**：纹理统计与真实 blot/切片照片仍有差距。
   上线前建议用真实论文图再跑一轮。
4. **master 的 AI 嵌入层（MobileNetV2 ONNX）未整合** —— 需要额外模型文件，
   且当前候选层已不是瓶颈。
