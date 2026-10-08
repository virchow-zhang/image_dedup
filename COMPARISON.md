# v2 → v3 → v4 升级与整合对比报告

> **结论（含反向验证后修正）**：
> 仓库里存在**两条互不相关的开发线**（`main` 与 `master`，无共同祖先），版本号还互相错位。
> 整合后的 v4 在**本仓库的基准上最好**（F1 0.979 / 0.977，误报 0），
> 但在 **master 自带的 14 类变换基准上 master 更好**（F1 0.851 vs 0.792）——
> **v4 并非无条件最优**。两者的强弱项是互补的，详见第 4 节。
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

## 5. 反向验证：把 v4 放到 master 自己的基准上

只做「master 跑我的基准」是单向的，会高估自己。补上反向验证
（`bench/make_master_bench.py`，调用 master 的 `tests/gen_synthetic.py`
生成 450 张 / 14 类变换，再转成统一评分格式）：

| 指标 | master-v3.0 | **v4** |
|---|---|---|
| 精确率 P | 0.821 | **1.000** |
| 召回率 R | **0.883** | 0.655 |
| F1 | **0.851** | 0.792 |
| TP / FP / FN | 2437 / 531 / 323 | 1808 / **0** / **952** |
| 总耗时 | 46.3 s | **11.9 s** |
| 峰值内存 | 739 MB | **79 MB** |

**在 master 自己的地盘上，master 的 F1 更高。** v4 赢在精确率（0 误报 vs 531）、
速度（3.9×）和内存（9.3×），输在召回（差 952 对）。

### v4 漏在哪：**变换的组合**

把 92 种「两两变换组合」的召回拉出来看，v4 有 23 种是 0 召回，全部集中在两类：

| 组合 | v4 召回 | master 召回 |
|---|---|---|
| `base` × `rot_free`（17° 任意角旋转） | 0/30 | 30/30 |
| `crop_60` × `rotate90` / `rotate180` / `flip_h` / `flip_v` | 0/30 | 0~1/30 |
| `flip_h` × `rot_free` | 0/30 | 0/30 |
| `brightness_70` × `copy_move` | 0/30 | — |

单独任何一种变换 v4 基本都能处理（旋转 90° 整数倍、镜像、缩放、裁剪都有专门通道），
但**两种变换叠加**就掉出候选集。根因是哈希类候选的固有限制：

```
实测 pHash 距离（256 位）
  exact_copy   0        ← 哈希层完全胜任
  rotate90     0
  scale_07     2
  rot_free    36        ← 超过默认候选阈值 24
  crop_60    118        ← 远超任何阈值
```

### 试过的两个修法，都没成功（负结果，如实记录）

**① 放宽候选阈值**（24 → 40 → 64）：

| 候选阈值 | R | F1 | FP |
|---|---|---|---|
| 24（原） | 0.638 | 0.779 | 0 |
| 40 | 0.655 | 0.792 | 0 |
| 64 | 0.655 | 0.792 | 1 |

只回收了 1.3% 召回 —— 说明**缺的不是阈值余量，是候选机制本身**。
（仍保留了 40 这个默认值：零误报代价换一点召回。）

**② 移植 master 的 ORB 词袋候选层**（`dedup/bow.py`，逐行对齐 master 的
`core/bow.py` + `_compute_bow_keys`）：

| 数据集 | 关 BOW | 开 BOW |
|---|---|---|
| master 基准 F1 | 0.792 | 0.792（**召回没涨**） |
| 我的 1200 张 P | 1.000 | **0.823**（41 个误报） |
| 我的 1200 张 检测耗时 | 3.0 s | **56.4 s**（18.7×） |

**结论：只移植候选层无法复现 master 的收益。** BOW 确实把那些组合变换的候选
放进来了，但它们在**判定层**被整图结构投票（pHash/dHash/SSIM 一致性）否决 ——
开 BOW 后仍有 23/92 种组合保持 0 召回。

这反过来说明：**master 的召回优势来自它的判定层**，即逐对 SIFT + 覆盖率判据
+ 边缘重叠 + 子图匹配，能凭「部分证据」判定；而 v4 的判定层要求整图结构一致。
要真正补齐，需要重做判定层而不是换候选索引。BOW 代码保留但**默认关闭**。

---

## 6. C# / GPU 的评估结论（更新版）

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

## 7. 基准方法

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

## 8. 复现方式

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

## 9. 已知短板与明确的强弱项

**v4 强在哪**
- 本仓库基准上 F1 最高（120 张 0.979 / 1200 张 0.977），且**误报 0**
- 唯一具备区域级定位能力的实现（图内复用 F1 0.722，定位 IoU 0.786）
- 最快最省内存：1200 张 3.0 s / 94 MB；同规模 master 为 44.7 s / 739 MB

**v4 弱在哪（不回避）**
1. **变换的叠加**：`裁剪∘旋转`、`镜像∘任意角旋转` 这类组合召回为 0。
   master 在这类上更强（F1 0.851 vs 0.792）。补齐必须重做**判定层**
   （接受部分证据），换候选索引没用 —— 已实测验证（第 5 节）。
2. **整图 180° 对称**：RANSAC 会拟合出全局旋转模型，内点撒满整图，
   区域包围盒退化成整图后被自映射过滤器丢掉 —— 图内旋转 180° 召回 0/2。
   补救路径（`salvage_self_maps`）已实现但默认关闭（实测得不偿失）。
3. **brightness 类仍有 1 例漏报**（120 张）——低信息量图上的亮度偏移。
4. **基准是合成的**：纹理统计与真实 blot/切片照片仍有差距，
   且两套基准难度**不可横比**（我的负样本是同风格刻意加难，
   master 的负样本只有随机基底对）。上线前建议用真实论文图再跑一轮。
5. **master 的 AI 嵌入层（MobileNetV2 ONNX）未整合** —— 需额外模型文件。

**怎么选**
- 要**低误报 + 区域定位 + 交互报告** → v4
- 要**极限召回**（尤其叠加变换）、愿意人工复核误报 → master 那条线
- 两者互补。真正的下一步是把 master 的**判定层**并入 v4，而不是二选一。
