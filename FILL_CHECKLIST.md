# 论文数据回填清单(编程手 ↔ 论文手交接)

> 本文档是"编程手跑数据 → 论文手回填"的交接单。
> **占位符约定**:论文中 `\textbf{(待回填:……)}` 是给编程手留的数值空位,
> 求解脚本跑完后由**论文手**逐个替换;替换时**不要删除占位符周边文字**,只换数字。
> **图表约定**:论文中 `% 待新增表` / `% 待新增图` 注释处缺结果表/图,注释内已写好可用的
> LaTeX 骨架(数值占位为 `0.xxxx`,百分比类为 `x.xx`);编程手出数出图后,
> 论文手取消注释、把 `0.xxxx` 替换为真实数值,再在正文里补一句结论(替换"待回填:一句结论")。
> **行号注意**:正文段落重排后行号会偏移,下列"论文落点"以表/图 label 为准。

---

## 0. 总览

| 脚本(编程手创建) | 输出内容 | 论文落点 |
|---|---|---|
| `code/src/solve_q1.py` | 问题一:合成负样本集、最优阈值 τ*、分类指标、ROC/F1(τ) 曲线数据 | `04` §问题一"求解结果":τ* 数值、表 `tab:q1-cls`、图 `fig:q1-metrics` |
| `code/src/solve_q2.py` | 问题二:检测/分割指标、训练损失、PR/混淆/可视化图 | `04` §问题二"求解结果":表 `tab:q2-det`、`tab:q2-seg`,图 `fig:q2-loss/pr/confusion/visual` |
| `code/src/solve_q3.py` | 问题三:鲁棒性/对比/消融/top4 实验数值与图 | `04` §问题三"求解结果":表 `tab:q3-robust/compare/ablation/top4`,图 `fig:q3-robust/sens/gradcam` |
| `code/src/solve_predict.py` | 演示提交文件 test_result.csv(413 验证集) | `submission/test_result.csv`(随 make pack 打包) |
| `code/src/fig_data_dist.py` | fig_data_dist.png(类别分布+单图实例数分布) | `03` 数据概览:图 `fig:data-dist` |
| `code/src/fig_data_scale.py` | fig_data_scale.png(bbox 尺度分布) | `03` 数据概览:图 `fig:data-scale` |

**数据**:已下发,全队都有。编程手开工时自行把数据集放入 `code/data/` 与 `data/`(结构与读法见 `data/README.md`)。
**公共约定**:
- 训练与划分固定随机种子(写入脚本注释);所有指标保留 4 位小数(百分比类保留 2 位)。
- 官方测试集(413 张,带标注)用作**验证集**;训练集 3300 张全为有残损图(**无负样本**,勿当二分类训练集用)。
- 图统一由 `fig_*.py` 生成(`make fig`);新建 fig 脚本后把脚本名加进 Makefile 的 `FIG_SCRIPTS`,并删除示例 `fig_example.py`。
- 训练环境:本机 NVIDIA GPU,检测/分割用 ultralytics(YOLOv8-seg)库,无需手写网络。

---

## 1. solve_q1.py —— 问题一(对应 04 §问题一)

- [ ] 合成负样本:从训练/验证图像裁取不含任何标注框的区域(裁剪窗口避开标注框并外扩边距,降低框外残损混入风险),随机拼合 640×640 整图,抽检剔除拼接伪影与框外残损("不含标注框"≠"不含残损"),建议共 800 张、调参半/评估半各 400,存 `code/data/syn_neg/`(验证集 413 张全为正样本,无真实负样本,此步必须做)
- [ ] 加载 solve_q2.py 训练好的检测器,对训练期验证正样本(330 张,与 solve_q2.py 的 10% 划分一致)、验证集正样本(413)与合成负样本两半推理
- [ ] 在调参集(330 正 + 调参半负)上扫描 τ∈{0.05,0.10,…,0.95},按 $f(I)=\mathbf{1}\{N(I;\tau)\ge 1\}$ 计算每个 τ 的 $F_1$;评估集(413 正 + 评估半负)仅用于报告,不参与选 τ
- [ ] 输出 τ* = argmax $F_1$(调参集),并在评估集上报告 τ* 下的 Acc/P/R/F1/AUC;ROC 得分用 $S(I)=\max_i s_i$(无框记 0)
- [ ] 输出 ROC 曲线(评估集)与 $F_1(\tau)$ 曲线(调参集)数据(供 fig_q1_metrics.py 出图)
- 论文落点:`04` §问题一"求解结果"——τ* 数值、表 `tab:q1-cls`、图 `fig:q1-metrics`

## 2. solve_q2.py —— 问题二(对应 04 §问题二,核心脚本)

- [ ] ultralytics YOLOv8-seg 训练:3 类、输入分辨率 1280、类别加权、P2 小目标检测头、强数据增强(建议 yolov8s-seg 起步,兼顾精度与速度)
- [ ] 训练验证划分:从 train 3300 划 10% 作训练期验证(早停/选模型),413 只在最终评估用一次,避免选择泄漏
- [ ] 验证集评估:per-class AP@0.5 与 AP@0.5:0.95、$AP_s$/$AP_m$/$AP_l$、per-class Dice 与 mIoU(检测指标按**全框口径**,不做 top-4 截断;top-4 口径在问题三单独考察)
- [ ] 产出:训练损失收敛曲线、三类 PR 曲线、混淆矩阵(3 类+背景,含漏检/误检行列)、检测+分割可视化样例(随机抽 4 张图,叠加框+掩码+类别)
- [ ] NMS 后处理 + top-4 筛选(排序方式默认按面积,见问题三 tab:q3-top4)
- 论文落点:`04` §问题二"求解结果"——表 `tab:q2-det`、`tab:q2-seg`,图 `fig:q2-loss/pr/confusion/visual`

## 3. solve_q3.py —— 问题三(对应 04 §问题三)

- [ ] 扰动鲁棒性:亮度×0.7、亮度×1.3、对比度调整、高斯噪声、高斯模糊 5 种扰动 → 各扰动下 mAP@0.5、$F_1$ 与衰减率 Δ_j
- [ ] 对比实验(问题一):检测归约 vs 自建小 CNN vs 传统特征+SVM——复用 solve_q1.py 的合成负样本两半:自建分类器以训练正样本 + 调参半负样本训练,三家统一在评估集(413 正 + 评估半负样本)上对比(训练/调参与评估不重叠,防泄漏)
- [ ] 对比实验(问题二):YOLOv8n/s/m-seg 三档 → 表 `tab:q3-eff`(参数量、FPS、mAP@0.5);统一硬件(记录 GPU 型号)、batch=1、输入 1280
- [ ] 消融实验(逐项累加:自基准配置起依次加入):类别加权 / P2 头 / 1280 输入 / 强增强 → mAP@0.5、$AP_s$
- [ ] top-4 排序对比(**每图≤4 框评测口径**):按面积(严重度代理) vs 按置信度 → mAP@0.5
- [ ] 灵敏度:τ_nms 扫描、输入分辨率 640/960/1280 → 出图数据
- [ ] Grad-CAM:对验证集样例生成激活热力图(3 类各 1-2 张)
- 论文落点:`04` §问题三"求解结果"——表 `tab:q3-robust/compare/ablation/top4`,图 `fig:q3-robust/sens/gradcam`

## 4. solve_predict.py —— 提交结果文件(演示)

原赛题除论文外还需提交测试集预测结果 `test_result.csv`。作业语境下主交付物是论文,
但保留该脚本可证明模型端到端可用;若老师发布真实无标注测试集,换图像目录重跑即可。

- [ ] 格式:`image_id,class_id,x_center,y_center,width,height`,坐标为归一化值(相对图像宽高,0~1)
- [ ] 每张图最多输出 4 个框(官方规则),排序默认按面积(严重度代理)
- [ ] 先用 413 张带标注验证集生成演示版,存 `submission/test_result.csv`(随 make pack 打包)

---

## 5. fig 脚本清单(全部纳入 make fig)

| 脚本 | 生成图 | 内容要求 | 论文落点 |
|---|---|---|---|
| `fig_data_dist.py` | fig_data_dist.png | 左:三类残损实例数柱状图;右:单图实例数直方图 | `03` 图 `fig:data-dist` |
| `fig_data_scale.py` | fig_data_scale.png | bbox 面积占比直方图(对数轴) | `03` 图 `fig:data-scale` |
| `fig_q1_metrics.py` | fig_q1_metrics.png | 左:ROC 曲线(评估集);右:$F_1(\tau)$ 曲线(调参集,标注 τ*) | `04` 图 `fig:q1-metrics` |
| `fig_q2_loss.py` | fig_q2_loss.png | 训练/验证损失收敛曲线 | `04` 图 `fig:q2-loss` |
| `fig_q2_pr.py` | fig_q2_pr.png | 三类 PR 曲线 | `04` 图 `fig:q2-pr` |
| `fig_q2_confusion.py` | fig_q2_confusion.png | 混淆矩阵热力图(3 类 + 背景) | `04` 图 `fig:q2-confusion` |
| `fig_q2_visual.py` | fig_q2_visual.png | 4 张样例:原图+预测框+掩码 | `04` 图 `fig:q2-visual` |
| `fig_q3_robust.py` | fig_q3_robust.png | 各扰动下 mAP@0.5 柱状图(含基准) | `04` 图 `fig:q3-robust` |
| `fig_q3_sens.py` | fig_q3_sens.png | 左:τ_nms→mAP;右:分辨率→mAP/FPS(双轴) | `04` 图 `fig:q3-sens` |
| `fig_q3_gradcam.py` | fig_q3_gradcam.png | Grad-CAM 热力图样例(原图+热区叠加) | `04` 图 `fig:q3-gradcam` |

**注意**:新建脚本后把脚本名加进 Makefile 的 `FIG_SCRIPTS` 行,并删除示例 `fig_example.py`。

---

## 6. 论文手新增呈现物汇总(编程手跑完数据后要做)

| 新增内容 | 位置 |
|---|---|
| 各问结果表/图(取消 `% 待新增` 注释,替换 `0.xxxx` / `x.xx`) | `04` 各小节"求解结果"、`03` 数据概览 |
| 各"求解结果"段结论句(替换"待回填:一句结论") | `04` 各小节 |
| τ* 等正文内数值(替换 `\textbf{(待回填:……)}`) | `04` §问题一"求解结果" |
| 模型检验/灵敏度分析结论句(替换 `\textbf{(待回填:……)}`,引 04 表/图,不重复列表) | `06` 各小节 |
| 摘要(每问一段结果数字) | `main.tex` 摘要(仍是"……"占位) |
| 参考文献补真实文献 | `08-references.tex`(两条示例占位) |
| 附录核心代码 | `09-appendix.tex` |
| 图:上述 fig 脚本全部跑通 → `make fig` | `code/src/*_fig.py` |

---

## 7. 回填完成后的自检清单(论文手)

1. `grep -rn "待回填" paper/sections/` 应为空;`grep -rn "0\.xxxx\|x\.xx" paper/sections/` 应为空;`main.tex` 摘要中的"……"占位已全部替换
2. 各节结论句与结果表数字一致(如 top-4 结论与表 `tab:q3-top4`、灵敏度结论与图 `fig:q3-sens` 一致)
3. 表中数值口径一致:mAP 统一 @0.5 与 @0.5:0.95 两列;分割统一 Dice/mIoU;衰减率统一按 $\Delta_j=(y^{clean}-y^{pert})/y^{clean}\times100\%$ 计算;`tab:q2-det`(全框口径)与 `tab:q3-top4`(top-4 口径)区分清楚;`tab:q3-eff` 中 YOLOv8s 的 mAP 与 `tab:q2-det` 一致
4. 数据概览中的 3713/3300/413、8038/1066 实例、3934/946/3158 与 `fig_data_dist.png` 一致;问题一调参集/评估集的合成负样本数(调参半/评估半)与正文/表注一致
5. `make fig` 全部 fig 脚本跑通;`make pdf` 通过(无未定义引用 ??);`make pack` 后核对 submission/ 下 zip、md5、重命名 PDF
6. `test_result.csv` 抽查:每图 ≤4 框、坐标 ∈[0,1]、class_id ∈{0,1,2};若发布真实测试集,行数 = 测试图像数 ×(≤4)
