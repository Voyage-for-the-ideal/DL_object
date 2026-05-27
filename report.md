# 深度学习基础大作业实验报告

A 股趋势预测与模拟交易——档位 C Transformer 架构(TFNE-C)

【组员 1 姓名】(【学号 1】) 【组员 2 姓名】(【学号 2】) 【组员 3 姓名】(【学号】 2026 年 5 月

方案代号:TFNE-C (TFT-lite + FT-Transformer + NewsLoRA, 三 seed 秩融合)

配置:config.server_c2_72h.yaml(Top3000 / 3 seeds / 72h GPU 预算)

基线对照:档位 A RGNE-Lite(见 基线报告.tex)

代码路径:code/; 入口 scripts/run_server_c2_72h.py

硬件: USTC 算力平台 NVIDIA RTX 5090 (31.4GB); conda ai25

作业编号:train_7945；完成时间 2026-05-24 15:04(UTC+8)

随机种子:42,123,456

## 摘要

本实验在档位 A 基线 (RGNE-Lite: GRU + TabMLP + NewsHash) 同一套数据与策略框架上, 将三个子模型升级为深度学习 Transformer 族: 时序分支采用 TFT-lite (双层 LSTM + 时序注意力 + 静态变量嵌入), 截面分支采用 FT-Transformer (特征 Token 化 +4 层自注意力), 新闻分支采用 Chinese-RoBERTa + LoRA 微调。三模态各训练 3 个随机种子, 经 Spearman ICIR 定权秩融合得到统一 alpha，再映射为目标权重组合并完成 \( \mathrm{T} + 1 \) 历史回测。

服务器实验采用流动性 Top 3000、序列窗口 60 日、2019-01 至 2026-05 样本。 训练截止 2025-03-31 (含 2025Q1); 验证集为 2025-04-01 至 2026-05-20 的赛前 holdout (273 个交易日), 全部 IC 与回测指标均在此验证集上报告; FDL2026 模拟赛(2026-06-01 至 06-10)视为样本外，不参与历史指标。融合后在验证集全市场 mean IC 为 0.077、ICIR 为 0.794 ; 历史回测总收益 43.5%、复利年化 CAGR 39.6%、夏普 1.28、最大回撤 -27.5%(期末净值 143.5 万元)。同口径档位 A 基线 (Top500、同验证窗) mean IC 0.016、Sharpe 1.53 . 总收益 32.2%；本方案排序指标 (IC/ICIR) 显著提升；组合层指标因股票池规模与策略参数 (n_max=30 vs 50) 不同，不宜仅凭 Sharpe 高低判定优劣。

关键词:Transformer；TFT；FT-Transformer；LoRA；多模态融合；IC/ICIR；历史回测

## 目录

1 任务背景与方案定位 5

1.1 作业背景与问题形式化 5

1.2 档位 A 基线 vs 档位 C 本实验 5

1.3 本实验配置说明 6

1.4 产物完整性说明 6

2 评价指标定义与计算公式 7

2.1 标签与符号约定 7

2.2 预测能力指标 7

2.2.1 日度 IC (Information Coefficient) 7

2.2.2 平均 IC (mean IC) 7

2.2.3 IC 标准差 (std IC) 8

2.2.4 ICIR (IC Information Ratio) 8

2.2.5 Top 10% 内 IC (top quantile IC) 8

2.2.6 多空 spread (Long-Short Spread) 8

2.2.7 策略 Top-K IC (IC@K, strategy_topk) 9

2.2.8 Top- \( K \) 命中率 (hit_rate_topk) 9

2.2.9 Top- \( K \) 内部分 spread (within_topk_spread) 10

2.2.10 信号层组合收益(signal_portfolio，无交易成本) 10

2.2.11 相对沪深 300 指标 (compute_relative_metrics) 10

2.2.12 融合权重 ICIR(模态定权，补充) 10

2.3 组合回测指标 11

2.3.1 总收益率 (Total Return) 11

2.3.2 复利年化收益率 (CAGR) 11

2.3.3 算术年化收益率 (Annualized Arithmetic Return) 11

2.3.4 年化波动率 (Annualized Volatility) 11

2.3.5 夏普比率 (Sharpe Ratio) 12

2.3.6 最大回撤 (Maximum Drawdown, MDD) 12

2.4 指标汇总表:档位 A vs 档位 C2 (验证集) 13

2.5 读数提醒:IC 与组合收益为何可能「不一致」 14

3 数据处理与特征工程 14

3.1 整体流水线与数据形态变化 14

3.2 数据来源与使用字段 15

3.3 第一步:构建面板 (01_build_panel.py) 15

3.4 第二步:特征计算与 GKX 变换 (02_make_features.py) 16

3.5 第三步:新闻链指 (02b_embed_news.py) 17

3.6 第四步:滑动窗口样本(DailyStockDataset) 17

3.7 时间划分与防泄露 18

4 模型设计与方法论 18

4.1 总体架构 TFNE-C 19

4.2 模态 A: TFT-lite(时序分支) 19

4.3 模态 B: FT-Transformer(截面分支) 20

4.4 模态 C: NewsLoRA (新闻分支) 20

4.5 训练目标与优化细节 21

4.6 多 seed 集成与秩融合 22

5 交易策略与回测设置 22

5.1 与作业 PDF 及档位 A 的关系 23

5.2 可交易 universe: 哪些股票允许买? 23

5.3 每日策略计算: target_weights 分步说明 23

5.4 策略合理性分析 25

5.5 回测引擎:整体流程 26

5.6 单日回测时间线 (T+1 无未来信息) 27

5.7 调仓逻辑 rebalance_day 详解 27

5.8 回测配置与验证集设置 28

5.9 回测过程观察(基于 equity_curve.csv) 28

5.10 从目标权重到模拟下单 29

6 实验结果与对比分析 29

6.1 预测能力: IC / ICIR / Spread 29

6.2 策略对齐指标与相对大盘: A vs C2 详析 30

6.3 历史回测 (验证集) 33

6.4 融合权重与模态贡献 35

6.5 相对档位 A 基线的综合对比 36

6.6 IC 与组合收益的关系(本实验数据) 37

7 更新数据与最新策略预测 37

7.1 流水线总览 37

7.2 本机命令 (Windows PowerShell) 37

7.3 注意事项 38

8 规范性与可复现性

目录

9 总结与反思 39

9.1 主要结论 39

9.2 回测现实约束: 已考虑什么、还缺什么 39

9.3 若进一步加入缺失约束: 如何做、有何意义 41

9.4 现实约束对报告结论的影响 (定性) 42

9.5 局限与后续工作 42

参考文献 42

10 主要产物路径 43

主要产物路径 43

## 1 任务背景与方案定位

### 1.1 作业背景与问题形式化

与档位 A 相同，任务为日频横截面排序:每个交易日 \( t \) 对股票池内个股 \( i \) 构造特征 \( {X}_{i, t} \) ,预测下一日相对沪深 300 的超额收益

\[
{y}_{i, t} = {r}_{i, t + 1} - {r}_{t + 1}^{\text{ mkt }}, \tag{1}
\]

模型输出分数 \( {s}_{i, t} \) ，主评价指标为 Spearman IC。策略仅在头部 quantile 选股，因此除全市场 IC 外, 还报告 Top 10% 内 IC 与多空 spread (定义同档位 A 基线报告)。

### 1.2 档位 A 基线 vs 档位 C 本实验

档位 A (RGNE-Lite) 在本机 RTX 3060 6GB 上完成,采用 Top 500 股票池、 \( L = {20} \) 日窗口、GRU/MLP/NewsHash 三模态; 与档位 C 共用同一验证窗 (2025-04-01 至 2026- 05-20, 273 日)，验证集 mean IC 0.016、Sharpe 1.53、总收益 32.2%。本实验(TFNE-C) 在 USTC 5090 服务器上扩展规模与模型复杂度，核心对比如表 1。

表 1: 档位 A 基线 (RGNE-Lite) vs 档位 C 本实验 (TFNE-C)

<table><tr><td>维度</td><td>档位 A(基线)</td><td>档位 C(本实验)</td></tr><tr><td>方案代号</td><td>RGNE-Lite</td><td>TFNE-C</td></tr><tr><td>时序模型</td><td>GRU, \( L = {20} \)</td><td>TFT-lite (LSTM+Attn), \( L = {60} \)</td></tr><tr><td>截面模型</td><td>TabMLP</td><td>FT-Transformer (4 blocks)</td></tr><tr><td>新闻模型</td><td>HashingVectorizer</td><td>RoBERTa-wwm-ext + LoRA</td></tr><tr><td>股票池</td><td>Top 500(本机)</td><td>Top 3000 (服务器)</td></tr><tr><td>随机种子</td><td>42 (1 seed)</td><td>42, 123, 456 (3 seeds)</td></tr><tr><td>训练截止</td><td>2025-03-31</td><td>2025-03-31</td></tr><tr><td>验证区间</td><td>2025-04-01 — 2026-05-20</td><td>2025-04-01 — 2026-05-20</td></tr><tr><td>验证 IC 天数</td><td>274</td><td>273</td></tr><tr><td>数据截止</td><td>2026-05-20</td><td>2026-05-20</td></tr><tr><td>样本外</td><td></td><td>DL2026 模拟赛 2026-06-01 — 06-10(不出现在 IC/回测表)</td></tr><tr><td>硬件</td><td>RTX 3060 6GB</td><td>RTX 5090 31.4GB</td></tr></table>

对比说明:两档实验已对齐验证时段与训练截止，便于公平对照；股票池(Top500 vs Top3000)与持仓上限(A: n_max=30；C: n_max=50)仍不同，表 1 主要用于说明方法升级路径；第 6 节给出同口径 valid 数值对照。

### 1.3 本实验配置说明

配置文件 config.server_c2_72h.yaml 核心设定:

---

	- artifacts_tag: tier_c2;

- liquidity_top_k: 3000;

---

- seeds: [42, 123, 456], 各模态独立训练后以 seed 内均值参与融合;

- TFT/FTT max_epochs: 18, patience: 6;

- News: 8 万子采样、4 epoch; predict_news_from_checkpoint=true(增量 predict 可用);

- amp=false, tf32=false (Top 全量 TFT 数值稳定优先)。

评价口径(情况 A):配置 test_end=null，不设独立 test 段；05/06 脚本仅在 valid 上输出 IC 与回测, 报告正文只引用 valid 数字。FDL2026 模拟赛 (2026-06-01 至 06-10)视为真实样本外，仅通过 07_predict_latest 生成调仓单，不参与历史 IC/回测表格。

### 1.4 产物完整性说明

本地 code/artifacts/tier_c2/ 已包含评估闭环产物:

- metrics/tier_c2/ic_summary.json -IC/ICIR/spread;

- backtest/tier_c2/valid/ 一净值与回测指标;

- predictions/tier_c2/fusion_weights.json、ensemble_scores.parquet;

- orders/tier_c2/orders_20260520.csv 一最新调仓;

- features/feature_meta.json 一特征列定义;

- checkpoints/tier_c2/\{tft, ftt, news\}/seed_*/best.pt -9 个 checkpoint (News 约 410MB/个)。

章节安排。 第 2 节先给出全部评价指标定义与 A/C2 验证集对照表; 第 6 节再展开实验结果分析。

## 2 评价指标定义与计算公式

本节置于全文前部(紧接任务背景)，先统一符号、公式与 A/C2 对照口径；后续数据处理、模型、策略与实验结果中的数字均引用本节。IC@K 等策略对齐指标中的 \( K = \mathrm{n}\_ \max \text{ 、 }{Q}_{0.90} \) 含义见第 5 节。验证集汇总见表 2; 数值机理详析见第 6 节。

### 2.1 标签与符号约定

- \( {s}_{i, t} \) : 股票 \( i \) 在交易日 \( t \) 收盘后得到的融合 score (模型输出);

- \( {y}_{i, t} = {r}_{i, t + 1} - {r}_{t + 1}^{\text{ mkt }} \) : 下一日相对沪深 300 的超额收益(标签)；

- \( {\mathcal{U}}_{t} \) : 第 \( t \) 日有效样本集合 (代码要求 \( \left| {\mathcal{U}}_{t}\right|  \geq  {30} \) );

- \( {P}_{t} \) : 组合日频净值 (初始 \( {P}_{0} = {10}^{6} \) 元);

- \( {r}_{t} = {P}_{t}/{P}_{t - 1} - 1 \) : 组合日收益率;

- \( n \) : 有效交易日数 (本实验验证集 \( n = {273} \) )。

### 2.2 预测能力指标

#### 2.2.1 日度 IC (Information Coefficient)

定义: 在每个交易日 \( t \) 的横截面上,衡量模型分数 \( {s}_{i, t} \) 与次日超额收益 \( {y}_{i, t} \) 的秩相关强度。本实验采用 Spearman 相关系数 (对极端值更稳健)。

公式:

\[
{\mathrm{{IC}}}_{t} = \operatorname{Spearman}\left( {{\left\{  {s}_{i, t}\right\}  }_{i \in  {\mathcal{U}}_{t}},{\left\{  {y}_{i, t}\right\}  }_{i \in  {\mathcal{U}}_{t}}}\right)  = \operatorname{Corr}\left( {\operatorname{rank}\left( {s}_{i, t}\right) ,\operatorname{rank}\left( {y}_{i, t}\right) }\right) . \tag{2}
\]

现实意义:IC 回答「今天分高的股票，明天是否更可能跑赢大盘」。IC \( {}_{t} = 0 \) 表示排序与收益无关; \( {\mathrm{{IC}}}_{t} = {0.07} \) 表示弱正相关——在日频 A 股中 0.02-0.08 已属可用区间，不必期待 IC 接近 1 。IC 是评价「排序模型」最核心的指标，与按 score 调仓的策略目标直接一致。

#### 2.2.2 平均 IC (mean IC)

定义:对样本期内全部有效交易日的 IC 序列取算术平均。

公式:

\[
\overline{\mathrm{{IC}}} = \frac{1}{D}\mathop{\sum }\limits_{{t = 1}}^{D}{\mathrm{{IC}}}_{t} \tag{3}
\]

其中 \( D \) 为 \( \mathrm{{IC}} \) 有效天数(本实验验证集 \( D = {273} \) )。

## 2 评价指标定义与计算公式

现实意义: 反映模型在一段时期内平均排序能力。本实验验证集 \( \overline{\mathrm{{IC}}} = {0.077} \) ,约为同口径档位 A 基线 (0.016) 的 4.8 倍, 说明 Transformer 升级显著改善了横截面 alpha 水平。

#### 2.2.3 IC 标准差 (std IC)

定义:IC 日序列的样本标准差 (ddof=1)。

公式:

\[
{\sigma }_{\mathrm{{IC}}} = \sqrt{\frac{1}{D - 1}\mathop{\sum }\limits_{{t = 1}}^{D}{\left( {\mathrm{{IC}}}_{t} - \overline{\mathrm{{IC}}}\right) }^{2}}. \tag{4}
\]

现实意义: 衡量 IC 的日际波动。 \( {\sigma }_{\mathrm{{IC}}} \) 越大,说明模型在某些交易日几乎失效、某些日又很强。本实验验证集 \( {\sigma }_{\mathrm{{IC}}} = {0.097} \) ,低于档位 \( \mathrm{A} \) 的 0.153,排序信号相对更稳。

#### 2.2.4 ICIR (IC Information Ratio)

定义:IC 均值与 IC 标准差之比, 又称 「IC 的信息比率」。

公式:

\[
\operatorname{ICIR} = \frac{\overline{\mathrm{{IC}}}}{{\sigma }_{\mathrm{{IC}}}}. \tag{5}
\]

现实意义: 类似 IC 的夏普比率 \( \mathrm{J} =  - 7 \) 极要求 IC 为正,还要求 IC 稳定。ICIR \( = {0.79} \) 意味着平均 IC 约为日际波动的 0.8 倍; ICIR 越高, 融合权重定权时该模态越可信。本实验验证集 ICIR= 0.794 ，显著高于档位 A 的 0.102。

#### 2.2.5 Top 10% 内 IC (top quantile IC)

定义: 仅在当日 score 位于最高 \( {10}\% \) 分位 \( \left( { \geq  {Q}_{0.90}}\right) \) 的股票子集上,重复计算 Spearman IC——与策略 q_min=0.90 对齐。

公式: 记 \( {\mathcal{H}}_{t} = \left\{  {i \in  {\mathcal{U}}_{t} \mid  {s}_{i, t} \geq  {Q}_{0.90}\left( {s}_{\cdot , t}\right) }\right\} \) ,

\[
{\mathrm{{IC}}}_{t}^{\text{ top }} = \operatorname{Spearman}\left( {{\left\{  {s}_{i, t}\right\}  }_{i \in  {\mathcal{H}}_{t}},{\left\{  {y}_{i, t}\right\}  }_{i \in  {\mathcal{H}}_{t}}}\right) ,\;{\overline{\mathrm{{IC}}}}^{\text{ top }} = \frac{1}{D}\mathop{\sum }\limits_{t}{\mathrm{{IC}}}_{t}^{\text{ top }}. \tag{6}
\]

现实意义:回答「在已经筛出的头部候选里，能否再精细排序」。本实验验证集 \( {\overline{\mathrm{{IC}}}}^{\text{ top }} = {0.014} \) ; 档位 A 同窗为 0.013。二者均偏低,策略更依赖头尾 spread 与 Top- \( K \) 内 spread，而非整个 10% 分位内的精细排序。

#### 2.2.6 多空 spread (Long-Short Spread)

定义: 每个交易日, score 最高 10% 股票组的平均超额收益, 减去 score 最低 10% 组的平均超额收益(未扣费、非真实对冲组合)。

公式:

\[
{\operatorname{spread}}_{t} = \underset{\text{ top }{10}\% \text{ 平均超额 }}{\underbrace{\frac{1}{\left| {\mathcal{T}}_{t}\right| }\mathop{\sum }\limits_{{i \in  {\mathcal{T}}_{t}}}{y}_{i, t}}} - \underset{\text{ bottom }{10}\% \text{ 平均超额 }}{\underbrace{\frac{1}{\left| {\mathcal{B}}_{t}\right| }\mathop{\sum }\limits_{{i \in  {\mathcal{B}}_{t}}}{y}_{i, t}}}, \tag{7}
\]

\[
\overline{\text{ spread }} = \frac{1}{D}\mathop{\sum }\limits_{{t = 1}}^{D}{\text{ spread }}_{t},\;\text{ 年化 spread } = \overline{\text{ spread }} \times  {252}. \tag{8}
\]

现实意义:直接度量 「高分组是否跑赢低分组」，比全市场 IC 更贴近 「只买头部」 的策略逻辑。本实验验证集日均 spread= 0.314%，年化约 79.0%——这是理论分组收益上限参考, 实际只做多组合会因成本、仓位约束、换手上限而低于该值。档位 A 同窗 long-short spread 为 -0.116%/日 (头尾分组在该窗口内略偏负), 与其窄池、标签噪声有关，但经策略层约束后回测仍可盈利(见表 2)。

#### 2.2.7 策略 Top-K IC (IC@K, strategy_topk)

定义: 仅在策略真实候选池内计算 Spearman IC: 先取 score \( \geq  {Q}_{0.90} \) ,再取 Top \( K \) ( \( K = \mathrm{n}\_ \max ,\mathrm{A} \) 为 30, C2 为 50)，在该 \( K \) 只子集上算 \( {\mathrm{{IC}}}_{t} \) 并平均。

公式: 记 \( {\mathcal{S}}_{t} = \operatorname{Top}K\left( \left\{  {i : {s}_{i, t} \geq  {Q}_{0.90}}\right\}  \right) \) ,

\[
{\operatorname{IC@K}}_{t} = \operatorname{Spearman}\left( {{\left\{  {s}_{i, t}\right\}  }_{i \in  {\mathcal{S}}_{t}},{\left\{  {y}_{i, t}\right\}  }_{i \in  {\mathcal{S}}_{t}}}\right) ,\;\overline{\mathrm{{IC@K}}} = \frac{1}{D}\mathop{\sum }\limits_{t}{\operatorname{IC@K}}_{t}. \tag{9}
\]

现实意义:比「全市场 IC」和「整个 Top 10% IC」更贴近 target_weights 的买入范围——回答「即将持有的 \( K \) 只里，能否再排好序」。验证集:A 的 \( \overline{\mathrm{{IC}}@\mathrm{K}} = \mathbf{0.{0303}} \) (K=30,274 日), C2 为 -0.0037 (K=50,273 日)。paradox: C2 全市场 IC 远高于 A，但 IC@K 反而略负，说明 Transformer 的 alpha 主要体现在「进 Top 池」而非「池内精细排序」; A 窄池 + 小 \( K \) 使 IC@K 更易为正。

#### 2.2.8 Top-K 命中率 (hit_rate_topk)

定义:每个交易日，在策略 Top- \( K \) 候选池内，次日超额收益 \( {y}_{i, t} > 0 \) 的股票占比， 再对日序列取平均。

公式:

\[
{\mathrm{{HR}}}_{t} = \frac{1}{\left| {\mathcal{S}}_{t}\right| }\mathop{\sum }\limits_{{i \in  {\mathcal{S}}_{t}}}\mathbf{1}\left\lbrack  {{y}_{i, t} > 0}\right\rbrack  ,\;\overline{\mathrm{{HR}}} = \frac{1}{D}\mathop{\sum }\limits_{t}{\mathrm{{HR}}}_{t}. \tag{10}
\]

现实意义:衡量「买入组合里有多少比例跑赢大盘」——不区分幅度，只看方向。验证集 A 为 \( \mathbf{{47}.0\% ,{C2}} \) 为 \( \mathbf{{50}.1\% } \) ,均接近随机 \( \left( {{50}\% }\right) \) ,说明单日方向预测仍难; 组合收益更多来自权重倾斜与多日累积，而非「每天都多数上涨」。

#### 2.2.9 Top- \( K \) 内部分 spread (within_topk_spread)

定义:在 Top-K 候选池内，按 score 排序后将池子对半分，比较前半(更高分)与后半的平均超额收益之差。

公式:

\[
{\mathrm{{ws}}}_{t} = {\bar{y}}_{{\mathcal{S}}_{t}^{\mathrm{{hi}}}} - {\bar{y}}_{{\mathcal{S}}_{t}^{\mathrm{{lo}}}},\;\overline{\mathrm{{ws}}} = \frac{1}{D}\mathop{\sum }\limits_{t}{\mathrm{{ws}}}_{t}. \tag{11}
\]

现实意义:直接对应「持仓内部:高分票是否比低分票赚更多」。验证集 A 年化 10.2%， C2 年化 21.8%(由 print_strategy_topk_metrics.py 读取 strategy_topk.json) ——C2 在策略尺度上的内部分化优于 A，与 IC@K 的负值并存(IC 看秩相关，spread 看均值差)。

#### 2.2.10 信号层组合收益(signal_portfolio，无交易成本)

定义:每个交易日对全截面 score 调用 target_weights 得到权重 \( {w}_{i, t} \) ，计算理论次日组合超额 \( \mathop{\sum }\limits_{i}{w}_{i, t}{y}_{i, t} \) ,再年化; 不含佣金、滑点、涨跌停与换手上限。

现实意义: 衡量「若每日按目标权重完美成交, 信号本身能解释多少超额」——是回测的上界参考, 不能替代 06_backtest.py。验证集 (strategy_topk.json): A 年化 -14.2%，C2 年化 +34.1%；而同窗扣费回测 A 为 +32.2%、C2 为 +43.5%。A 的 signal 与回测严重背离, 说明档位 A 启用了 risk_adjust 改变 score、且信号层未建模 T+1/成本/软换仓——读数时必须以回测为准，signal_portfolio 仅作诊断。

#### 2.2.11 相对沪深 300 指标(compute_relative_metrics)

定义: 将组合日收益 \( {r}_{t} \) 与基准 \( \left( {{000300}.\mathrm{{SH}}}\right) \) 日收益 \( {r}_{t}^{b} \) 对齐,报告超额与回归指标。

公式 (节选):

\[
{R}_{\mathrm{{exc}},\mathrm{{tot}}} = \frac{{P}_{T}}{{P}_{0}} - \frac{{B}_{T}}{{B}_{0}},\;\operatorname{IR} = \frac{\left( {\bar{r} - {\bar{r}}_{b}}\right)  \times  {252}}{\operatorname{std}\left( {{r}_{t} - {r}_{t}^{b}}\right) \sqrt{252}}, \tag{12}
\]

\[
{r}_{t} = \alpha  + \beta {r}_{t}^{b} + {\varepsilon }_{t}\; \Rightarrow  \;{\alpha }_{\text{ ann }} = \widehat{\alpha } \times  {252}. \tag{13}
\]

现实意义:回答「策略是否真跑赢大盘，而非仅随 beta 波动」。验证集同窗:沪深 300 总收益 24.8%; A 超额总收益 5.9%、IR 0.67、 \( {\alpha }_{\text{ ann }} \) 5.4%; C2 超额总收益 15.0%、 IR 0.74、 \( {\alpha }_{\text{ ann }} \) 9.0%、 \( \beta  = {1.33} \) (C2 市场暴露更高,解释其 MDD 更深)。

#### 2.2.12 融合权重 ICIR(模态定权，补充)

定义: 对模态 \( m \in  \{ A, B, C\} \) 分别计算验证集 IC 序列的 ICIR,再对非负 ICIR 归一化得到融合权重 \( {w}_{m} \) 。

公式:

\[
{\operatorname{ICIR}}_{m} = \frac{{\overline{\mathrm{{IC}}}}_{m}}{{\sigma }_{{\mathrm{{IC}}}_{m}}},\;{w}_{m} = \frac{\max \left( {{\mathrm{{ICIR}}}_{m},0}\right) }{\mathop{\sum }\limits_{k}\max \left( {{\mathrm{{ICIR}}}_{k},0}\right) }. \tag{14}
\]

## 2 评价指标定义与计算公式

现实意义: 自动降低弱模态 (如档位 A 的 NewsHash) 权重、提高强模态权重。本实验 \( {w}_{A} = {35.3}\% ,{w}_{B} = {38.1}\% ,{w}_{C} = {26.6}\% \) ,三模态均衡参与融合。

### 2.3 组合回测指标

设验证集净值序列为 \( {\left\{  {P}_{t}\right\}  }_{t = 0}^{n} \) ,日收益 \( {r}_{t} = {P}_{t}/{P}_{t - 1} - 1 \) 。

#### 2.3.1 总收益率 (Total Return)

定义:回测区间内，期末净值相对期初净值的增长比例。

公式:

\[
{R}_{\text{ tot }} = \frac{{P}_{T}}{{P}_{0}} - 1. \tag{15}
\]

现实意义: 区间实际盈亏,最直观。本实验验证集 \( {R}_{\text{ tot }} = \mathbf{{43}.5\% } \) (100 万 \( \rightarrow  {143.5} \) 万)，273 个交易日；档位 A 同窗 \( {R}_{\text{ tot }} = \mathbf{{32}.2\% } \) 。

#### 2.3.2 复利年化收益率 (CAGR)

定义:假设按相同复利速度持续一整年，由区间总收益外推得到的年化率。

公式:

\[
\mathrm{{CAGR}} = {\left( \frac{{P}_{T}}{{P}_{0}}\right) }^{{252}/n} - 1. \tag{16}
\]

现实意义: 便于跨不同长度区间比较, 但短样本会放大 headline 数字。本实验 CAGR = 39.6% (273 日); 档位 A 同窗 CAGR = 29.4%。

#### 2.3.3 算术年化收益率 (Annualized Arithmetic Return)

定义:日收益算术均值乘以 252 。

公式:

\[
{\mu }_{\text{ ann }} = \bar{r} \times  {252},\;\bar{r} = \frac{1}{n}\mathop{\sum }\limits_{{t = 1}}^{n}{r}_{t}. \tag{17}
\]

现实意义:夏普比率的分子，反映「典型一天收益放大到一年」的水平。本实验 \( {\mu }_{\text{ ann }} = {37.8}\% \) ,与 CAGR 39.6% 接近 (273 日样本较长,两者差异小于短样本情形)。

#### 2.3.4 年化波动率 (Annualized Volatility)

定义: 日收益标准差年化。

公式:

\[
{\sigma }_{\text{ ann }} = \operatorname{std}\left( {r}_{t}\right)  \cdot  \sqrt{252},\;\operatorname{std}\left( {r}_{t}\right)  = \sqrt{\frac{1}{n - 1}\mathop{\sum }\limits_{{t = 1}}^{n}{\left( {r}_{t} - \bar{r}\right) }^{2}}. \tag{18}
\]

2 评价指标定义与计算公式

现实意义: 衡量组合净值起伏剧烈程度——承担的风险尺度。本实验 \( {\sigma }_{\text{ ann }} = {29.6}\% \) ; 2025-04-07 单日 -12.15% 等极端日会抬升波动率。波动越高, 达到相同 Sharpe 所需的收益越高。

#### 2.3.5 夏普比率 (Sharpe Ratio)

定义: 单位风险所获得的超额收益 (相对无风险利率 \( {r}_{f} \) )。

公式:

\[
\text{ Sharpe } = \frac{{\mu }_{\text{ ann }} - {r}_{f}}{{\sigma }_{\text{ ann }}} = \frac{\bar{r} \times  {252} - {r}_{f}}{\operatorname{std}\left( {r}_{t}\right)  \cdot  \sqrt{252}}. \tag{19}
\]

本实验取 \( {r}_{f} = 0 \) 。

现实意义:综合评价「收益/风险」效率。本实验 Sharpe= 1.28; 档位 A 同窗 Sharpe= 1.53 略高,与 Top500 窄池、更低 MDD 及 \( \beta  \approx  {1.0} \) 有关。

#### 2.3.6 最大回撤 (Maximum Drawdown, MDD)

定义:净值从历史最高点至后续最低点的最大跌幅。

公式:

\[
{\text{ peak }}_{t} = \mathop{\max }\limits_{{\tau  \leq  t}}{P}_{\tau },\;{\mathrm{{DD}}}_{t} = \frac{{P}_{t} - {\text{ peak }}_{t}}{{\text{ peak }}_{t}},\;\mathrm{{MDD}} = \mathop{\min }\limits_{t}{\mathrm{{DD}}}_{t} \leq  0. \tag{20}
\]

现实意义:回答「最坏情况下从高点最多亏多少」。本实验 MDD = -27.5%；档位 A 同窗 MDD = -9.2%。C2 更深与 Top3000 广度、 \( \beta  = {1.33} \) 及 2025-04 系统性下跌有关。

### 2.4 指标汇总表:档位 A vs 档位 C2 (验证集)

表 2:评价指标验证集对照(2025-04-01 — 2026-05-20; 数据来自 ic_summary.json / backtest/*/valid/metrics.json)

<table><tr><td>类别</td><td>指标</td><td>档位 A</td><td>档位 C2</td></tr><tr><td rowspan="5">预测(全池)</td><td>全市场 mean IC</td><td>0.016</td><td>0.077</td></tr><tr><td>ICIR</td><td>0.102</td><td>0.794</td></tr><tr><td>IC 标准差 \( {\sigma }_{\mathrm{{IC}}} \)</td><td>0.153</td><td>0.097</td></tr><tr><td>Top 10% 内 IC</td><td>0.013</td><td>0.014</td></tr><tr><td>多空 spread(日)</td><td>-0.116%</td><td>0.314%</td></tr><tr><td rowspan="5">预测(策略对齐)</td><td>IC@K( \( K = \mathrm{n} \) _____max)</td><td>0.0303</td><td>-0.0037</td></tr><tr><td>IC@K 的 ICIR</td><td>0.146</td><td>-0.020</td></tr><tr><td>Top- \( K \) 命中率</td><td>47.0%</td><td>50.1%</td></tr><tr><td>Top- \( K \) 内 spread(年化)</td><td>10.2%</td><td>21.8%</td></tr><tr><td>信号层组合(年化，无成本)</td><td>-14.2%</td><td>34.1%</td></tr><tr><td rowspan="6">回测(组合)</td><td>总收益 \( {R}_{\mathrm{{tot}}} \)</td><td>32.2%</td><td>43.5%</td></tr><tr><td>CAGR</td><td>29.4%</td><td>39.6%</td></tr><tr><td>Sharpe</td><td>1.53</td><td>1.28</td></tr><tr><td>MDD</td><td>-9.2%</td><td>-27.5%</td></tr><tr><td>相对沪深 300 超额总收益</td><td>5.9%</td><td>15.0%</td></tr><tr><td>信息比率 IR(相对大盘)</td><td>0.67</td><td>0.74</td></tr><tr><td rowspan="2">相对大盘回归</td><td>年化 Alpha \( {\alpha }_{\text{ ann }} \)</td><td>5.4%</td><td>9.0%</td></tr><tr><td>Beta \( \beta \)</td><td>1.01</td><td>1.33</td></tr></table>

表 3:评价指标:公式与 C2 验证集数值(273 F)，节选)

<table><tr><td>指标</td><td>核心公式</td><td>现实意义</td><td>C2</td></tr><tr><td>mean IC</td><td>\( \frac{1}{D}\sum {\mathrm{{IC}}}_{t} \)</td><td>平均排序能力</td><td>0.077</td></tr><tr><td>ICIR</td><td>\( \overline{\mathrm{{IC}}}/{\sigma }_{\mathrm{{IC}}} \)</td><td>IC 稳定性</td><td>0.794</td></tr><tr><td>IC@K</td><td>IC@K</td><td>策略池内再排序</td><td>-0.004</td></tr><tr><td>Top- \( K \) 命中率</td><td>HR</td><td>池内跑赢大盘比例</td><td>50.1%</td></tr><tr><td>Top- \( K \) 内 spread</td><td>WS</td><td>池内高低分分化</td><td>0.086%/日</td></tr><tr><td>Top 10% IC</td><td>\( {\overline{\mathrm{{IC}}}}^{\mathrm{{top}}} \)</td><td>宽头部细分</td><td>0.014</td></tr><tr><td>多空 spread</td><td>\( {\bar{y}}_{\text{ top }} - {\bar{y}}_{\text{ bot }} \)</td><td>头尾分组</td><td>0.314%/日</td></tr><tr><td>总收益</td><td>\( {P}_{T}/{P}_{0} - 1 \)</td><td>区间盈亏</td><td>43.5%</td></tr><tr><td>Sharpe</td><td>\( {\mu }_{\mathrm{{ann}}}/{\sigma }_{\mathrm{{ann}}} \)</td><td>收益风险比</td><td>1.28</td></tr><tr><td>MDD</td><td>\( \min {\mathrm{{DD}}}_{t} \)</td><td>最大回撤</td><td>-27.5%</td></tr><tr><td>超额 IR</td><td>\( \operatorname{IR}\left( {r - {r}^{b}}\right) \)</td><td>相对大盘稳定超额</td><td>0.74</td></tr></table>

### 2.5 读数提醒:IC 与组合收益为何可能「不一致」

1. IC 衡量 Top3000 全体排序; 策略只用 Top 10% 且最多 50 只——更该看 spread 与回测 \( {R}_{\text{ tot }} \) ;

2. spread 年化 T9% 是未扣费、含做空底部的参考; 实际只做多组合 43.5% 总收益更低, 属正常;

3. IC@K 与全市场 IC 可能背离: C2 全池 IC 高但 IC@K 略负, 读数时须同时看表 2;

4. 档位 A 与档位 C2 已对齐验证时段; 对比时仍须注意股票池与 n_max 差异。

## 3 数据处理与特征工程

本节按代码实际执行顺序,说明「原始 CSV \( \rightarrow \) 模型可读张量」的全过程。档位 C 与档位 A 共用同一套 code/src/ 流水线, 差异在于: 本实验股票池扩至 Top 3000、时序窗口 \( L = {60} \) 、数据截止 2026-05-20,并在服务器上以 16 进程并行 prep。

### 3.1 整体流水线与数据形态变化

数据处理分三步，对应 scripts/01_build_panel.py、02_make_features.py、02b_embed_news.py

1. 建面板:逐日 merge 量价、基本面、资金流 CSV，构造标签 \( {y}_{i, t} \) ；

2. 做特征:计算 16 维时序 +9 维截面因子，逐日 GKX 变换，构造 60 日滑动窗口；

3. 处理新闻:全市场快讯链指到个股，供 NewsLoRA 使用。

---

数据形态变化 (本实验):

daily/YYYYMMDD.csv (单日全市场截面)

\( \Downarrow \) merge metric、moneyflow、ST 标记

	panel_daily.parquet(长表:每行 = 某股某日，Top3000 过滤后约 3000 只 \( \times  {1500} + \)

交易日)

	⇓ 技术指标 + GKX winsorize+rank

	features/panel_features.parquet (prep job 7922 完成后约 1.4 GB)

⇓ 新闻链指

news/panel_news.parquet

\( \Downarrow \) DailyStockDataset 预计算 \( L = {60} \) 序列

训练样本: 每个 (ts_code, trade_date) \( \rightarrow  \left( {{60},{16}}\right) \) 序列 + \( \left( {9\text{ , }}\right) \) 截面 + 文本 + 标签

---

图 1: 从原始 CSV 到训练样本的形态变化

### 3.2 数据来源与使用字段

数据来自课程云盘, 服务器路径 . ./data/, 使用字段与档位 A 一致, 本实验额外利用更长的 2025 年行情与新闻。

表 4:数据来源与用途

<table><tr><td>类型</td><td>路径</td><td>用途</td></tr><tr><td>日频量价</td><td>daily/YYYYMMDD.csv</td><td>开高低收、成交量、成交额</td></tr><tr><td>基本面</td><td>metric/YYYYMMDD.csv</td><td>PE、PB、换手率、市值等</td></tr><tr><td>资金流</td><td>moneyflow/YYYYMMDD.csv</td><td>大/中/小单买卖金额</td></tr><tr><td>新闻快讯</td><td>news/YYYYMMDD.csv</td><td>标题 + 正文, 链指到个股</td></tr><tr><td>ST 名单</td><td>stock_st/</td><td>剔除 ST</td></tr><tr><td>指数</td><td>market/000300.SH.CSV</td><td>沪深 300，算超额收益</td></tr><tr><td>基础信息</td><td>basic.csv</td><td>名称、行业、上市日期</td></tr></table>

### 3.3 第一步:构建面板 (01_build_panel.py)

(1)逐日读取与合并 对每个 trade_date，左连接 daily、metric、moneyflow，标记 ST, 纵向拼接为长表。每一行对应一个 (ts_code, trade_date) 观测。

(2)股票池过滤——Top 3000 与档位 A Top 500 相比，本实验在服务器上采用更宽覆盖:

- 剔除 ST、北交所 (.BJ);

- 上市满 60 个交易日;

- 20 日平均成交额と 5000 万元;

- 按全样本平均流动性排序, 保留 Top 3000 (prep 日志确认最终 3000 只)。

更宽股票池使 IC 统计更稳定 (验证集每日有效样本约 3000 只), 但也引入更多中小票噪声，对组合层 MDD 有放大效应 (见第 6 节)。

## (3)标签构造——无未来信息

\[
{y}_{i, t} = {r}_{i, t + 1} - {r}_{t + 1}^{\mathrm{{mkt}}},\;{r}_{i, t + 1} = \frac{{P}_{i, t + 1} - {P}_{i, t}}{{P}_{i, t}}. \tag{21}
\]

代码用 groupby(ts_code).close.pct_change().shift(-1) 实现:在 trade_date=t 行， 特征仅用 \( \leq  t \) 的信息，标签为 \( t \rightarrow  t + 1 \) 超额——与 \( \mathrm{A} \) 股「盘后出信号、次日成交」一致。

(4)资金流衍生 构造 net_mf_ratio(净流入/成交额)、elg_imb、lg_imb(大/特大单不平衡度),取值约在 \( \left\lbrack  {-1,1}\right\rbrack \) ,便于跨市值比较。

### 3.4 第二步:特征计算与 GKX 变换(02_make_features.py)

(1)时序技术指标(16 维，模态 A 输入) 对每只股票单独 rolling 计算，列名见 feature_meta.json:

表 5:时序特征含义(模态 A， \( L = {60} \) 窗口)

<table><tr><td>特征名</td><td>计算方式</td><td>直观含义</td></tr><tr><td>ret_1d/5d/.../60d</td><td>收盘价 pct_change</td><td>多尺度动量; 60 日 ret 为档位 \( \mathrm{C} \) 新增相对 A 的长周期项</td></tr><tr><td>vol_10d/20d/60d</td><td>日收益 rolling std</td><td>波动率</td></tr><tr><td>amp</td><td>(high – low)/close</td><td>振幅</td></tr><tr><td>vwap_dev</td><td>(close – vwap)/vwap</td><td>收盘价偏离均价</td></tr><tr><td>log_amount</td><td>\( \log (1 + \) 成交额)</td><td>流动性</td></tr><tr><td>net_mf_ratio 等</td><td>见上节</td><td>资金结构</td></tr><tr><td>turnover_rate_f</td><td>metric</td><td>换手</td></tr><tr><td>rsi_14</td><td>14 日 RSI</td><td>超买超卖</td></tr></table>

窗口从档位 A 的 \( L = {20} \) 扩展到 \( L = {60} \) (约一个季度),使 TFT-lite 能捕捉更长周期趋势与波动 regime; 代价是 prep 与训练样本量增大 (训练集约 420 万股-日行, 验证集约 81 万)。

表 6:截面特征含义(模态 B)

<table><tr><td>特征</td><td>来源</td><td>含义</td></tr><tr><td>turnover_rate_f, volume_ratio</td><td>metric</td><td>活跃度</td></tr><tr><td>pb, pe_ttm, pe_missing</td><td>metric</td><td>估值与缺失指示</td></tr><tr><td>log_mv</td><td>派生</td><td>规模因子</td></tr><tr><td>net_mf_ratio, elg_imb, lg_imb</td><td>moneyflow</td><td>当日资金结构</td></tr></table>

(2)截面特征(9 维，模态 B 输入)

(3)GKX 截面预处理 原则:禁止全样本 StandardScaler，否则引入未来分布信息。 对每个 trade_date、每个特征列独立执行 (src/preprocess.py, winsor_q=0.01):

1. Winsorize: 截断当日最高/最低 1% 极端值;

2. Rank \( \rightarrow  \left\lbrack  {-1,1}\right\rbrack \) : 在当日 Top3000 横截面内排序并线性映射。

小例子:2025-04-01 当天 3000 只股票的 pe_ttm, rank 最高者映射为 +1 ，最低者为 -1 。 GRU/MLP/Transformer 每天看到的输入尺度一致, 且仅使用当日及以前信息。

### 3.5 第三步:新闻链指 (O2b_embed_news.py)

时间映射 15:00 之后发布的新闻映射到下一交易日；否则映射到当天(若为交易日)。 保证 \( t \) 日收盘后使用的文本不包含 \( t \) 日收盘后的未公开信息。

名称匹配 在 title/content 中匹配 basic.csv 的 name，同日同股多条新闻拼接。规则匹配存在漏链/误链, 但 RoBERTa-LoRA 仍能在验证集获得 26.6% 融合权重 (见表 24), 显著优于档位 A NewsHash 的 0.7%。

### 3.6 第四步:滑动窗口样本 (DailyStockDataset)

- 样本 = (ts_code, trade_date);

- 时序: 过去 60 日 \( \times  {16} \) 维 \( \rightarrow  {\mathbb{R}}^{{60} \times  {16}} \) ,不足则 zero-padding + mask;

- 截面: 当日 9 维 \( \rightarrow  {\mathbb{R}}^{9} \) ;

- 静态:行业 id、log_mv、listing_age(供 TFT-lite)；

- 标签:excess_ret_1d。

划分原则:split_by_date 严格按 trade_date 切 train/valid，训练集内 shuffle，禁止把验证日混入训练。

### 3.7 时间划分与防泄露

表 7:本实验数据划分 (来自 ic_summary.json)

<table><tr><td>集合</td><td>截止规则</td><td>IC 天数</td><td>用途</td></tr><tr><td>训练集</td><td>≤ 2025-03-31</td><td>1503 (A) / 1509 (C)</td><td>三模态训练</td></tr><tr><td>验证集</td><td>2025-04-01 — 2026-05-20</td><td>273-274</td><td>唯一报告区间: IC + 融合定权 +</td></tr><tr><td>样本外</td><td>2026-06-01 — 06-10 (模拟赛)</td><td>-</td><td>真实下单; 不写入 IC/回测表</td></tr></table>

验证集为赛前 holdout: 训练含 2025Q1，验证覆盖 2025Q2 至 2026-05-20(模拟赛前)，使 ICIR 定权与回测更贴近 FDL2026 时间分布；模拟赛当周为样本外。

表 8:防数据泄露自检(本实验)

<table><tr><td>错误做法</td><td>本实验做法</td><td>代码位置</td></tr><tr><td>全样本 fit Scaler</td><td>逐日截 面 sorize+rank win-</td><td>preprocess.py</td></tr><tr><td>用 \( t + 1 \) 数据做 \( t \) 特征</td><td>特征仅用 \( \leq  t \)</td><td>features.py</td></tr><tr><td>15:00 后新闻当天可用</td><td>映射到下一 trade_date</td><td>news_link.py</td></tr><tr><td>随机打乱全部日期</td><td>按日期切 train/valid</td><td>dataset.py</td></tr><tr><td>在 valid 上调参再报告 <br> valid</td><td>固定 3 seed, 融合权重 frozen</td><td>fusion_weights.json</td></tr></table>

## 4 模型设计与方法论

本节说明 TFNE-C 三模态各自的学习目标、网络结构与训练/融合原理。可以将其理解为: 三位「专家」分别阅读价量序列、截面因子与新闻文本，各自输出对 \( {y}_{i, t} \) 的预测, 再按历史 ICIR 加权投票。

### 4.1 总体架构 TFNE-C

---

模态 A (TFT-lite): \( \left( {{60},{16}}\right) \) 序列 + 静态变量 \( \xrightarrow[]{\text{ LSTM+Attn }}{z}_{A}\left( {\times 3\text{ seed }}\right) \)

	模态 \( \mathrm{B} \) (FT-Transformer): 9 维截面 \( \frac{\text{ Self-Attn } \times  4}{},{z}_{B}\left( {\times 3\text{ seed }}\right) \)

		模态 \( \mathrm{C} \) (NewsLoRA): 新闻文本 \( \frac{\text{ RoBERTa } + \text{ LoRA }}{2},{z}_{C}\left( {\times 3\text{ seed }}\right) \)

		seed 内均值 逐日秩变换 \( {r}_{A},{r}_{B},{r}_{C}\xrightarrow[]{\text{ ICIR 定权 }} \) score 策略层 回测

---

图 2: TFNE-C 总体架构示意

三模态独立训练、独立保存 best.pt, 不在训练阶段做 end-to-end 联合优化——这样某模态 (如新闻缺失日) 可自动降权, 工程上亦便于增量 predict。

### 4.2 模态 A: TFT-lite(时序分支)

设计动机 档位 A 用 2 层 GRU \( \left( {L = {20}}\right) \) 编码价量,参数量小但难以对 60 步序列中「哪些交易日更重要」做显式加权。TFT-lite 借鉴 Temporal Fusion Transformer 思想:LSTM 提取序列表示，时序注意力对有效时间步加权汇聚，再与行业/市值等静态变量融合。

网络结构 (SeqTFTLite) 输入形状 \( \left( {B,{60},{16}}\right) , B \) 为 batch 内股-日样本数。前向过程:

1. 双层 LSTM (hidden=128, dropout=0.2) \( \rightarrow  {h}_{t} \in  {\mathbb{R}}^{128}, t = 1,\ldots ,{60} \) ;

2. 时序注意力: \( {e}_{t} = {w}^{\top }\tanh \left( {W{h}_{t}}\right) \) ,对 padding 位置 mask 为 \( - \infty ,{\alpha }_{t} = \operatorname{softmax}\left( {e}_{t}\right) \) , 上下文 \( c = \mathop{\sum }\limits_{t}{\alpha }_{t}{h}_{t} \)

3. 静态嵌入:行业 Embedding + log_mv、listing_age 投影为 \( s \in  {\mathbb{R}}^{128} \) ；

4. 融合头: \( \left\lbrack  {c;s}\right\rbrack   \rightarrow  \operatorname{MLP} \rightarrow  {z}_{A} \in  \mathbb{R} \) 。

训练与早停 损失为 \( \operatorname{Huber}\left( {\delta  = {1.0}}\right) \) ,优化器 AdamW ( \( {lr} = 3 \times  {10}^{-4} \) ), batch=4096 。 每个 epoch 结束在验证集上算 mean IC, patience=6 早停。训练日志显示三 seed 最佳验证 IC 差异较大:

表 9: TFT-lite 各 seed 最佳验证 IC (训练日志 job 7945)

<table><tr><td>seed</td><td>best val IC</td><td>best epoch</td><td>说明</td></tr><tr><td>42</td><td>0.0164</td><td>5</td><td>早停于 epoch 11, 拟合较浅</td></tr><tr><td>123</td><td>0.0662</td><td>16</td><td>充分训练后 IC 最高</td></tr><tr><td>456</td><td>0.0448</td><td>16</td><td>中等</td></tr></table>

seed 内均值(融合前) 约 0.04-0.05 量级，仍低于 FTT

原因分析: Top3000 价量序列信噪比低, 单 seed GRU/TFT 在日频上 IC 往往接近 0 ；三 seed 集成 + 更长训练(最多 18 epoch)后，融合阶段 TFT 仍获 35.3% 权重， 说明其在部分市场阶段(如趋势明显的月份)提供增量，而非单点 IC 最高。

### 4.3 模态 B: FT-Transformer(截面分支)

原理 Gorishniy 等 (2021) 提出: 表格数据上, 将每个数值特征视为一个 token, 通过 Transformer 自注意力学习特征间交互 (如「低 PB + 高换手 + 资金流入」的非线性组合), 往往优于纯 MLP。

实现 (TabFTTransformer, rtdl) 9 个数值 token + 类别 token, \( {d}_{\text{ token }} = {128},{n}_{\text{ blocks }} = 4 \) , attention/FFN dropout \( = {0.2} \) ,输出标量 \( {z}_{B} \) 。相对档位 A TabMLP (256-64-1),参数量与表达力均更强。

表 10: FT-Transformer 各 seed 最佳验证 IC

<table><tr><td>seed</td><td>best val IC</td><td>best epoch</td></tr><tr><td>42</td><td>0.0613</td><td>6</td></tr><tr><td>123</td><td>0.0660</td><td>4</td></tr><tr><td>456</td><td>0.0665</td><td>6</td></tr></table>

三 seed IC 集中在 0.061-0.067, 方差远小于 TFT, 说明截面因子信号更稳定—— 与融合权重 38.1%(三模态最高)一致。对比档位 A MLP 权重 34.1% 但全市场 IC 仅 0.016:FTT 不仅权重相近，且绝对 IC 水平显著更高，是档位 C 相对 A 排序能力提升的主因。

### 4.4 模态 C: NewsLoRA(新闻分支)

原理 档位 A 用 HashingVectorizer 将文本映射为 256 维稀疏向量, 无法建模语义, 融合权重仅 0.7%。本实验采用 hf1/chinese-roberta-wwm-ext 预训练编码器 + LoRA ( r = 8, α = 16)低秩微调:在冻结大部分 BERT 参数的同时，用少量可训练参数适配金融新闻域。

训练设置 从训练集随机子采样 8 万条 (有新闻的股-日), 最大长度 128 token, 4 epoch, batch=64, \( \operatorname{lr} = 2 \times  {10}^{-4} \) 。各 seed 最佳验证 IC:

表 11: NewsLoRA 各 seed 最佳验证 IC

<table><tr><td>seed</td><td>best val IC</td><td>best epoch</td></tr><tr><td>42</td><td>0.0366</td><td>1</td></tr><tr><td>123</td><td>0.0453</td><td>4</td></tr><tr><td>456</td><td>0.0390</td><td>1</td></tr></table>

原因:新闻覆盖不全(无新闻日 rank_C 为 NaN)，且事件驱动 alpha 本身稀疏; 但 0.04 量级 IC 已远高于 NewsHash，使融合权重从 0.7% 升至 26.6%。推理阶段 04_ensemble_predict.py 支持从 checkpoint 全量推理 (约 8 h),保证增量 predict 时 News 模态完整参与。

### 4.5 训练目标与优化细节

三模态回归目标均为 \( {y}_{i, t} \) (超额收益),但模型选择指标为验证集 Spearman IC 而非 MSE——因为策略按 score 排序选股, IC 与下游目标一致。

Huber 损失

\[
{\mathcal{L}}_{\text{ Huber }}\left( e\right)  = \left\{  {\begin{array}{ll} \frac{1}{2}{e}^{2}, & \left| e\right|  \leq  \delta \\  \delta \left( {\left| e\right|  - \frac{1}{2}\delta }\right) , & \left| e\right|  > \delta  \end{array}\;\left( {\delta  = {1.0}}\right) }\right. \tag{22}
\]

对涨跌停等极端 \( y \) 比纯 MSE 更稳健,与档位 A 相同。

数值稳定 Top3000 全量 TFT 训练时开启 AMP 曾出现 loss=nan，故本实验 amp=false、 tf32=false,以约 20% 速度换取稳定收敛。

表 12: 档位 C 训练超参与实测耗时 (job 7945)

<table><tr><td>项</td><td>取值</td><td>说明</td></tr><tr><td>优化器</td><td>AdamW, \( \operatorname{lr} = 3 \times  {10}^{-4} \)</td><td>TFT/FTT</td></tr><tr><td>News lr</td><td>\( 2 \times  {10}^{-4} \)</td><td>LoRA 微调</td></tr><tr><td>损失</td><td>Huber \( \left( {\delta  = {1.0}}\right) \)</td><td></td></tr><tr><td>batch</td><td>4096(股-日)</td><td></td></tr><tr><td>TFT/FTT</td><td>max 18 epoch, patience 6</td><td></td></tr><tr><td>News</td><td>4 epoch, 8 万样本/seed</td><td></td></tr><tr><td>TFT 训练</td><td>\( 3 \times \) seed,共 \( \approx  {18.4}\mathrm{\;h} \)</td><td></td></tr><tr><td>FTT 训练</td><td>\( 3 \times \) seed,共 \( \approx  {11.9}\mathrm{\;h} \)</td><td></td></tr><tr><td>News 训练</td><td>3×seed，共 ≈0.7 h</td><td></td></tr><tr><td>04 融合推理</td><td>\( \approx  {3.0}\mathrm{\;h} \)</td><td>534 万行 \( \times  6 \) 次推理</td></tr><tr><td>墙钟总时长</td><td>\( \approx  {39}\mathrm{\;h} \)</td><td></td></tr></table>

### 4.6 多 seed 集成与秩融合

Step 1: seed 内平均 对每个模态、每个 (trade_date, ts_code),对 \( {z}^{\left( \text{ seed }\right) } \) 取算术平均得 \( {z}_{A},{z}_{B},{z}_{C} \) 。

Step 2: 逐日截面秩变换 在每个 trade_date 内,对 \( {z}_{A},{z}_{B},{z}_{C} \) 分别 winsorize \( \left( {q = {0.01}}\right) \) 再 rank 到 \( \left\lbrack  {0,1}\right\rbrack \) ,得 \( {r}_{A},{r}_{B},{r}_{C} \) 。缺失模态 (如无新闻) 对应 rank 为 NaN,不参与当日加权。

Step 3: ICIR 定权 在验证集上分别计算 \( \mathrm{A}/\mathrm{B}/\mathrm{C} \) 的日度 IC 序列,

\[
{\operatorname{ICIR}}_{m} = \frac{{\overline{\mathrm{{IC}}}}_{m}}{\operatorname{std}\left( {\mathrm{{IC}}}_{m}\right) },\;{w}_{m} = \frac{\max \left( {{\mathrm{{ICIR}}}_{m},0}\right) }{\mathop{\sum }\limits_{k}\max \left( {{\mathrm{{ICIR}}}_{k},0}\right) }. \tag{23}
\]

本实验得到 \( {w}_{A} = {35.3}\% ,{w}_{B} = {38.1}\% ,{w}_{C} = {26.6}\% \) 。融合 score:

\[
{\text{ score }}_{i, t} = {w}_{A}{r}_{A, i, t} + {w}_{B}{r}_{B, i, t} + {w}_{C}{r}_{C, i, t}\text{ . } \tag{24}
\]

与档位 A 对比: A 为 GRU 62.9% / MLP 36.4% / Hash 0.7%——时序独大、新闻近零。C 三分支更均衡, 说明 Transformer 升级后新闻与截面均成为有效 alpha 来源, 而非仅靠价量时序。

## 5 交易策略与回测设置

预测模型输出的是分数 (score), 不能直接变成账户收益。策略层 (src/strategy.py) 负责回答:「今天应该买哪些股票、各买多少？」回测层 (src/backtest.py) 再按 A 股交易规则模拟成交，得到净值曲线。本节详细说明策略逻辑、合理性分析，以及回测引擎的逐步执行过程。

### 5.1 与作业 PDF 及档位 A 的关系

作业 PDF 推荐 Top- \( k \) 轮换策略: 首日等权买入得分最高的 \( n \) 只 \( \left( {n = 5 - {30}}\right) \) ,之后每日卖出持仓中得分最低的 \( k \) 只、换入全市场得分最高的 \( k \) 只。

本方案(与档位 A 同框架)的扩展:

- 保留「按 score 排序选股」核心思想;

- 将「等权 \( n \) 只」改为按分数 softmax 式加权，使 score 越高权重越大；

- 加入单票/行业上限、现金缓冲、波动率防守、换手上限——更易满足比赛「尽可能满仓」并控制集中度风险;

- 本实验 n_max=50 (档位 A 为 30)，适配 Top3000 更宽候选池。

### 5.2 可交易 universe: 哪些股票允许买?

在计算目标权重前，tradable_universe 对当日面板过滤 (与 prep 规则一致):

- 剔除 ST、北交所 (.BJ);

- 当日成交量 \( > 0 \) (排除停牌);

- 上市满 60 个交易日;

- 20 日平均成交额と 5000 万元。

意义:模型对 Top3000 全池都有 score，但策略只在「realistically 能买到」的标的上建仓，避免回测买入不可成交股票。过滤后当日有效 score 通常仍为数百至千余只 (验证集 IC 统计要求每日 \( \geq  {30} \) 只有效样本)。

### 5.3 每日策略计算:target_weights 分步说明

函数 target_weights(scores, industries, log_mv, params, mkt_vol) 输入当日可交易股票的 score、行业、对数市值，输出目标权重 \( {w}_{i} \) (不含现金；现金由 \( {c}_{t} \) 单独留出)。

Step 1: 头部候选池 (与 IC 评估对齐)

\[
{\mathcal{C}}_{t} = \left\{  {i \mid  {s}_{i, t} \geq  {Q}_{0.90}\left( {s}_{\cdot , t}\right) }\right\}  , \tag{25}
\]

即 score 位于当日 Top 10% (q_min=0.90)。Top3000 下 \( \left| {\mathcal{C}}_{t}\right|  \approx  {300} \) 只。这与 IC 评估中的 Top 分位指标一致——策略只在模型最有把握的头部区域建仓，而非全市场均匀暴露。

Step 2: 取前 n_max 只 按 score 降序取前 50 只 (n_max=50)，记为集合 \( {\mathcal{H}}_{t} \) 。

Step 3: 分数 \( \rightarrow \) 初始权重 (softmax 式) 对 \( {\mathcal{H}}_{t} \) 内 score 做百分位 rank \( {r}_{i} \in  \left\lbrack  {0,1}\right\rbrack \) , 计算

\[
{\widetilde{w}}_{i} = \exp \left( \frac{{r}_{i} - {0.5}}{\tau }\right) ,\;\tau  = {0.5},\;{w}_{i}^{\left( 0\right) } = \frac{{\widetilde{w}}_{i}}{\mathop{\sum }\limits_{{j \in  {\mathcal{H}}_{t}}}{\widetilde{w}}_{j}}. \tag{26}
\]

\( \tau  = {0.5} \) 时, score 最高者权重约为最低者的 \( {e}^{1} \approx  {2.7} \) 倍——比等权更倾斜于头部,但不如 argmax 全押一只极端。

## Step 4: 风险约束迭代

1. 单票上限 (w_max=0.05):任一 \( {w}_{i} \leq  5\% \) ，超出部分迭代重分给未触顶股票(最多 5 轮);

2. 行业上限 (w_ind_max=0.25):同一 industry 合计 ≤25%，超标则行业内等比例缩放;

3. 微盘降权 (downweight_microcap=True): \( {\mathcal{H}}_{t} \) 内 \( \log \_ \mathrm{{mv}} \) 最低 10% 股票权重 \( \times  {0.5} \) , 再归一化——降低流动性差小票的风险；

4. 留现金:

\[
{c}_{t} = {c}_{\min } + {\gamma }_{\mathrm{{vol}}} \cdot  \mathbb{1}\left\lbrack  {{v}_{t}^{\mathrm{{mkt}}} > {v}_{80}^{\mathrm{{mkt}}}}\right\rbrack  ,\;{c}_{\min } = 3\% ,{\gamma }_{\mathrm{{vol}}} = 5\% , \tag{27}
\]

其中 \( {v}_{80}^{\text{ mkt }} \) 为训练期市场波动率 80% 分位 (compute_mkt_vol_threshold 在 train_end=20250331 前估计)。最终 \( {w}_{i} \leftarrow  {w}_{i} \cdot  \left( {1 - {c}_{t}}\right) \) 。

数值示例:最新调仓 orders_20260520.csv 含 50 只股票, 权重合计 97.0%, 最高单票 002594.SZ 为 5.01%(触顶 w_max)，其余 3% 为现金缓冲——符合「尽可能满仓」 要求。

表 13: 本实验交易策略参数 (config.server_c2_72h.yaml)

<table><tr><td>参数</td><td>取值</td><td>作用</td></tr><tr><td>q_min</td><td>0.90</td><td>只在 Top 10% score 中选股</td></tr><tr><td>n_max</td><td>50</td><td>目标组合最多 50 只</td></tr><tr><td>tau</td><td>0.5</td><td>权重集中度; 越小越集中</td></tr><tr><td>w_max</td><td>5%</td><td>单票权重上限</td></tr><tr><td>w_ind_max</td><td>25%</td><td>单行业权重上限</td></tr><tr><td>c_min</td><td>3%</td><td>基础现金比例</td></tr><tr><td>gamma_vol</td><td>5%</td><td>高波动日额外现金</td></tr><tr><td>delta</td><td>1%</td><td>目标偏离 \( < 1\% \) 总资产则不调</td></tr><tr><td>omega_max</td><td>25%</td><td>单日最大换手率</td></tr></table>

### 5.4 策略合理性分析

(1)Top 10% 候选 + 加权持仓——与 alpha 来源一致 验证集全市场 mean IC 为 0.077，但 Top 10% 内 IC 仅 0.014:说明模型对「头尾分组」的区分力强于「头部内部精细排序」。策略只交易 Top 10% 并最多持 50 只，与 spread 指标(日均 0.314%)所衡量的对象一致，避免在全市场噪声段浪费仓位。

(2)softmax 加权 vs 等权——利用 score 幅度信息 秩融合后 score 已在 \( \left\lbrack  {0,1}\right\rbrack \) 附近, rank+exp 加权使更高 score 获得更多资金,比 PDF 等权 Top- \( k \) 更充分利用模型输出; \( \tau  = {0.5} \) 在「集中押注最强信号」与「分散 idiosyncratic 风险」之间折中。

(3)单票 5%、行业 25%——控制集中度 Top3000 池中单一行业(如券商、新能源) 可能在某日同时占据多个高分位。行业 cap 防止组合在单一板块上过度暴露；单票 5% cap 防止对个别「极端高分」标的 all-in。回测中 2025-04-07 单日 -12.15% 的极端下跌说明系统性风险仍无法完全通过分散消除，但 caps 限制了单票冲击。

(4)3%+ 波动率防守现金——兼顾满仓与风控 比赛要求尽可能满仓，完全不留现金不现实(涨跌停、整手约束会导致部分资金闲置)。3% 基础现金 + 高波动日额外 5% 是规则化风控:当 \( {v}_{t}^{\text{ mkt }} \) 超过训练期 80% 分位时自动降杠杆，不依赖人工择时。

(5)软换仓(delta、omega_max)——贴近真实交易摩擦 硬换仓(每日完全对齐目标权重)在 A 股会产生过高换手与成本。本策略:

---

- delta=1%:忽略小于总资产1%的偏离，减少无意义小单；

---

- omega_max=25%:单日买卖金额不超过总资产的 25%，避免 score 日际剧变导致一次性清仓/建仓。

副作用(需在报告中诚实说明):验证集回测中持仓数 n_hold 从期初 34 只逐步升至后期约 160-206 只(均值约 184)，高于 n_max=50 的目标持仓数——原因是换手上限使 「应卖出的旧仓」无法在一日内全部清掉，属于软换仓的滞后效应，而非目标权重公式本身持 200 只。

表 14: 档位 A 与本实验策略参数差异

<table><tr><td>维度</td><td>档位 A</td><td>本实验</td><td>合理性</td></tr><tr><td>n_max</td><td>30</td><td>50</td><td>Top3000 候选更宽, 略增分散度</td></tr><tr><td>其余参数</td><td>相同</td><td>相同</td><td>对比时策略层一致, 差异主要来自 score</td></tr></table>

## (6)与档位 A 策略差异

### 5.5 回测引擎:整体流程

回测入口 scripts/06_backtest.py:读取 panel_features.parquet 与 ensemble_scores.parqu 在验证集 20250401-20260520 上调用 run_backtest, 输出 equity_curve.csv 与 metrics.json。

---

回测主循环(每个交易日 \( t = 1,\ldots , T - 1 \) ):

(1) 用 \( t \) 日收盘价计算当前净值 \( {W}_{t} = \operatorname{cash} + \mathop{\sum }\limits_{i}{\text{ shares }}_{i} \cdot  {P}_{i, t}^{\text{ close }} \)

(2) 取 \( t \) 日 score,经 tradable_universe 过滤 \( \rightarrow \) target_weights 得 \( {w}_{i}^{\text{ tgt }} \)

(3) 在 \( t + 1 \) 日以开盘价(缺失则收盘价)执行 rebalance_day

		(4)扣除佣金/印花税/滑点; 更新 T+1 可卖标记

(5) 记录 \( t \) 日净值、现金、持仓数 n_hold

---

图 3: 回测引擎 run_backtest 主循环示意

### 5.6 单日回测时间线 (T+1 无未来信息)

表 15: 单日回测时间线 ( \( t \) 为信号日, \( t + 1 \) 为执行日)

<table><tr><td>时点</td><td>发生什么</td></tr><tr><td>\( t \) 日收盘后</td><td>使用 \( \leq  t \) 的特征与新闻得到融合 score \( {s}_{i, t} \) (与训练标签 \( {y}_{i, t} = \; {r}_{i, t + 1} - {r}_{t + 1}^{\text{ mkt }} \) 的时间对齐)</td></tr><tr><td>\( t \) 日收盘后</td><td>在可交易 universe 上计算目标权重 \( {w}_{i, t}^{\text{ tgt }} \)</td></tr><tr><td>\( t + 1 \) 日开盘</td><td>以 \( {P}_{i, t + 1}^{\text{ open }} \) 尝试调仓 (无 open 则用 close)</td></tr><tr><td>执行顺序</td><td>先卖后买:卖出释放现金，再买入新标的</td></tr><tr><td>整手约束</td><td>买卖股数 \( = \lfloor \left| {\Delta V}\right| /\left( {{100} \cdot  P}\right) \rfloor  \times  {100} \)</td></tr><tr><td>涨停过滤</td><td>\( t + 1 \) 日涨幅 \( \geq  {9.5}\% \) 的标的不买</td></tr><tr><td>T+1 可卖</td><td>\( t + 1 \) 日买入的仓位 sellable=False, \( t + 2 \) 起可卖</td></tr><tr><td>成本</td><td>佣金 0.015%、卖出印花税 0.05%、滑点 0.05%(双边均扣)</td></tr></table>

因果性说明: \( t \) 日记录的净值使用 \( t \) 日收盘价估值,但调仓执行在 \( t + 1 \) ——与「盘后出信号、次日交易」一致，不使用 \( t + 1 \) 日收盘价在 \( t \) 日决策。

### 5.7 调仓逻辑 rebalance_day 详解

设当前总资产 \( W \) ，当前持仓市值 \( {V}_{i}^{\text{ cur }} \) ，目标权重 \( {w}_{i}^{\text{ tgt }} \) 。

## Step 1: 计算目标市值与偏离

\[
{V}_{i}^{\mathrm{{tgt}}} = W \cdot  {w}_{i}^{\mathrm{{tgt}}},\;\Delta {V}_{i} = {V}_{i}^{\mathrm{{tgt}}} - {V}_{i}^{\mathrm{{cur}}}. \tag{28}
\]

若 \( \left| {\Delta {V}_{i}}\right| /W < \delta \) (1%),则跳过该标的一一降低噪声换手。

Step 2: T+1 卖出约束 若 \( \Delta {V}_{i} < 0 \) (应卖) 但该仓 \( t + 1 \) 日不可卖 (当日新买),则跳过一一符合 A 股 \( \mathrm{T} + 1 \) 规则。

Step 3: 换手上限缩放 估算单日换手率 \( \omega  = \mathop{\sum }\limits_{i}\left| {\Delta {V}_{i}}\right| /\left( {2W}\right) \) 。若 \( \omega  > {\omega }_{\max } = {25}\% \) ,所有 \( \Delta {V}_{i} \) 等比例缩小——模拟「大单无法一日全部成交」。

Step 4: 先卖后买 + 整手 卖出: 按整手向下取整, 不超过当前持仓; 买入: 按整手向下取整，不超过可用现金。无法成交的零头留到下一日。

Step 5: 费用扣除 每笔成交金额 \( {V}_{\text{ trade }} \) :

- 佣金: \( {V}_{\text{ trade }} \times  {0.00015} \) (买卖均收)；

- 印花税: \( {V}_{\text{ trade }} \times  {0.0005} \) (仅卖出)；

- 滑点: \( {V}_{\text{ trade }} \times  {0.0005} \) 。

粗略单边总成本约 0.02%-0.07%，273 日累计对净值的侵蚀显著，是组合收益低于 spread 理论值的重要原因。

### 5.8 回测配置与验证集设置

表 16: 回测配置(BacktestConfig + 验证区间)

<table><tr><td>项</td><td>取值</td><td>说明</td></tr><tr><td>初始资金</td><td>100 万元</td><td></td></tr><tr><td>回测区间</td><td>2025-04-01 — 2026-05-20</td><td>273 个交易日</td></tr><tr><td>score 来源</td><td>ensemble_scores.parquet</td><td>融合后 score</td></tr><tr><td>面板来源</td><td>panel_features.parquet</td><td>含价格、行业、市值</td></tr><tr><td>整手</td><td>100 股</td><td>lot_size=100</td></tr><tr><td>无风险利率</td><td>0</td><td>Sharpe 计算用</td></tr><tr><td>年化天数</td><td>252</td><td>CAGR/波动率年化</td></tr></table>

本配置 test_end=null，仅对 valid 段回测; 模拟赛 2026-06-01 至 06-10 为样本外， 不在 backtest/*/test/ 中报告。

### 5.9 回测过程观察 (基于 equity_curve.csv)

从验证集净值序列可提取以下过程特征，用于理解表 22 中指标的形成机制:

表 17: 验证集回测过程统计(由净值曲线计算)

<table><tr><td>指标</td><td>数值</td><td>含义</td></tr><tr><td>期初净值 <br> 谷底净值</td><td>100.01 万元(2025-04-01) <br> ≈72.5 万元</td><td>初始资金 + 首日微小调仓 <br> 对 应 MDD -27.5% (metrics.json)</td></tr><tr><td>期末净值 <br> 最大单日跌幅 <br> 最大单日涨幅 <br> 持仓数 n_hold</td><td>143.5 万元(2026-05-20) <br> -12.15%(2025-04-07) <br> +4.81%(2026-04-08) <br> \( {34} \rightarrow  {206} \) (均值约 184)</td><td>总收益 43.5% <br> 系统性下跌 + 高 beta 暴露 <br> 反弹段 <br> 软换仓滞后，实际持仓可高于 n_max</td></tr><tr><td>现金占比</td><td>均值约 0.37%</td><td>大部分时间接近满仓</td></tr></table>

过程解读:

1. 2025-04 初:市场剧烈调整，组合在 4 月前两周快速建仓 (n_hold 34 → 97)，4 月 7 日单日 -12% 导致净值大幅回撤，4 月整月收益 -17.22%；

2. 2025-06 底:净值触底 74.96 万，此后随 IC 滚动均值回升(图 5)与 2025-08 (+16.87%)、2026-01(+15.28%)等强势月份修复；

3. 2026-05 末:净值 143.5 万，5 月单月 +19.40%，与最新调仓高 score 板块(制造、 券商等) 一致;

4. 成本与换手: 273 日累计调仓在换手上限约束下渐进进行，费用从 gross spread 中扣除，使 40% 总收益低于 spread 年化 79% 的理论参考。

### 5.10 从目标权重到模拟下单

07_predict_latest.py 在最后一个交易日生成 orders_YYYYMMDD.csv, 列 ts_code、 target_weight。模拟赛操作:

1. 每个交易日 \( T \) 收盘后更新数据并运行预测流水线;

2. 目标市值 \( = {W}_{T} \times \) target_weight;

3. 与当前持仓 diff 得买卖清单, \( T + 1 \) 按 100 股整手委托。

Top 15 权重见图 8 (见第 6 节)。

## 6 实验结果与对比分析

本节结合 ic_summary.json、backtest/tier_c2/valid/ 与训练日志, 对预测指标、 净值曲线、融合权重进行数值一机理对照分析，并与档位 A 基线比较。

### 6.1 预测能力: IC / ICIR / Spread

表 18: 本实验——训练集预测指标(融合 score, 1509 日)

<table><tr><td>指标</td><td>mean IC / spread</td><td>std</td><td>ICIR / 年化</td><td>天数</td></tr><tr><td>全市场 IC</td><td>0.0742</td><td>0.096</td><td>0.769</td><td>1509</td></tr><tr><td>Top 10% 内 IC</td><td>0.0161</td><td>0.079</td><td>0.202</td><td>1509</td></tr><tr><td>多空 spread(日)</td><td>0.475%</td><td>1.06%</td><td>年化 119.7%</td><td>1509</td></tr></table>

表 19: 本实验——验证集预测指标(2025-04-01 — 2026-05-20，273 日)

<table><tr><td>指标</td><td>mean IC / spread</td><td>std</td><td>ICIR / 年化</td><td>天数</td></tr><tr><td>全市场 IC</td><td>0.0769</td><td>0.097</td><td>0.794</td><td>273</td></tr><tr><td>Top 10% 内 IC</td><td>0.0143</td><td>0.079</td><td>0.181</td><td>273</td></tr><tr><td>多空 spread(日)</td><td>0.314%</td><td>1.11%</td><td>年化 79.0%</td><td>273</td></tr></table>

(1)全市场 IC 与 ICIR——排序能力显著提升 验证集 mean IC = 0.0769 , ICIR = 0.794 ; 训练集 IC = 0.0742 , ICIR = 0.769 。验证略高于训练, 说明在 2025-04 起的 holdout 上未出现明显过拟合崩塌。对比同口径档位 A 验证 IC 0.016、ICIR 0.102:本实验 IC 约为 A 的 4.8 倍，ICIR 约为 7.8 倍。

原因分解:

- 截面 Transformer (主因): FTT 单 seed 验证 IC 0.061-0.067, 远高于 A 的 MLP 在 500 只池上的弱信号; Top3000 扩大横截面, IC 估计更稳定 (std 0.097 vs A 的 0.153);

- 新闻语义 (次因): NewsLoRA 权重 \( {26.6}\% \) ,提供 A 不具备的事件驱动信息;

- 更长窗口与三 seed (辅助): \( L = {60} \) 与多 seed 降低 TFT 单 seed 方差,使时序模态在融合中占 35.3% 而非 A 的 GRU 独大 62.9%。

(2)Top 10% 内 IC 与 spread——「宽头部难细分，头尾仍可区分」 验证集 Top 10% 内 IC: C2 为 0.014 , A 为 0.013 , 均偏低。多空 spread: C2 日均 0.314% (年化 79.0%)，A 为 -0.116%/日——A 在「全池头尾 10%」分组上略负，但经 target_weights 与 Top500 窄池筛选后, 组合回测仍可获 32.2% 总收益; 说明策略层约束改变了可交易 alpha 的来源。

### 6.2 策略对齐指标与相对大盘: A vs C2 详析

表 2 与表 20 汇总策略对齐指标 (数据源: metrics/*/strategy_topk.json, 本机命令 python scripts/print_strategy_topk_metrics.py, 2026-05-25)。报告正文仅引用 valid 行; train 供过拟合自检; test 行因 test_end=null 样本过少(A: \( n = 1 \) , C2: \( n = 0 \) ) 不纳入结论。

(1)IC@K:C2「进池强、池内弱」，A「全池弱、池内相对更好」

- 档位 A 验证集 \( \left( {K = {30}}\right)  : \mathrm{{IC}}@\mathrm{K} =  + {0.0303},\mathrm{{ICIR}}@\mathrm{K} =  + {0.146}\left( {n = {274}}\right) \) 。Top500 经 \( {Q}_{0.90} \) 过滤后的 30 只候选内, score 对次日超额仍呈弱正秩相关—— \( K \) 较小、候选更精, Spearman 更易为正。

- 档位 \( \mathrm{C}2 \) 验证集 \( \left( {K = {50}}\right)  : \mathrm{{IC}}@\mathrm{K} =  - {0.0037},\mathrm{{ICIR}}@\mathrm{K} =  - {0.020}\left( {n = {273}}\right) \) 。全市场 IC 0.077 未转化为「最终 50 只」内的稳定排序; alpha 主要体现在进 Top 池与头尾 spread，而非池内座次。

- 训练 vs 验证: A 的 IC@K 从 train +0.0132 升至 valid +0.0303 (holdout 上未塌); C2 从 train +0.0029 降至 valid -0.0037 (池内排序略弱化, 但全市场 IC 仍升)。

- 启示:模拟赛若依赖 Top-K 加权，C2 应更看 spread / 回测; 若要提升 IC@K，需针对 \( K \) 缩小候选或二次筛票，而非仅追全市场 IC。

## (2)命中率、Top- \( K \) 内 spread 与信号层收益

- 命中率 (valid): A 47.0%, C2 50.1%——均接近 50%，单日「跑赢大盘」比例接近随机；组合 PnL 来自权重倾斜与多日复利。

- Top-K 内 spread (valid, 年化): A 10.2%, C2 21.8% (train: A -1.4%, C2 16.6%)。C2 池内 「高分半仓 vs 低分半仓」分化更厚，与 IC@K 略负不矛盾。

- 信号层 signal_ann (valid,无成本): A -14.2% vs 同窗回测 +32.2%; C2 +34.1% vs 回测 +43.5%。A 的严重背离因 risk_adjust 与信号层假设(每日完美调仓、无成本)与真实回测路径不同——组合结论以扣费回测为准。

(3)相对大盘:C2 超额更厚, 但 beta 更高 验证集同窗沪深 300 总收益 24.8%，Sharpe 1.39, MDD -7.8%。

- 档位 A: 超额总收益 5.9%，IR 0.67， \( {\alpha }_{\text{ ann }} = {5.4}\% ,\beta  = {1.01} \) ——接近市场中性暴露，回撤浅(MDD -9.2%)。

- 档位 \( \mathbf{C}2 \) : 超额总收益 \( {15.0}\% \) , IR 0.74, \( {\alpha }_{\text{ ann }} = {9.0}\% ,\beta  = {1.33} \) ——绝对 alpha 更高,但市场下行时放大亏损 (2025-04 单日 \( - {12}\% \) 与 \( \beta  > 1 \) 一致)。

表 20: 策略 Top-K 指标完整输出 (print_strategy_topk_metrics.py, 2026-05-25)

<table><tr><td>档位</td><td>集合</td><td>IC@K</td><td>ICIR@K</td><td>命中率</td><td>within_ann</td><td>signal_ann</td></tr><tr><td rowspan="2">A \( \left( {K = {30}}\right) \)</td><td>train</td><td>+0.0132</td><td>+0.056</td><td>50.3%</td><td>-1.4%</td><td>+13.1%</td></tr><tr><td>valid</td><td>+0.0303</td><td>+0.146</td><td>47.0%</td><td>10.2%</td><td>-14.2%</td></tr><tr><td rowspan="2">C2 \( \left( {K = {50}}\right) \)</td><td>train</td><td>+0.0029</td><td>+0.015</td><td>51.1%</td><td>16.6%</td><td>+52.4%</td></tr><tr><td>valid</td><td>-0.0037</td><td>-0.020</td><td>50.1%</td><td>21.8%</td><td>+34.1%</td></tr></table>

注:within_ann \( = \) Top- \( K \) 内 spread 年化；signal_ann \( = \) 信号层组合年化超额 (无交易成本)。 valid 区间 20250401-20260520; A valid \( n = {274} \) , C2 valid \( n = {273} \) 。

表 21: 策略对齐指标验证集对照(摘自表 20 valid 行)

<table><tr><td>指标</td><td>档位 A</td><td>档位 C2</td></tr><tr><td>IC@K</td><td>+0.0303</td><td>-0.0037</td></tr><tr><td>ICIR@K</td><td>+0.146</td><td>-0.020</td></tr><tr><td>Top- \( K \) 命中率</td><td>47.0%</td><td>50.1%</td></tr><tr><td>Top- \( K \) 内 spread(年化)</td><td>10.2%</td><td>21.8%</td></tr><tr><td>信号层组合(年化，无成本)</td><td>-14.2%</td><td>+34.1%</td></tr><tr><td>同窗回测总收益(扣费)</td><td>+32.2%</td><td>+43.5%</td></tr></table>

(4)训练 vs 验证 spread 下降 训练 spread 年化 119.7%，验证 79.0%，验证更低属正常: holdout 时段含 2025 年 4 月极端行情 (见下节), 且验证集不可用于调参, 组合层不会 [过拟合 spread]。

![31_312_934_1025_503_0.jpg](images/31_312_934_1025_503_0.jpg)

图 4: 训练集与验证集 IC / ICIR 对比 (融合 score)

![31_202_1613_1248_439_0.jpg](images/31_202_1613_1248_439_0.jpg)

图 5: 日度 Spearman IC 及 20 日滚动均值 (虚线: 验证集 2025-04-01 起点)

图 5 显示: 2025-04 初 IC 波动加剧 (与市场系统性下跌同步), 2025-07 后滚动 IC 均值多在 0.05-0.10 区间，2026 年初再度抬升——说明 alpha 具有时变特征，ICIR 0.794 已部分反映这种日际波动。

### 6.3 历史回测 (验证集)

表 22: 本实验——验证集历史回测 (273 个交易日)

<table><tr><td>指标</td><td>TFNE-C</td></tr><tr><td>总收益率 \( {R}_{\mathrm{{tot}}} \)</td><td>43.5%</td></tr><tr><td>复利年化 CAGR</td><td>39.6%</td></tr><tr><td>算术年化收益</td><td>37.8%</td></tr><tr><td>年化波动率</td><td>29.6%</td></tr><tr><td>夏普比率</td><td>1.28</td></tr><tr><td>最大回撤 MDD</td><td>-27.5%</td></tr><tr><td>期末净值</td><td>143.5 万元</td></tr><tr><td>相对沪深 300 超额总收益</td><td>15.0%</td></tr><tr><td>信息比率 IR</td><td>0.74</td></tr><tr><td>年化 Alpha</td><td>9.0%</td></tr><tr><td>Beta</td><td>1.33</td></tr></table>

## (1)净值路径与最大回撤

- 期初 (2025-04-01) 净值 100.01 万元;

- 最大回撤 MDD - 27.5%(273 日样本)，主要受 2025-04 系统性下跌驱动；

- 期末 (2026-05-20) 净值 143.5 万元，总收益 43.5%。

单日极端:最大单日跌幅 2025-04-07，日收益 -12.15%(全市场系统性下跌，持仓 85-97 只, beta 暴露大)。这解释了验证集前 4 个月净值承压。

表 23: 验证集分月组合收益率 (由 equity_curve.csv 计算)

<table><tr><td>月份</td><td>月收益</td><td>解读</td><td>月份</td><td>月收益</td><td>解读</td></tr><tr><td>2025-04</td><td>-17.22%</td><td>关税冲击，IC 波动大</td><td>2025-11</td><td>-6.80%</td><td>调整</td></tr><tr><td>2025-05</td><td>-4.68%</td><td>延续弱势</td><td>2025-12</td><td>+9.52%</td><td>反弹</td></tr><tr><td>2025-06</td><td>+4.34%</td><td>触底回升</td><td>2026-01</td><td>+15.28%</td><td>强势</td></tr><tr><td>2025-07</td><td>+8.00%</td><td>修复</td><td>2026-02</td><td>-2.75%</td><td></td></tr><tr><td>2025-08</td><td>+16.87%</td><td>最佳月之一</td><td>2026-03</td><td>-9.66%</td><td rowspan="2">回调</td></tr><tr><td>2025-09</td><td>+0.05%</td><td>横盘</td><td>2026-04</td><td>+10.11%</td></tr><tr><td>2025-10</td><td>-0.81%</td><td></td><td>2026-05</td><td>+19.40%</td><td>数据末强势</td></tr></table>

(2)分月收益——与 IC 时变相互印证 分析: 2025-04/05 两月累计亏损约 21%，与 MDD 谷底一致；2025-08、2026-01、2026-05 等高 IC 月份组合收益也较高，说明排序 alpha 在部分 regime 下可转化为 PnL，但并非每月稳定(2025-09 几乎零收益)。273 日样本下 CAGR 36.6% 高于 A 同窗 29.4%——两档已对齐验证时段，可比较总收益与 Sharpe, 但仍需考虑股票池与 n_max 差异。

![33_307_1040_1031_400_0.jpg](images/33_307_1040_1031_400_0.jpg)

图 6: 验证集净值曲线 (2025-04 起, 273 个交易日)

(3)为何 IC 高但 Sharpe 低于档位 A？档位 A 同验证窗 Sharpe 1.53 > 本实验 1.28， 但总收益 A 为 32.2%(273 F) < C 为 43.5%(273 F)。原因包括:

1. 股票池与持仓:A 为 Top500、n_max=30；C 为 Top3000、n_max=50，波动与 MDD 结构不同;

2. 行情暴露:C 含 2025-04 暴跌，MDD -27.5% 远深于 A 的 -9.2%；A 窄池在部分月份 Sharpe 更优;

3. 持仓更分散:n_max=50 vs 30，单票 alpha 对组合拉动被稀释；

4. 股票池更大:Top3000 尾部票噪声多，spread 虽高但扣费后组合收益折损更大。 结论: 本实验排序层明显优于 \( \mathrm{A} \) ,组合层互有胜负,报告应同时呈现 IC 与分段回测。

### 6.4 融合权重与模态贡献

表 24: 验证集 ICIR 融合权重 (fusion_weights.json)

<table><tr><td>代号</td><td>模型</td><td>权重</td><td>单模态 best IC (seed 内</td></tr><tr><td>A</td><td>TFT-lite</td><td>35.30%</td><td>0.016-0.066</td></tr><tr><td>B</td><td>FT-Transformer</td><td>38.12%</td><td>0.061-0.067</td></tr><tr><td>C</td><td>NewsLoRA</td><td>26.58%</td><td>0.037-0.045</td></tr></table>

权重与单模态 IC 排序大体一致 (FTT > News > TFT )，但 TFT 权重 (35.3%) 高于其单点 IC 暗示值——因 ICIR 定权看的是 IC 时间序列的均值/波动比，TFT 在部分趋势月份 IC 尖峰高,抬升 ICIR。对比 A: 新闻从 \( {0.7}\%  \rightarrow  {26.6}\% \) 是最显著的架构红利。

三模态 ICIR 融合权重 (TFNE-C2)

![34_208_944_534_468_0.jpg](images/34_208_944_534_468_0.jpg)

图 7: 三模态 ICIR 融合权重

![34_797_947_659_410_0.jpg](images/34_797_947_659_410_0.jpg)

图 8: 最新调仓 Top15 目标权重 (2026-05-20)

最新调仓 Top 权重: 002594.SZ (5.01%)、301419.SZ (4.27%)、301360.SZ (3.94%) 等, 以制造业、券商、新能源为主——与 2026-05 末高 score 板块一致, 可作为模拟赛下单参考。

### 6.5 相对档位 A 基线的综合对比

表 25: 档位 A 基线 vs 档位 C2 (同验证窗; 含策略对齐与相对大盘指标)

<table><tr><td>指标</td><td>档位 A(RGNE-Lite)</td><td>档位 C2 (TFNE-C)</td></tr><tr><td>验证时段</td><td>2025-04 — 2026-05(273 日)</td><td>2025-04-2026-05 (273 日)</td></tr><tr><td>股票池 / \( K \)</td><td>Top 500 / \( K = {30} \)</td><td>Top 3000 / \( K = {50} \)</td></tr><tr><td>全市场 mean IC</td><td>0.016</td><td>0.077</td></tr><tr><td>ICIR</td><td>0.102</td><td>0.794</td></tr><tr><td>Top 10% 内 IC</td><td>0.013</td><td>0.014</td></tr><tr><td>多空 spread(日)</td><td>-0.116%</td><td>0.314%</td></tr><tr><td>IC@K</td><td>0.0303</td><td>-0.0037</td></tr><tr><td>Top-K 命中率</td><td>47.0%</td><td>50.1%</td></tr><tr><td>Top-K 内 spread (日)</td><td>0.040%</td><td>0.086%</td></tr><tr><td>验证集总收益</td><td>32.2%</td><td>43.5%</td></tr><tr><td>验证集 CAGR</td><td>29.4%</td><td>39.6%</td></tr><tr><td>验证集 Sharpe</td><td>1.53</td><td>1.28</td></tr><tr><td>最大回撤 MDD</td><td>-9.2%</td><td>-27.5%</td></tr><tr><td>超额总收益(vs 沪深 300)</td><td>5.9%</td><td>15.0%</td></tr><tr><td>信息比率 IR</td><td>0.67</td><td>0.74</td></tr><tr><td>年化 Alpha / Beta</td><td>5.4% / 1.01</td><td>9.0% / 1.33</td></tr><tr><td>融合: 时序权重</td><td>53.8% (GRU)</td><td>35.3%(TFT)</td></tr><tr><td>融合: 截面权重</td><td>34.1% (MLP)</td><td>38.1%(FTT)</td></tr><tr><td>融合: 新闻权重</td><td>12.1% (Hash)</td><td>26.6%(LoRA)</td></tr></table>

## 综合结论:

1. 预测层:Transformer 升级 + Top3000 + 三 seed 使 IC/ICIR 大幅提升, 是本次实验最坚实的改进；

2. 新闻层: LoRA 微调使新闻从「几乎无效」变为「近三分之一权重」，验证多模态设计的必要性;

3. 组合层:同验证窗上 C2 总收益 43.5%、超额 IR 0.74 优于 A，但 MDD -27.5% 与 \( \beta  = {1.33} \) 提示需控制市场暴露；A Sharpe 1.53 更高源于窄池与低 beta;

4. 策略对齐层:A 验证 IC@K(0.0303)优于 C2(-0.0037)，C2 的 Top- \( K \) 内 spread (21.8%) 更厚; A 的 signal_ann 与回测背离, 组合结论以回测为准;

5. 与 \( \mathrm{A} \) 的关系: \( \mathrm{A} \) 证明了流水线可复现； \( \mathrm{C} \) 在相同策略框架下替换模型与规模，主要改善排序指标，组合表现需结合市场环境解读。

### 6.6 IC 与组合收益的关系(本实验数据)

与档位 A 基线报告类似, 本实验亦存在 「IC 0.077 不算极高, 但 273 日总收益 40%」 的现象:

- IC 衡量 Top3000 全体排序; 策略只用 Top 10%(≈300 只)且最多持 50 只;

- spread 年化 79% 是未扣费多空参考, 实际组合 40% 总收益低于 spread 上限, 符合成本与只做多约束;

- 2025-04 极端下跌日 -12% 说明排序再好也无法完全免疫系统性风险, Sharpe 1.28 已部分反映该风险。

## 7 更新数据与最新策略预测

训练完成后，模拟赛与日常调仓均不再重训，固定使用 config.server_c2_72h.yaml 与 tier_c2 checkpoint。

### 7.1 流水线总览

---

							../data/*.csv(追加最新行情与新闻)

		↓01_build_panel → 02_make_features → 02b_embed_news (CPU)

↓04_ensemble_predict (GPU: TFT/FTT/News checkpoint 推理 + 秩融合)

			↓07_predict_latest(读融合 score → orders_YYYYMMDD. csv)

---

图 9: 更新数据后的预测与调仓流水线 (不重训)

### 7.2 本机命令 (Windows PowerShell)

## (1)数据未变，仅刷新调仓

---

	conda activate ai25

	cd code

python scripts/07_predict_latest.py --config config.server_c2_72h.yaml

---

## (2)CSV 有新日期，扩展融合 score

conda activate ai25

cd code

python scripts/01_build_panel.py --config config.server_c2_72h.yaml

python scripts/02_make_features.py --config config.server_c2_72h.yaml

python scripts/02b_embed_news.py --config config.server_c2_72h.yaml

python scripts/04_ensemble_predict.py --config config.server_c2_72h.yaml

python scripts/07_predict_latest.py --config config.server_c2_72h.yaml

04 已支持 News checkpoint 自动推理, 通常无需重训 News 模态。

### 7.3 注意事项

- 配置不可混用:勿用档位 A 的 config.tier_a_local.yaml 或 tier_a checkpoint 加载本实验权重；

- 融合权重:默认读 fusion_weights.json，删除该文件才会在 valid 段重估 ICIR 权重;

- 换股票池或改 L: 需重新 prep + 全量训练。

## 8 规范性与可复现性

表 26: 复现命令 (服务器训练 + 本机增量预测)

步骤

工作目录:/home/scc/pb23151824/pro/code

python scripts/run_server_c2_72h.py prep

python scripts/run_server_c2_72h.py train && ... eval

配置固定为 config.server_c2_72h.yaml

本机出图: conda activate ai25;

python tier_c2_project/scripts/plot_tier_c2_report.py 本机预测: 见第 7 节

档位 A 基线复现: 见 基线报告.tex;

python scripts/run_tier_a_local.py

- 作业日志: logs/train_7945.out; 完成标志 [CHECK] train+eval OK;

- RoBERTa 加载时 cls.predictions.* 显示 UNEXPECTED 为正常现象。

## 9 总结与反思

### 9.1 主要结论

1. TFNE-C 在 RTX 5090 上约 39 小时完成 Top3000 × 3 seed 全流水线，工程可行；

2. 相对同口径档位 A 基线, FT-Transformer 与 NewsLoRA 显著抬升融合 IC (0.016 \( \rightarrow  {0.077}) \) 与 ICIR (0.102 \( \rightarrow  {0.794} \) );

3. 新闻模态从近零权重升至 \( {26.6}\% \) ，验证 RoBERTa 微调优于哈希新闻编码；

4. 验证集 273 日回测总收益 43.5%、Sharpe 1.28，组合层具备一定实战价值，但 MDD -27.5% 仍需关注；

5. 回测引擎已纳入部分 A 股现实约束 (见下节)，但仍有简化假设，模拟赛结果可用于进一步校验;

6. 赛前 holdout 切分更贴近模拟赛, FDL2026 实战表现待赛后补充。

### 9.2 回测现实约束: 已考虑什么、还缺什么

作业与档位 A 均强调:历史回测应尽可能贴近真实交易。对照 src/backtest.py 与 src/strategy.py, 对本实验回测逐项自检如下。

表 27: 回测现实约束自检 (backtest.py/strategy.py)

<table><tr><td>约束</td><td>状态</td><td>本实验实现方式</td><td>说明</td></tr><tr><td>手续费</td><td>✓</td><td>买卖均扣佣金 0.015% (commission=0.00015)</td><td>每笔成交按金额比例扣费</td></tr><tr><td>印花税</td><td>✓</td><td>仅 卖 出 扣 0.05% (stamp_tax=0.0005)</td><td>符合 A 股卖出单边征收</td></tr><tr><td>滑点</td><td>✓</td><td>买 卖 均 扣 0.05% (slippage=0.0005)</td><td>固定比例，未区分买卖方向</td></tr><tr><td>最小交易单位</td><td>✓</td><td>100 \( \lfloor \left| {\Delta V}\right| /\left( {100P}\right) \rfloor  \times  {100} \)</td><td>零股丢弃，可能留现金碎片</td></tr><tr><td>T+1 可卖</td><td>✓</td><td>当 日 买 sellable=False, 日才可卖</td><td>符合 A 股 \( \mathrm{T} + 1 \)</td></tr><tr><td>T+1 成交时点</td><td>✓</td><td>\( t \) 日信号, \( t + 1 \) 日开盘价成交</td><td>避免用 \( t \) 日收盘价即时成交的泄露</td></tr><tr><td>涨停不买</td><td>✓</td><td>\( t + 1 \) 日 pct_chg≥ 9.5% 跳过买入</td><td>简化 阈 值， 未 区 分 10%/20% 板</td></tr><tr><td>停牌/无量</td><td>✓</td><td>tradable_universe 要求 vol>0</td><td>成交量为 0 则不进入可买池</td></tr><tr><td>ST / 北交所</td><td>✓</td><td>候选池剔除 is_st、.BJ</td><td>与 prep 规则一致</td></tr><tr><td>流动性门槛</td><td>✓</td><td>20 日平均成交额 \( \geq  {5000} \) 万</td><td>与面板构建一致</td></tr><tr><td>跌停不卖</td><td>✘</td><td>未实现</td><td>跌停日仍可按开盘价卖出， 可能高估可执行性</td></tr><tr><td>分板块涨跌幅</td><td>✘</td><td>统一 9.5% 阈值</td><td>科创/创业 20%、ST 5% 未区分</td></tr><tr><td>成交量容量</td><td>✘</td><td>未限制单笔占当日成交量比例</td><td>大单可「无限」按开盘价成交</td></tr><tr><td>最低佣金</td><td>✘</td><td>无「每笔最低 5 元」规则</td><td>小单成本可能被低估</td></tr><tr><td>冲击成本</td><td>✘</td><td>仅固定滑点</td><td>未随订单规模动态放大</td></tr><tr><td>软换仓滞后</td><td>部分</td><td>delta/omega_max</td><td>非规则缺失，但导致持仓数远超 n_max</td></tr></table>

总体判断:本实验回测已经考虑手续费、印花税、滑点、100 股整手、T+1、涨停不买、停牌(无量)过滤等核心约束，并非「理想化零成本回测」。报告中的 Sharpe 1.28、 总收益 43.5% 是在扣费后净值上计算的。但仍存在若干简化，尤其跌停卖出与流动性容量未建模, 极端行情下回测收益可能略偏乐观。

### 9.3 若进一步加入缺失约束:如何做、有何意义

(1)跌停，一字板:卖出失败 现状:仅限制「涨停不买」，未限制「跌停不卖」。2025-04-07 类暴跌日中，若持仓大面积跌停，实盘往往无法按目标价卖出，回测却仍可卖出。

建议实现:在 rebalance_day 卖出分支增加:若 pct_chg≤ \( - {9.5}\% \) (或结合 open==high 判断一字跌停)，则 shares_sell=0；科创/创业板可用 ±19.5%、ST 用 ±4.5%。

意义:更真实反映尾部风险，MDD 与 Sharpe 可能进一步下降，但避免回测过度美化——对评估 2025-04 单月 -17% 类情景尤为重要。

(2)分板块涨跌停阈值 现状:主板、科创、创业、ST 统一用 9.5% 过滤买入。

建议实现:根据 ts_code 前缀或 basic 中板块字段选择阈值 (10% / 20% / 5%)。

意义:Top3000 含大量创业板/科创板, 统一 9.5% 会误判部分 「未封板但涨幅较大」的可买标的，或漏判 20% 板涨停。

(3)成交量/成交额容量约束 现状:假设任意规模订单均可按开盘价完全成交。

建议实现: 单笔买入不超过当日成交量 (或成交额) 的 \( \alpha \) (如 \( 1\%  - 5\% \) ),超出部分顺延至后续交易日；或采用 VWAP 近似。

意义:本策略 n_max=50、单票上限 5%，对中小票通常可成交；但若 scale 到更大资金或更小流动性股票，容量约束会显著压低可执行 alpha——加入后可检验策略资金容量。

(4)最低佣金与过户费 现状:纯比例费率，无单笔下限。

建议实现:fee \( = \max \) (amount \( \times \) rate,5元)；必要时加过户费(现行极低，可忽略)。

意义:对小市值、小偏离调仓(delta=1%边缘单)成本更真实，长期换手略降、净值略低。

(5)动态滑点 / 冲击成本 现状:固定 0.05% 滑点，与订单大小无关。

建议实现: slip \( = {\operatorname{slip}}_{0} + \kappa  \cdot  \left( {\text{ order\_vol }/\text{ ADV }}\right) \) ,其中 ADV 为平均日成交额。

意义:反映「大单推价」效应；对验证 Top3000 宽基策略在大资金下的可扩展性有意义。

(6)完善软换仓与目标持仓对齐 现状:omega_max=25% 使旧仓无法快速清掉，n_hold 可升至 200+，偏离 n_max=50 的设计意图。

建议实现:对「不在目标 Top50 且权重应为 0」的 legacy 持仓设强制减仓通道(不受 delta 限制、或单独更高卖出行配额)；或降低 omega_max 做敏感性分析。

意义:使回测持仓结构与 orders_*.csv 输出一致，组合收益更贴近「50 只核心组合」而非「累积杂仓」。

### 9.4 现实约束对报告结论的影响(定性)

- 已扣费:43.5% 总收益已含佣金/印花税/滑点，不是 gross spread 的 79% 年化；

- 涨停/停牌:减少「买不到」的乐观假设，2025 年部分反弹日可能略低估收益；

- 未建模跌停卖不出:2025-04-06 回撤段可能略低估实盘困难，真实 MDD 或略大于 -27.5%;

- 无容量限制:100 万本金下影响有限；若放大到数千万，应优先加容量与冲击成本；

- 模拟赛:同花顺平台另有委托排队、部分成交等规则，应以实战截图与回测交叉验证。

### 9.5 局限与后续工作

- 与档位 A 已对齐验证时段; 股票池、持仓参数仍不同, 对比需分段表述;

- 不设独立 test 段; 样本外检验依赖 FDL2026 模拟赛(2026-06-01 至 06-10)，该段不写入 IC/回测表；

- 回测可加强:跌停不卖、分板块涨跌幅、成交量容量、最低佣金(见表 27)；

- news_max_length 为 128 ;04 全量推理约 8 h, 增量 predict 可优化;

- TFT 单 seed 间 IC 方差仍较大, 可探索更长训练或架构改进。

## 参考文献

1. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. Review of Financial Studies.

2. Gorishniy, Y., et al. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS (FT-Transformer).

3. Lim, B., et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.

4. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.

## 10 主要产物路径

<table><tr><td>内容</td><td>路径</td></tr><tr><td>本实验配置</td><td>code/config.server_c2_72h.yaml</td></tr><tr><td>IC 汇总</td><td>code/artifacts/metrics/tier_c2/ic_summary.json</td></tr><tr><td>融合权重</td><td>code/artifacts/predictions/tier_c2/fusion_weights.json</td></tr><tr><td>融合 score</td><td>code/artifacts/predictions/tier_c2/ensemble_scores.parquet</td></tr><tr><td>回测指标</td><td>code/artifacts/backtest/tier_c2/valid/metrics.json</td></tr><tr><td>净值曲线</td><td>code/artifacts/backtest/tier_c2/valid/equity_curve.csv</td></tr><tr><td>最新订单</td><td>code/artifacts/orders/tier_c2/orders_20260520.csv</td></tr><tr><td>模型 checkpoint</td><td>code/artifacts/checkpoints/tier_c2/\{tft, ftt, news\}/seed_*/best.pt</td></tr><tr><td>报告图表</td><td>tier_c2_project/figures/fig*.png</td></tr><tr><td>档位 A 基线报告</td><td>基线报告.tex</td></tr></table>