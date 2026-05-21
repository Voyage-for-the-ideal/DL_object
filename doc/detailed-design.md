# 基于深度学习的 A 股短期趋势预测与模拟交易详细设计文档

## 1. 文档范围

本文档基于 `doc/proposal.md` 编写，用于指导后续编码、测试、实验和报告撰写。设计目标是构建一套可复现、可回放、尽量避免未来信息泄露的量化研究与模拟交易系统。

本文档覆盖以下内容：

- 数据读取、交易日历、股票池过滤和样本构造。
- 量价、基本面、资金流、新闻文本特征工程。
- 标签、数据集、预处理和未来信息泄露控制。
- 模型训练、评估、回测和每日预测。
- 策略生成、交易约束、输出文件和模块测试方案。

本文档不固定具体实验参数，例如手续费、滑点、复杂策略权重阈值等。此类参数均写入配置文件，由实验阶段调参确认。

## 2. 已确认设计决策

| 事项 | 设计决策 |
| --- | --- |
| 小股票池验证阶段 | 使用沪深 300 股票池 |
| 官方股票池 | 排除 ST 股票和北交所股票 |
| 数据时间范围 | 使用 2019-01-01 至最新数据 |
| 训练区间 | 2019-01-01 至 2025-12-31 |
| 验证 / 回测区间 | 2026-01-01 至 2026-05-20 |
| 主标签 | `label_1d(T) = close(T+2) / close(T+1) - 1` |
| 辅助标签 | 未来 3 日、未来 5 日收益率 |
| 新闻第一版方案 | TF-IDF |
| 新闻扩展方案 | 使用现有中文 Transformer / 金融文本模型做情绪或文本向量 |
| 主策略 | 分数加权 + 风险约束 |
| 对照策略 | TopK 等权组合 |
| 多策略 / 多模型组合 | 放入扩展设计 |
| 交易成本 | 手续费、滑点、印花税等写成配置项，不在本文档固定具体数值 |

## 3. 公开资料调研后的模型设计原则

截至 2026-05-21，可公开验证的信息无法证明某一家顶级量化机构正在生产环境中使用某一个固定模型。顶级机构的真实投研模型通常不完全公开。因此本文档不声称复刻任何机构的内部策略，只参考公开研究和开源量化平台形成工程设计原则。

可参考的公开资料包括：

- Microsoft Qlib 是面向 AI 量化研究的开源平台，公开 benchmark 覆盖传统机器学习、深度学习时序模型、图模型、Transformer 类模型、TFT、TRA、DoubleEnsemble 等多类模型，而不是只押单一网络结构。参考：[Qlib](https://github.com/microsoft/qlib)、[Qlib Benchmarks](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)。
- Gu、Kelly、Xiu 的机器学习资产定价研究表明，非线性模型、树模型和神经网络可用于捕捉传统线性因子模型难以表达的关系，但评估必须放在严格的样本外框架下。参考：[Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article/33/5/2223/5758276)。
- BlackRock、Man AHL 等公开材料强调系统化投资中会结合大规模数据、机器学习、文本信息、风险管理和组合构建，而不是仅比较单个预测模型的误差。参考：[BlackRock Systematic Investing](https://www.blackrock.com/sg/en/investment-strategies/systematic-investing)、[The Rise of Machine Learning at Man AHL](https://www.man.com/insights/the-rise-of-machine-learning)、[Man AHL](https://www.man.com/man-ahl-team)。
- Bridgewater 的公开投资思想更强调资产和风险来源的分散，而不是公开某个预测模型。本文档只将其作为“多信号、多策略、风险预算”的扩展设计参考，不宣称复制 Bridgewater 策略。参考：[Bridgewater All Weather Story](https://www.bridgewater.com/research-and-insights/the-all-weather-story)。
- Temporal Fusion Transformer 等时序模型提供了多步预测、变量选择和解释性设计思路，可作为后续模型增强方向。参考：[Temporal Fusion Transformers](https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/)。

据此，模型模块采用以下原则：

1. 不将 Transformer 作为唯一主线。
2. 建立统一模型接口，使线性模型、树模型、MLP、序列模型、Transformer 和集成模型可以共享数据集、训练、评估和回测流程。
3. 第一版必须包含至少一个神经网络模型，并保留强基线，避免只和弱模型比较。
4. 最终模拟盘不直接依赖单模型收益，而使用统一信号评分、风险约束和回测指标进行选择。
5. 多模型组合、市场状态自适应和风险预算放入扩展设计，避免第一版工程过大。

## 4. 总体架构

系统采用离线研究 + 每日预测两条流程。

```text
原始 CSV 数据
  -> 数据读取与日历对齐
  -> 股票池过滤
  -> 特征工程
  -> 预处理与样本构造
  -> 模型训练
  -> 预测评分
  -> 指标评估
  -> 回测引擎
  -> 每日信号与订单建议
```

建议代码结构：

```text
src/
  config/
    default.yaml
  data/
    loader.py
    calendar.py
    universe.py
    news_features.py
    schema.py
  features/
    price_features.py
    metric_features.py
    moneyflow_features.py
    preprocess.py
    feature_store.py
  datasets/
    tabular_dataset.py
    window_dataset.py
    split.py
  models/
    base.py
    linear.py
    gbdt.py
    mlp.py
    sequence.py
    transformer.py
    ensemble.py
  training/
    trainer.py
    losses.py
    callbacks.py
  evaluation/
    metrics.py
    reports.py
  backtest/
    engine.py
    portfolio.py
    strategy.py
    execution.py
    metrics.py
  predict/
    daily_signal.py
    daily_order.py
  utils/
    logging.py
    seed.py
    io.py
README.md
requirements.txt
```

每个模块只通过明确的数据结构通信，禁止直接在模型、策略或回测代码中读取原始 CSV。这样可以保证模块可独立测试。

## 5. 配置设计

统一配置文件为 `src/config/default.yaml`。实验阶段可以复制出不同配置文件。

核心配置项：

```yaml
data:
  root: "A股数据"
  start_date: "2019-01-01"
  train_end_date: "2025-12-31"
  valid_start_date: "2026-01-01"
  valid_end_date: "2026-05-20"
  universe_mode: "hs300"
  official_universe_exclude_st: true
  official_universe_exclude_bse: true

label:
  main: "label_1d"
  horizons: [1, 3, 5]

features:
  lookback_windows: [5, 10, 20]
  use_price: true
  use_metric: true
  use_moneyflow: true
  use_news_tfidf: true
  news_tfidf_max_features: 128
  news_cutoff_time: null

preprocess:
  winsorize: true
  winsorize_method: "train_quantile"
  lower_quantile: 0.01
  upper_quantile: 0.99
  impute_method: "train_median"
  scale_method: "train_standard"

dataset:
  type: "window"
  lookback: 20
  min_history_days: 20

model:
  name: "transformer_encoder"
  task: "regression"
  input_dim: null
  hidden_dim: 128
  dropout: 0.1

training:
  seed: 42
  batch_size: 1024
  epochs: 50
  learning_rate: 0.001
  loss: "mse"
  early_stopping_patience: 5
  device: "auto"

backtest:
  initial_cash: 1000000
  execution_price: "open"
  commission_rate: null
  stamp_tax_rate: null
  slippage_rate: null
  min_lot_size: 100
  enforce_t_plus_one: true
  block_limit_up_buy: true
  block_limit_down_sell: true
  target_full_invested: true

strategy:
  name: "score_weighted_risk_control"
  benchmark_strategy: "topk_equal_weight"
  top_k: 20
  max_single_weight: 0.10
  max_industry_weight: 0.30
  max_turnover: null
  min_amount: null

outputs:
  root: "outputs"
```

其中 `commission_rate`、`stamp_tax_rate`、`slippage_rate`、`max_turnover`、`min_amount` 等不在本文档固定数值。

## 6. 数据模块详细设计

### 6.1 `data.loader`

职责：

- 读取 `basic.csv`、`trade_cal.csv`。
- 按日期读取 `daily/`、`metric/`、`moneyflow/`、`stock_st/`、`index_weight/`、`market/`、`news/`。
- 将日期字段统一为 `YYYYMMDD` 或 `datetime64`，内部处理时保持一致。
- 对读取结果做字段检查，不在加载阶段做特征工程。

主要接口：

```python
class CsvDataLoader:
    def load_basic(self) -> pd.DataFrame: ...
    def load_trade_calendar(self) -> pd.DataFrame: ...
    def load_daily(self, trade_date: str) -> pd.DataFrame: ...
    def load_metric(self, trade_date: str) -> pd.DataFrame: ...
    def load_moneyflow(self, trade_date: str) -> pd.DataFrame: ...
    def load_st(self, trade_date: str) -> pd.DataFrame: ...
    def load_index_weight(self, trade_date: str | None = None) -> pd.DataFrame: ...
    def load_market(self, index_code: str) -> pd.DataFrame: ...
    def load_news(self, date: str) -> pd.DataFrame: ...
```

输入输出约定：

- 所有按日截面数据必须包含 `trade_date`。
- 个股数据必须包含 `ts_code`。
- 不存在的日期文件返回空 DataFrame，并由调用方决定是否跳过或报错。

独立测试：

- 读取一个已知日期的 `daily` 文件，检查字段完整。
- 读取不存在日期时不导致程序崩溃。
- 日期格式转换前后交易日排序一致。

### 6.2 `data.calendar`

职责：

- 根据 `trade_cal.csv` 生成交易日序列。
- 支持自然日和交易日转换。
- 提供 `T+1`、`T+2`、`T+n` 查询。
- 判断日期是否交易日。

主要接口：

```python
class TradingCalendar:
    def is_trading_day(self, date: str) -> bool: ...
    def previous_trade_date(self, date: str) -> str: ...
    def next_trade_date(self, date: str, n: int = 1) -> str: ...
    def trade_dates_between(self, start: str, end: str) -> list[str]: ...
```

独立测试：

- 对节假日和周末返回非交易日。
- `next_trade_date(T, 2)` 与交易日列表中的索引偏移一致。
- 回测区间首尾日期可被正确展开。

### 6.3 `data.universe`

职责：

- 构造沪深 300 小股票池。
- 构造官方比赛股票池。
- 按日排除 ST 股票。
- 排除 `basic.csv` 中 `market = 北交所` 的股票。
- 过滤上市时间过短、停牌、成交额异常低股票。

主要接口：

```python
class UniverseBuilder:
    def get_hs300_universe(self, trade_date: str) -> set[str]: ...
    def get_official_universe(self, trade_date: str) -> set[str]: ...
    def get_tradeable_universe(self, trade_date: str, mode: str) -> set[str]: ...
```

沪深 300 股票池设计：

- 使用 `index_weight/` 中 `index_code = 000300.SH` 的成分股。
- 若某交易日没有完全对应的权重文件，使用不晚于该日的最近一期成分股。
- 该回退逻辑只能向过去查找，不能使用未来成分股。

官方股票池设计：

- 从 `basic.csv` 中排除北交所股票。
- 按 `stock_st/` 的当日列表排除 ST 股票。
- 停牌通过当日 `daily` 缺失、成交量为 0 或成交额异常低进行过滤。

独立测试：

- 北交所股票永远不进入官方股票池。
- 某日 ST 股票只在该日及对应状态日期被过滤，不用未来 ST 状态过滤历史样本。
- 沪深 300 成分回退只使用历史日期。

### 6.4 `data.news_features`

职责：

- 读取 `news/` 中每日新闻。
- 将新闻按预测信号日聚合。
- 第一版生成 TF-IDF 特征。
- 扩展版支持 Transformer 文本情绪或 embedding。

第一版限制：

- 当前新闻数据字段为 `datetime`、`content`、`title`，没有股票代码字段。
- 因此第一版新闻特征作为市场级文本特征，按日期合并到所有股票样本。
- 股票级新闻匹配属于扩展功能，需通过股票名称、简称、行业关键词或外部实体识别实现。

TF-IDF 设计：

- 训练期拟合 `TfidfVectorizer`。
- 验证期、回测期和每日预测只调用训练期保存的 vectorizer。
- 每个交易日聚合标题和正文，生成固定维度文本向量。
- 附加新闻数量、标题数量、正文长度均值等统计特征。

未来信息控制：

- 预测 `T+1` 的信号日期为 `T`。
- 新闻聚合只允许使用 `T` 日信号生成时刻之前可获得的新闻。
- `news_cutoff_time` 不在本文档固定，写入配置；若为空，则由实验脚本明确记录实际口径。

独立测试：

- TF-IDF 不在验证期重新拟合。
- 2026 年某日特征不能包含该日之后新闻。
- 没有新闻的日期输出全零文本向量和新闻数量 0。

## 7. 特征模块详细设计

### 7.1 特征表统一格式

所有特征模块输出统一格式：

```text
trade_date, ts_code, feature_1, feature_2, ...
```

合并后的训练面板主键为：

```text
(trade_date, ts_code)
```

其中 `trade_date = T` 表示信号生成日，模型可使用 `T` 日收盘后已知的数据预测 `T+1` 之后的收益。

### 7.2 `features.price_features`

输入：

- `daily/` 合并后的个股日频量价数据。

核心特征：

- 日收益率：`close / pre_close - 1`。
- 开盘到收盘收益率：`close / open - 1`。
- 最高最低振幅：`high / low - 1`。
- 成交量变化率。
- 成交额变化率。
- VWAP 偏离度：`close / vwap - 1`。
- 过去 5 / 10 / 20 日收益率均值、波动率、动量。
- 过去 5 / 10 / 20 日成交量均值和成交额均值。

未来信息控制：

- 对样本日 `T` 的滚动特征只能使用 `<= T` 的数据。
- 标签相关的 `T+1`、`T+2` 收盘价不得进入任何特征。

独立测试：

- 构造一个 5 行人工序列，验证第 3 行滚动特征不读取第 4、5 行。
- `close(T+1)` 改变时，`T` 日特征不变化。

### 7.3 `features.metric_features`

输入：

- `metric/` 每日基本面指标。

核心特征：

- 换手率、自由流通换手率、量比。
- PE、PE_TTM、PB、PS、PS_TTM。
- 股息率。
- 总市值、流通市值。
- 市值对数特征。
- 市值分组或截面排名特征。

处理原则：

- 对 PE 等可能缺失或极端的指标进行缺失值处理和截尾。
- 截尾阈值由训练期或历史滚动窗口拟合，不能使用全样本。

独立测试：

- 亏损公司 PE 缺失时能被正确填充。
- 极端值裁剪阈值不由验证期数据决定。

### 7.4 `features.moneyflow_features`

输入：

- `moneyflow/` 每日资金流数据。

核心特征：

- 小单、中单、大单、特大单买卖差额。
- 各类资金净流入额。
- 净流入额占成交额比例。
- 过去 5 / 10 / 20 日资金流滚动均值。
- 资金流强度截面排名。

独立测试：

- 净流入计算与原始字段一致。
- 滚动资金流特征只使用历史窗口。

### 7.5 `features.preprocess`

职责：

- 缺失值填充。
- 极端值截尾。
- 标准化或排名化。
- 保存预处理器参数。

预处理模式：

| 模式 | 说明 | 泄露风险控制 |
| --- | --- | --- |
| `train_standard` | 用训练期均值方差标准化 | 验证期只 transform |
| `train_median` | 用训练期中位数填充 | 验证期只 transform |
| `train_quantile` | 用训练期分位数截尾 | 验证期只 transform |
| `daily_rank` | 当日截面排名 | 只使用信号日当日截面 |
| `rolling_history` | 使用历史滚动窗口统计量 | 窗口右端不超过 T |

推荐第一版：

- 数值特征先做训练期分位数截尾。
- 缺失值使用训练期中位数填充。
- 标准化使用训练期均值方差。
- 可额外保存日内截面 rank 特征，用于增强排序能力。

独立测试：

- 预处理器 `fit` 只接收训练期数据。
- 验证期新增极端值不会改变训练期裁剪阈值。
- 保存和加载预处理器后，同一输入得到同一输出。

### 7.6 `features.feature_store`

职责：

- 缓存中间特征，避免重复读取和计算。
- 记录特征生成配置、日期范围和字段列表。
- 支持按日期读取训练面板。

缓存目录：

```text
outputs/cache/features/
```

缓存不提交。若配置变化，缓存必须失效或写入新的配置哈希目录。

## 8. 标签与数据集设计

### 8.1 标签构造

主标签：

```text
label_1d(T) = close(T+2) / close(T+1) - 1
```

辅助标签：

```text
label_3d(T) = close(T+4) / close(T+1) - 1
label_5d(T) = close(T+6) / close(T+1) - 1
```

设计解释：

- `T` 为信号生成日。
- `T+1` 为实际买入日。
- `T+2` 为主标签最早可卖出日。
- 标签只用于训练和评估，不能参与特征工程。

样本过滤：

- 若 `T+1` 或标签结束日没有交易数据，则该样本不能用于训练该 horizon。
- 若股票在 `T` 不属于当前股票池，则不生成样本。
- 若股票在 `T+1` 无法买入，回测时不执行交易；训练样本是否剔除由配置控制。

独立测试：

- 人工构造 4 个交易日价格，验证 `label_1d(T)` 使用第 2 和第 3 个未来交易日。
- 改变 `close(T)` 不影响标签，改变 `close(T+1)` 或 `close(T+2)` 会影响标签。

### 8.2 `datasets.tabular_dataset`

适用模型：

- 线性模型。
- LightGBM / XGBoost / CatBoost。
- MLP。

样本格式：

```text
X: [num_samples, num_features]
y: [num_samples]
index: [trade_date, ts_code]
```

### 8.3 `datasets.window_dataset`

适用模型：

- TCN / GRU / LSTM。
- Transformer Encoder。
- TFT 或其他多步时序模型扩展。

样本格式：

```text
X: [num_samples, lookback, num_features]
y: [num_samples]
index: [trade_date, ts_code]
```

窗口定义：

- 对信号日 `T`，输入窗口为 `[T-lookback+1, ..., T]`。
- 窗口内任意缺失日期按交易日序列对齐。
- 缺失股票历史可填充、掩码或剔除，具体由配置控制。

独立测试：

- `lookback = 20` 时，每个样本最后一天必须等于信号日 `T`。
- 窗口不能包含 `T+1` 或之后日期。

### 8.4 时间切分

默认切分：

```text
训练期：2019-01-01 至 2025-12-31
验证 / 回测期：2026-01-01 至 2026-05-20
```

训练内部可选再划分一段 `inner_valid` 用于早停，但不能改变最终报告中对 2026 验证 / 回测区间的记录。

禁止事项：

- 禁止随机把 2026 样本混入训练。
- 禁止在全样本上拟合 scaler、imputer、TF-IDF、截尾阈值。
- 禁止用未来股票池状态过滤历史样本。

## 9. 模型模块详细设计

### 9.1 统一模型接口

所有模型实现统一接口：

```python
class BaseAlphaModel:
    def fit(self, train_data, valid_data, config) -> "ModelArtifact": ...
    def predict(self, data) -> pd.DataFrame: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "BaseAlphaModel": ...
```

预测输出统一为：

```text
trade_date, ts_code, score, model_name
```

`score` 是连续预测分数，用于 IC、排序、策略权重和集成。

### 9.2 第一版模型清单

第一版不把模型限制为课程示例中列出的 MLP / GRU / Transformer，而采用“强基线 + 神经网络 + 时序模型”的结构。

| 模型 | 类型 | 用途 | 是否第一版实现 |
| --- | --- | --- | --- |
| Ridge / ElasticNet | 线性基线 | 检查特征是否有线性预测能力 | 是 |
| LightGBM Regressor | 树模型强基线 | 检查神经网络是否优于成熟表格模型 | 是 |
| MLP | 深度学习基线 | 满足神经网络要求，处理截面特征 | 是 |
| Transformer Encoder | 深度时序模型 | 主深度模型候选，处理历史窗口 | 是 |
| GRU / LSTM / TCN | 深度时序模型 | 中间对照或算力受限替代 | 预留接口，可选实现 |
| TFT / TRA / GAT | 高级模型 | 多步、适应性、图结构建模扩展 | 扩展设计 |
| Model Ensemble | 集成模型 | 多模型组合和稳健性提升 | 扩展设计 |

设计理由：

- Ridge / ElasticNet 便于解释和定位数据问题。
- LightGBM 是强表格基线，如果神经网络无法超过它，报告中仍可进行有效分析。
- MLP 提供最简单的深度学习基线。
- Transformer Encoder 对历史窗口建模能力强，符合短期趋势预测任务。
- GRU / LSTM / TCN 不作为第一版硬性范围，避免工程过大；若 Transformer 训练成本过高，可启用为替代时序模型。

### 9.3 `models.linear`

输入：

- `TabularDataset`。

模型：

- Ridge。
- ElasticNet。

输出：

- 连续收益预测分数。

用途：

- 作为最简单可解释基线。
- 快速检查标签、特征和切分是否正确。

### 9.4 `models.gbdt`

输入：

- `TabularDataset`。

模型：

- 第一版优先 LightGBM Regressor。
- 若环境安装困难，可降级为 scikit-learn HistGradientBoostingRegressor。

用途：

- 强基线。
- 评估表格特征质量。
- 输出特征重要性，辅助报告分析。

### 9.5 `models.mlp`

输入：

- `TabularDataset`。

结构：

```text
Input
  -> Linear + BatchNorm + ReLU + Dropout
  -> Linear + BatchNorm + ReLU + Dropout
  -> Linear
  -> score
```

训练：

- 损失函数默认 MSE。
- 可选 HuberLoss。
- 可选 RankIC 相关损失作为扩展。

独立测试：

- 随机输入 batch 可完成 forward。
- 输出 shape 为 `[batch_size]` 或 `[batch_size, 1]`。
- 固定 seed 时重复训练的小样本结果可复现到可接受范围。

### 9.6 `models.transformer`

输入：

- `WindowDataset`，形状 `[batch, lookback, feature_dim]`。

结构：

```text
Input Window
  -> Linear Feature Projection
  -> Positional Encoding
  -> Transformer Encoder Blocks
  -> Last Token Pooling 或 Attention Pooling
  -> Regression Head
  -> score
```

关键参数：

- `lookback`。
- `hidden_dim`。
- `num_layers`。
- `num_heads`。
- `dropout`。
- `pooling`。

设计约束：

- 只做历史窗口编码，不使用未来 mask 之外的信息。
- 输出仍是单个连续分数。
- 不在模型内部做股票池过滤或标签构造。

独立测试：

- 输入窗口长度变化时，模型输出维度保持正确。
- 同一窗口预测不依赖 batch 中其他未来日期样本。
- 保存后加载模型，预测分数一致。

### 9.7 扩展模型设计

扩展模型不作为第一版必须交付，但接口需预留。

#### 9.7.1 GRU / LSTM / TCN

用途：

- 作为 Transformer 之外的低成本时序模型。
- 当数据量或算力不足时用于对照。

#### 9.7.2 TFT

用途：

- 多 horizon 预测。
- 加入静态特征，如行业、市值分组。
- 提供变量选择和解释性分析。

#### 9.7.3 GAT / 图模型

用途：

- 使用行业、指数成分、相关性网络构建股票图。
- 建模股票之间的截面关系。

#### 9.7.4 TRA / 自适应模型

用途：

- 针对金融时间序列非平稳性，让模型在不同市场状态下使用不同预测头。

#### 9.7.5 多模型组合

扩展组合方式：

- 每日 rank average。
- 验证期 IC 加权。
- 滚动 IC 加权。
- 按模型回撤和波动动态降权。
- 分因子族组合：价格动量、资金流、估值、新闻文本分别训练模型，再做风险预算。

与 Bridgewater 公开思想的关系：

- 本项目不复制 Bridgewater 策略。
- 只借鉴“不要集中暴露于单一风险来源”的思想，把不同模型和不同信号族看作不同 alpha 来源。
- 组合权重由验证期表现、波动、相关性和回撤约束决定。

## 10. 训练模块详细设计

### 10.1 `training.trainer`

职责：

- 固定随机种子。
- 加载配置、数据集、模型。
- 训练模型。
- 输出日志、曲线、模型配置和轻量 artifact。

流程：

```text
load_config
  -> build_dataset
  -> fit_preprocessors_on_train
  -> transform_train_valid
  -> build_model
  -> train
  -> evaluate_on_train_valid
  -> save_artifacts
```

输出：

```text
outputs/runs/{run_id}/
  config.yaml
  train_log.csv
  metrics.json
  loss_curve.png
  valid_predictions.csv
  model_summary.txt
```

模型权重如体积较大，不作为提交内容；本地可保存到 `outputs/runs/{run_id}/checkpoints/`。

### 10.2 损失函数

第一版：

- MSELoss。
- HuberLoss 可选。

扩展：

- Pairwise rank loss。
- Negative RankIC loss。
- 多 horizon 加权损失。

### 10.3 训练日志

每个 epoch 至少记录：

- train loss。
- valid loss。
- valid IC。
- valid RankIC。
- 学习率。

独立测试：

- 小样本训练 1 个 epoch 能完整输出日志。
- 训练过程中断时已有日志可读取。

## 11. 评估模块详细设计

### 11.1 `evaluation.metrics`

预测指标：

- MSE。
- MAE。
- 日度 IC：同一交易日内 `score` 与 `label` 的 Pearson 相关。
- 日度 RankIC：同一交易日内 `score` 与 `label` 的 Spearman 相关。
- IC 均值。
- ICIR：`mean(daily_ic) / std(daily_ic)`。
- 方向准确率：`sign(score) == sign(label)`。
- 分组收益：按预测分数分成 5 组或 10 组，观察单调性。

输入：

```text
trade_date, ts_code, score, label
```

输出：

```text
outputs/evaluation/{run_id}/
  prediction_metrics.json
  daily_ic.csv
  group_return.csv
  ic_curve.png
  group_return_plot.png
```

独立测试：

- 构造完全正相关样本，IC 为 1。
- 构造完全负相关样本，IC 为 -1。
- 单日股票数量过少时跳过该日 IC，避免 NaN 污染整体指标。

## 12. 回测模块详细设计

### 12.1 回测时间语义

系统内部区分：

- `signal_date = T`：模型在 T 日盘后生成分数。
- `trade_date = T+1`：策略根据上一交易日信号下单。
- `label_end_date`：标签或持有期结束日期。

回测循环以 `trade_date` 为主，但读取 `previous_trade_date(trade_date)` 的信号。

### 12.2 `backtest.engine`

职责：

- 按交易日推进。
- 读取信号、行情、股票池和当前持仓。
- 调用策略生成目标持仓或订单。
- 调用执行模块模拟成交。
- 更新现金、持仓和净值。

核心状态：

```python
PortfolioState:
    cash: float
    positions: dict[str, Position]
    total_value: float
    trade_date: str

Position:
    ts_code: str
    shares: int
    cost_price: float
    last_price: float
    buy_date: str
```

输出：

```text
outputs/backtest/{run_id}/
  nav.csv
  trades.csv
  positions.csv
  orders.csv
  backtest_metrics.json
  nav_curve.png
```

### 12.3 `backtest.execution`

职责：

- 判断订单是否可执行。
- 计算成交价格、手续费、滑点和印花税。
- 处理最小交易单位。
- 执行 T+1、涨跌停、停牌约束。

交易约束：

| 约束 | 设计 |
| --- | --- |
| T+1 | 持仓 `buy_date == trade_date` 时不可卖出 |
| 停牌 | 当日无 `daily` 行或成交量 / 成交额异常时不可交易 |
| 涨停不能买 | 根据当日价格状态或涨跌幅阈值判断 |
| 跌停不能卖 | 根据当日价格状态或涨跌幅阈值判断 |
| 最小交易单位 | 买入股数向下取整到 100 股整数倍 |
| 满仓 | 策略尽量使用现金，但不能违反交易约束 |
| 手续费 / 滑点 / 印花税 | 配置项，不在设计文档固定 |

执行价格：

- `open`、`vwap`、`close` 三种模式由配置选择。
- 默认值写在配置中，实验报告必须说明实际使用口径。

独立测试：

- 当日买入股票当日不能卖出。
- 买入 101 股时实际下单股数应为 100 或按资金约束向下取整。
- 涨停股票买单被拒绝。
- 跌停股票卖单被拒绝。

### 12.4 `backtest.strategy`

#### 12.4.1 TopK 等权对照策略

逻辑：

1. 对当日可交易股票按 `score` 降序排序。
2. 取前 `top_k`。
3. 等权生成目标持仓。
4. 根据执行约束生成订单。

用途：

- 与主策略对照。
- 验证信号排序是否能转化为收益。

#### 12.4.2 分数加权 + 风险约束主策略

输入：

- 当日预测分数。
- 当前持仓。
- 当前现金。
- 当日可交易股票池。
- 行业、市值、成交额等约束特征。

步骤：

1. 过滤不可交易股票。
2. 对预测分数做日内截面 rank 或 z-score。
3. 保留前 `candidate_k` 或分数高于阈值的候选股票。
4. 将分数转换为初始权重。
5. 应用单票权重上限。
6. 应用行业权重上限。
7. 应用流动性过滤或成交额下限。
8. 应用换手率约束。
9. 生成目标持仓。
10. 根据 T+1 和交易约束生成可执行订单。

权重转换可选方案：

- 正分数归一化。
- rank 分数归一化。
- softmax 分数归一化。

第一版建议使用简单可解释的 rank 分数归一化，复杂优化器放入扩展。

独立测试：

- 单票权重不超过配置上限。
- 行业权重不超过配置上限。
- 所有目标权重和不超过 1。
- 当高分股票不可买入时，策略能选择后续候选股票或保留现金。

### 12.5 `backtest.metrics`

回测指标：

- 累计收益率。
- 年化收益率。
- 年化波动率。
- 夏普比率。
- 最大回撤。
- 换手率。
- 平均持仓数量。
- 现金比例。
- 成交失败比例。

基准：

- 沪深 300。
- 上证指数。
- 创业板指数可选。
- MLP + TopK 等权。
- LightGBM + TopK 等权。

独立测试：

- 单调上涨净值最大回撤为 0。
- 已知收益序列的累计收益率和年化收益率计算正确。

## 13. 每日预测模块详细设计

### 13.1 `predict.daily_signal`

职责：

- 自动识别最新可用交易日 `T`。
- 读取 `<= T` 的所有必要数据。
- 使用训练期保存的预处理器和模型。
- 输出下一交易日可使用的预测分数。

输出：

```text
outputs/signals/YYYYMMDD_signal.csv
```

字段：

```text
signal_date, next_trade_date, ts_code, score, rank, model_name
```

运行前检查：

- 最新日期 daily 是否存在。
- 预处理器是否已保存。
- 模型是否可加载。
- 特征字段与训练期一致。
- 没有读取 `T+1` 或之后数据。

### 13.2 `predict.daily_order`

职责：

- 读取最新信号。
- 读取当前持仓和现金。
- 运行主策略。
- 输出模拟盘操作建议。

输出：

```text
outputs/orders/YYYYMMDD_orders.csv
```

字段：

```text
signal_date, trade_date, ts_code, action, target_weight, target_shares,
estimated_price, estimated_amount, reason
```

注意：

- 每日订单建议不保证真实成交。
- 若模拟盘实际成交失败，需要根据交易软件反馈手动调整，并在报告中记录偏差。

## 14. 输出与交付物

代码交付：

- `src/` 源代码。
- `README.md`。
- `requirements.txt`。
- `doc/proposal.md`。
- `doc/detailed-design.md`。

实验输出：

- 训练日志。
- 训练 / 验证损失曲线。
- IC / ICIR 指标。
- 分组收益图。
- 回测净值曲线。
- 回测交易记录。
- 每日预测信号。
- 每日订单建议。

不提交：

- 原始数据。
- 大体积缓存。
- 大体积模型权重。
- 临时文件。

## 15. 模块测试计划

### 15.1 单元测试

| 模块 | 测试重点 |
| --- | --- |
| `data.loader` | 文件读取、字段检查、缺失文件处理 |
| `data.calendar` | 交易日偏移、节假日、区间展开 |
| `data.universe` | ST、北交所、沪深 300 成分回退 |
| `news_features` | TF-IDF 只在训练期拟合、无新闻日期处理 |
| `price_features` | 滚动窗口不使用未来 |
| `metric_features` | 缺失值和极端值处理 |
| `moneyflow_features` | 净流入和滚动均值 |
| `preprocess` | fit / transform 分离 |
| `datasets` | 样本日期、窗口长度、标签对齐 |
| `models` | forward shape、保存加载一致 |
| `evaluation` | IC、ICIR、方向准确率 |
| `execution` | T+1、最小单位、涨跌停、停牌 |
| `strategy` | 权重约束、候选替补、满仓倾向 |

### 15.2 集成测试

最小集成测试使用：

- 少量交易日。
- 少量股票。
- 人工或真实小样本数据。

测试流程：

```text
读取数据
  -> 构造特征
  -> 构造标签
  -> 训练 MLP 1 个 epoch
  -> 输出预测
  -> 跑 3 至 5 个交易日回测
  -> 生成信号和订单文件
```

通过条件：

- 流程不中断。
- 无未来日期读取。
- 输出文件字段完整。
- 回测资产不出现负股数、负现金异常，除非策略明确允许融资；第一版不允许融资。

### 15.3 泄露检查测试

必须实现以下检查：

1. 删除 `T+1` 之后文件后，仍能为 `T` 生成信号。
2. 改变验证期数据后，训练期预处理参数不变化。
3. 改变未来价格后，历史特征不变化。
4. 每日股票池过滤只使用当日及历史状态。
5. TF-IDF 词表不包含只在验证期出现的新词，除非重新训练并明确记录。

## 16. 第一版实现顺序

建议按以下顺序实现：

1. 配置、日志、随机种子。
2. 数据加载、交易日历、股票池。
3. 量价特征、标签、TabularDataset。
4. 预处理器和泄露检查。
5. Ridge / LightGBM / MLP。
6. 预测指标和损失曲线。
7. 回测引擎和 TopK 等权策略。
8. 分数加权 + 风险约束策略。
9. WindowDataset 和 Transformer Encoder。
10. 新闻 TF-IDF 特征。
11. 每日信号与订单输出。
12. README、requirements 和报告图表整理。

该顺序保证每一步都能独立验证，不会等全部模块完成后才发现基础数据或日期对齐错误。

## 17. 扩展设计

扩展方向不影响第一版验收。

### 17.1 新闻 Transformer

使用现有中文 Transformer 或金融文本模型生成：

- 新闻情绪分数。
- 新闻主题向量。
- 市场风险事件特征。

若要做股票级新闻，需要先建立新闻到股票的实体匹配模块。

### 17.2 多模型组合

候选组合：

- MLP + Transformer + LightGBM rank average。
- 按验证期 IC 加权。
- 按滚动 IC 加权。
- 按模型相关性降低同质模型权重。
- 按市场状态选择模型。

### 17.3 多策略组合

候选策略：

- TopK 等权。
- 分数加权。
- 风险约束。
- 低换手版本。
- 止损 / 止盈版本。

组合思想：

- 不把资金集中到单一策略。
- 观察不同策略在上涨、震荡、下跌市场中的表现。
- 使用风险预算而不是只按历史收益率分配权重。

### 17.4 图模型

构图方式：

- 同行业连接。
- 同指数成分连接。
- 历史收益相关性连接。
- 资金流相似性连接。

模型：

- GCN。
- GAT。
- 图注意力 + 时序模型。

### 17.5 组合优化

在主策略基础上进一步引入优化器：

- 最大化预测收益。
- 约束单票权重。
- 约束行业权重。
- 约束换手率。
- 约束跟踪误差。
- 约束流动性。

第一版使用启发式权重生成即可，优化器作为增强项。

## 18. 风险与应对

| 风险 | 影响 | 设计应对 |
| --- | --- | --- |
| 数据读取慢 | 迭代效率低 | 使用沪深 300 先跑通，缓存中间特征 |
| 数据泄露 | 回测虚高 | 预处理 fit / transform 分离，滚动窗口检查 |
| 新闻无股票代码 | 难以做个股文本特征 | 第一版做市场级 TF-IDF，个股新闻匹配放扩展 |
| 神经网络弱于 LightGBM | 结果不理想 | 保留强基线并分析原因，不强行只报告深度模型 |
| Transformer 训练成本高 | 难以迭代 | 保留 MLP 和可选 GRU / TCN 接口 |
| 模拟盘成交失败 | 实盘偏离回测 | 回测加入涨跌停、停牌、滑点、最小单位 |
| 过拟合 | 验证期不稳定 | 时间切分、早停、正则化、多基线对比 |
| 策略过复杂 | 第一版无法完成 | 主策略用启发式风险约束，复杂组合放扩展 |

## 19. 验收对应关系

| 需求验收项 | 设计覆盖 |
| --- | --- |
| 读取 `A股数据/` | `data.loader` |
| 无未来信息泄露样本 | `features.preprocess`、`datasets`、泄露检查测试 |
| 至少一个神经网络模型 | `models.mlp`、`models.transformer` |
| 损失曲线 | `training.trainer` |
| IC / ICIR / 方向准确率 | `evaluation.metrics` |
| 2026 回测 | `backtest.engine` |
| T+1、手续费、滑点等约束 | `backtest.execution` |
| 基准对比 | `backtest.metrics`、TopK、指数基准 |
| 最新日期预测 | `predict.daily_signal` |
| 复现实验 | `README.md`、`requirements.txt`、配置和日志 |
