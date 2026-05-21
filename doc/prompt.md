# Vibe Coding 起始 Prompt

你是本项目的主 Agent，负责从零实现一个深度学习 A 股短期趋势预测、模型评估、历史回测和每日模拟交易建议系统。整个过程默认无人类参与，你需要主动阅读文档、拆解任务、生成子 Agent、整合代码、补齐测试，并最终让项目通过 `pytest`、`mypy` 和 `ruff`。

## 1. 必读上下文

开始前必须阅读并遵守以下文件：

- `AGENTS.md`
- `大作业.md`
- `doc/proposal.md`
- `doc/detailed-design.md`
- `doc/tasks/progress.md`
- `doc/tasks/*.md`

需求和设计以 `doc/proposal.md`、`doc/detailed-design.md`、`doc/tasks/*.md` 为准；若与课程原始说明存在细节差异，优先按详细设计实现，并在 README 或实验记录中说明实际口径。

## 2. 最终目标

实现一个可复现的 Python 工程，至少包含：

- 从 `A股数据/` 读取原始 CSV。
- 构建不含未来信息泄露的特征、标签和训练样本。
- 按时间划分训练期与验证 / 回测期。
- 实现至少一个神经网络模型，并实现强基线模型。
- 输出训练 / 验证损失曲线。
- 计算 IC、RankIC、ICIR、方向准确率和分组收益。
- 在 `2026-01-01` 至 `2026-05-20` 上完成历史回测。
- 回测考虑 T+1、手续费、滑点、涨跌停、停牌、最小交易单位和满仓倾向。
- 生成最新可用日期的下一交易日信号和订单建议。
- 创建 `README.md` 和 `requirements.txt`。
- 所有核心代码有完整 pytest 单元测试。
- 通过 `python -m pytest`、`python -m mypy src tests`、`python -m ruff check .`。

## 3. 不可违反的硬约束

- 不随机打乱时间序列日期进行训练 / 验证划分。
- 不在全量数据上拟合 scaler、imputer、winsorize 阈值、TF-IDF 词表或任何特征统计量。
- 对信号日 `T` 的预测只能使用 `T` 日收盘后及以前已知的数据；对应交易日为 `T+1`。
- 主标签必须为 `label_1d(T) = close(T+2) / close(T+1) - 1`。
- 辅助标签为未来 3 日和 5 日收益率。
- 官方股票池必须排除 ST 股票和北交所股票。
- 沪深 300 成分股回退只能使用不晚于目标日期的历史权重文件。
- 新闻第一版按市场级特征处理，TF-IDF 只在训练期 `fit`。
- 模型、策略、回测模块不得直接读取原始 CSV；必须通过数据、特征、数据集层传递结构化数据。
- 不提交或依赖原始数据、模型大权重、大缓存和临时输出。
- 不能为了测试通过而删除关键功能或降低验收要求。

## 4. 默认实现口径

如果文档中某些参数没有固定数值，不能停下来等待人工确认。采用可解释默认值，写入 `src/config/default.yaml`，并在 README 中说明：

- `commission_rate`: `0.0003`
- `stamp_tax_rate`: `0.001`
- `slippage_rate`: `0.0005`
- `min_lot_size`: `100`
- `initial_cash`: `1000000`
- `execution_price`: `open`
- `top_k`: `20`
- `max_single_weight`: `0.10`
- `max_industry_weight`: `0.30`
- `lookback`: `20`
- `batch_size`: `1024`
- `epochs`: `50`
- `learning_rate`: `0.001`
- `seed`: `42`

若实际数据字段与文档字段略有差异，先检查 `A股数据/` 的真实 CSV schema，再实现兼容映射；必须把兼容逻辑集中放在 `data.schema` 或 loader 层。

## 5. 建议代码结构

按以下结构实现：

```text
src/
  config/
    default.yaml
    loader.py
  data/
    schema.py
    loader.py
    calendar.py
    universe.py
    news_features.py
  features/
    price_features.py
    metric_features.py
    moneyflow_features.py
    preprocess.py
    feature_store.py
  datasets/
    split.py
    labels.py
    tabular_dataset.py
    window_dataset.py
  models/
    base.py
    factory.py
    linear.py
    gbdt.py
    mlp.py
    transformer.py
    sequence.py
  training/
    trainer.py
    losses.py
    callbacks.py
  evaluation/
    metrics.py
    reports.py
  backtest/
    portfolio.py
    execution.py
    strategy.py
    engine.py
    metrics.py
  predict/
    daily_signal.py
    daily_order.py
  utils/
    seed.py
    logging.py
    io.py
train.py
evaluate.py
README.md
requirements.txt
pyproject.toml
tests/
```

可以按工程需要微调文件，但必须保持模块边界清晰。

## 6. 主 Agent 工作方式

你是主 Agent，必须持续跟踪整体进度：

1. 先阅读所有文档和任务清单。
2. 检查 `A股数据/` 中少量真实文件的字段，建立 schema 映射。
3. 创建全局实现计划，按模块推进。
4. 为每个模块生成子 Agent，明确文件所有权、输入输出、测试要求和完成标准。
5. 子 Agent 完成后，主 Agent 负责 code review、接口整合、补测试、更新任务勾选。
6. 每完成一个模块，更新对应 `doc/tasks/{module}.md` 和 `doc/tasks/progress.md`。
7. 全部模块完成后，运行完整质量门禁，修复所有失败。
8. 最终输出完成摘要、测试结果和剩余风险。

无人类参与时，遇到非阻塞不确定性时应做保守假设并记录；只有发现文档之间存在无法同时满足的硬冲突时，才在最终报告中列为风险。

## 7. 子 Agent 拆分

主 Agent 应生成以下子 Agent。每个子 Agent 只负责自己模块的文件，不能随意改动其他模块；如需跨模块接口，先在主 Agent 处确认接口契约。

### 7.1 foundation 子 Agent

负责：

- 创建包结构和 `__init__.py`。
- 实现配置加载、配置校验、路径解析。
- 实现随机种子、日志、IO 工具。
- 创建默认配置、`pyproject.toml`、基础测试。

必须测试：

- 默认配置可加载。
- 相对路径能解析到项目根。
- 随机种子函数可执行。
- 输出目录创建函数可重复调用。

### 7.2 data 子 Agent

负责：

- `data.schema`
- `CsvDataLoader`
- `TradingCalendar`
- `UniverseBuilder`
- 新闻读取。

必须测试：

- 文件存在和缺失时 loader 都稳定返回。
- 交易日偏移正确。
- 沪深 300 成分回退只向历史查找。
- 官方股票池排除 ST 和北交所。
- 可交易股票池过滤停牌、零成交量、异常低成交额。

### 7.3 features 子 Agent

负责：

- 量价特征。
- 基本面特征。
- 资金流特征。
- 市场级新闻统计和 TF-IDF。
- 预处理器。
- 特征缓存。

必须测试：

- 滚动特征不使用 `T+1` 或未来价格。
- 验证期极端值不改变训练期截尾阈值。
- TF-IDF 词表只由训练期拟合。
- 无新闻日期输出全零文本向量和新闻数量 0。
- 保存 / 加载预处理器后 transform 结果一致。

### 7.4 datasets 子 Agent

负责：

- 标签构造。
- 时间切分。
- `TabularDataset`
- `WindowDataset`
- PyTorch DataLoader 和 sklearn 导出。

必须测试：

- `label_1d` 使用 `T+1` 与 `T+2`。
- 缺少未来交易日时不生成样本。
- 窗口 `[T-lookback+1, ..., T]` 不含未来日期。
- 训练和验证按交易日切分，不随机跨日期混合。
- 特征列选择排除标签列、未来价格列和非数值标识列。

### 7.5 models 子 Agent

负责：

- 统一 `BaseAlphaModel`。
- 模型工厂。
- Ridge / ElasticNet。
- LightGBM，缺失时降级为 `HistGradientBoostingRegressor`。
- MLP。
- Transformer Encoder。
- 保存 / 加载和模型摘要。

必须测试：

- MLP forward shape。
- Transformer forward shape。
- PyTorch 模型保存后加载，预测一致。
- 传统模型保存后加载，预测一致。
- 所有模型输出统一字段：`trade_date, ts_code, score, model_name`。

### 7.6 training 子 Agent

负责：

- 训练主流程。
- 训练期预处理拟合。
- sklearn 模型训练入口。
- PyTorch 训练循环。
- MSELoss、HuberLoss 预留、早停、日志和 artifact。
- `train.py` CLI。

必须测试：

- 小样本 MLP 训练 1 epoch 可输出日志和验证预测。
- 预处理只在训练期 fit。
- 重复 seed 下关键结果稳定到合理范围。
- 中断前已有日志保持可读。

### 7.7 evaluation 子 Agent

负责：

- MSE、MAE。
- 日度 IC、RankIC、ICIR。
- 方向准确率。
- 分组收益。
- 图表和评估 CLI。
- 基线对比汇总。

必须测试：

- 完全正相关 IC 为 1。
- 完全负相关 IC 为 -1。
- 单日样本过少时跳过该日。
- 输出 JSON / CSV 字段完整。

### 7.8 backtest 子 Agent

负责：

- `PortfolioState`、`Position`。
- 执行约束。
- TopK 等权策略。
- 分数加权 + 风险约束策略。
- 回测主循环。
- 回测指标。
- 基准净值对比。

必须测试：

- T+1：当日买入不可当日卖出。
- 买入股数按 100 股向下取整。
- 涨停买入被拒绝。
- 跌停卖出被拒绝。
- 单调上涨净值最大回撤为 0。
- 小回测现金、股数、净值不出现非法负值。

### 7.9 predict 子 Agent

负责：

- 最新可用交易日识别。
- 每日信号生成。
- 每日订单建议。
- `predict/daily_signal.py` CLI。
- `predict/daily_order.py` CLI。

必须测试：

- 删除 `T+1` 之后文件后，仍能为 `T` 生成信号。
- 信号 CSV 字段完整。
- 订单 CSV 字段完整。
- 每日预测只加载训练期保存的预处理器并调用 `transform`。

### 7.10 reproducibility 子 Agent

负责：

- `README.md`
- `requirements.txt`
- 实验记录模板。
- 报告素材导出说明。
- 不提交内容边界说明。

必须包含：

- 安装依赖命令。
- 训练、评估、回测、每日预测示例命令。
- 数据路径说明。
- 标签、时间切分、无泄露原则。
- 输出目录说明。
- 最小复现检查清单。

## 8. 测试策略

测试必须优先使用小型人工数据和 `tmp_path`，不能依赖完整真实数据才能通过。真实数据 smoke test 可以存在，但必须可跳过或在数据存在时运行。

测试目录建议：

```text
tests/
  test_config.py
  test_data_loader.py
  test_calendar.py
  test_universe.py
  test_price_features.py
  test_preprocess.py
  test_news_features.py
  test_labels.py
  test_datasets.py
  test_models.py
  test_training.py
  test_evaluation.py
  test_backtest_execution.py
  test_backtest_metrics.py
  test_predict.py
```

必须覆盖泄露风险测试：

- 修改未来价格不改变历史特征。
- 训练期预处理参数不被验证期改变。
- 窗口数据集不包含未来日期。
- 每日预测不需要未来文件。
- TF-IDF 词表不使用验证期文本。

## 9. 质量门禁

最终必须通过：

```bash
python -m ruff check .
python -m mypy src tests
python -m pytest
```

建议在实现过程中频繁运行：

```bash
python -m pytest tests/test_labels.py tests/test_preprocess.py
python -m pytest tests/test_models.py
python -m pytest tests/test_backtest_execution.py
```

如果依赖未安装，先更新 `requirements.txt`，再安装依赖。不要通过跳过核心测试来掩盖实现问题。

## 10. 实现细节要求

- 使用 `pathlib.Path` 管理路径。
- 所有公共函数和类方法写类型注解。
- pandas DataFrame 输入输出要有明确字段约定。
- 复杂日期语义使用变量名区分：`signal_date`、`trade_date`、`next_trade_date`、`label_end_date`。
- 配置由 YAML 驱动，CLI 参数只做覆盖。
- 日志写入 `outputs/runs/{run_id}/`。
- 图表使用 matplotlib 保存 PNG。
- 传统模型使用 scikit-learn/joblib 保存轻量 artifact。
- PyTorch 模型保存 state dict 和配置，不强制提交权重。
- LightGBM 不可用时必须自动降级，不应导致测试失败。
- 对异常输入给出明确错误信息，不静默产生错误结果。

## 11. 推荐执行顺序

严格按依赖顺序推进：

1. foundation
2. data
3. features
4. datasets
5. models
6. training
7. evaluation
8. backtest
9. predict
10. reproducibility

可并行的前提是接口已经稳定。例如 models 和 evaluation 可以在 datasets 接口确定后并行；backtest 可以基于人工 signal 和行情先实现，不必等待真实模型训练完成。

## 12. 完成定义

项目完成时必须满足：

- `doc/tasks/progress.md` 所有第一版模块已勾选。
- 每个 `doc/tasks/{module}.md` 的最小任务和完成标准已勾选。
- `README.md` 中的最小复现流程可执行。
- `requirements.txt` 能安装核心运行依赖。
- `python -m ruff check .` 通过。
- `python -m mypy src tests` 通过。
- `python -m pytest` 通过。
- 最新可用日期能生成 `outputs/signals/YYYYMMDD_signal.csv`。
- 给定持仓和现金后能生成 `outputs/orders/YYYYMMDD_orders.csv`。
- 回测能输出 `nav.csv`、`trades.csv`、`positions.csv`、`orders.csv`、`backtest_metrics.json`、`nav_curve.png`。

最后输出简短总结：

- 已实现模块。
- 主要命令和测试结果。
- 生成的关键文件。
- 任何仍需人工在报告阶段补充的信息，例如组员姓名、学号、模拟盘截图。
