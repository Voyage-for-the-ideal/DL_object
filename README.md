# A 股短期趋势预测与模拟交易

本项目实现课程大作业所需的可复现 Python 工程：读取 `A股数据/`，构造无未来信息泄露的特征和标签，训练基线模型与神经网络模型，评估 IC / RankIC / ICIR / 方向准确率，并把每日预测分数转成模拟盘订单建议。

## 先理解两个关键词

**训练 / 验证面板是什么？**

面板文件就是模型训练用的二维表格 CSV。每一行是一只股票在某个信号日 `T` 的一个样本，例如：

- `trade_date`：信号日 `T`。
- `ts_code`：股票代码。
- 特征列：只使用 `T` 收盘后以及更早可获得的信息，例如价格、估值、资金流、滚动统计。
- 标签列：未来收益，例如 `label_1d(T) = close(T+2) / close(T+1) - 1`。

训练面板 `train_panel.csv` 和验证面板 `valid_panel.csv` 的区别只是日期范围不同。项目按时间切分，不随机打乱时间序列样本，避免把未来信息混进训练。

**baseline 是什么？**

baseline 是基准模型，用来证明复杂模型是否真的有提升。本项目推荐先跑 `ridge`，也就是岭回归。它不是深度学习模型，但训练快、稳定，适合检查数据和评估流程是否跑通。课程要求至少使用一个神经网络模型，所以正式结果还需要跑 `mlp` 或 `transformer_encoder`。

`mlp` 直接使用二维面板特征；`transformer_encoder` 会按股票和日期把面板重组为窗口序列，输入形状为 `[样本数, dataset.lookback, 特征数]`。默认 `dataset.lookback=20`，即每个样本使用同一只股票最近 20 个交易日的特征序列。

## 环境准备

```powershell
cd D:\code\DL_foundation\DL_object
pip install -r requirements.txt
```

先做一次质量检查：

```powershell
python -m ruff check .
python -m mypy src tests
python -m pytest
```

## 一键跑通完整流程

下面命令从原始数据开始，依次生成面板、训练模型、评估验证集、生成最新信号、生成模拟盘订单。

### 1. 生成训练 / 验证面板

默认配置读取 `src/config/default.yaml`。

```powershell
python prepare_panels.py
```

输出：

```text
outputs/cache/train_panel.csv
outputs/cache/valid_panel.csv
```

### 2. 先训练 baseline

```powershell
python train.py `
  --train-panel outputs/cache/train_panel.csv `
  --valid-panel outputs/cache/valid_panel.csv `
  --model-name ridge `
  --run-id demo_ridge
```

输出目录：

```text
outputs/runs/demo_ridge/
```

其中会包含验证集预测、指标、模型文件、预处理器和损失曲线。

### 3. 再训练神经网络模型

先用 1 个 epoch 快速确认流程：

```powershell
python train.py `
  --train-panel outputs/cache/train_panel.csv `
  --valid-panel outputs/cache/valid_panel.csv `
  --model-name mlp `
  --override training.epochs=1 `
  --run-id demo_mlp
```

正式实验可以增加 epoch：

```powershell
python train.py `
  --train-panel outputs/cache/train_panel.csv `
  --valid-panel outputs/cache/valid_panel.csv `
  --model-name mlp `
  --override training.epochs=50 `
  --run-id mlp_full
```

若使用 Qlib 风格的窗口 Transformer，命令仍然读取同一份面板，训练流程会自动构造历史窗口：

```powershell
python train.py `
  --train-panel outputs/cache/train_panel.csv `
  --valid-panel outputs/cache/valid_panel.csv `
  --model-name transformer_encoder `
  --override training.epochs=20 `
  --run-id verify_transformer_20ep
```

### 4. 评估验证集预测

```powershell
python evaluate.py `
  --predictions outputs/runs/demo_mlp/valid_predictions.csv `
  --output-dir outputs/evaluation/demo_mlp `
  --label-column label_1d
```

输出包括：

```text
outputs/evaluation/demo_mlp/prediction_metrics.json
outputs/evaluation/demo_mlp/daily_ic.csv
outputs/evaluation/demo_mlp/group_return.csv
```

### 5. 生成最新每日信号

数据最新交易日会自动从 `A股数据/daily/` 中识别。若使用 MLP：

```powershell
python -m src.predict.daily_signal `
  --preprocessor outputs/runs/demo_mlp/preprocessor.joblib `
  --model outputs/runs/demo_mlp/model.pt `
  --model-name mlp
```

若使用 Ridge baseline：

```powershell
python -m src.predict.daily_signal `
  --preprocessor outputs/runs/demo_ridge/preprocessor.joblib `
  --model outputs/runs/demo_ridge/model.joblib `
  --model-name ridge
```

若使用 `transformer_encoder`，程序会自动读取最近 `dataset.lookback` 个交易日并构造预测窗口：

```powershell
python -m src.predict.daily_signal `
  --preprocessor outputs/runs/verify_transformer_20ep/preprocessor.joblib `
  --model outputs/runs/verify_transformer_20ep/model.pt `
  --model-name transformer_encoder
```

手动传入 `--feature-file` 时，Transformer 不能只给单日特征，文件中至少要包含 `dataset.lookback` 个交易日的历史行。

输出示例：

```text
outputs/signals/20260520_signal.csv
```

### 6. 生成模拟盘订单建议

把上一步生成的信号文件和同一天行情文件传入。下面以 `20260520` 为例：

```powershell
python -m src.predict.daily_order `
  --signal-file outputs/signals/20260520_signal.csv `
  --quotes-file A股数据/daily/20260520.csv `
  --cash 1000000
```

输出：

```text
outputs/orders/20260520_orders.csv
```

### 7. 可选：批量训练多模型

如果使用 Git Bash、WSL 或 Linux 环境，可以直接跑批量脚本。它会按需生成面板，并依次训练 `ridge`、`elasticnet`、`gbdt`、`mlp` 和 `transformer_encoder`：

```bash
bash scripts/train_all_models.sh
```

常用环境变量：

```bash
REBUILD_PANELS=1 MODELS="ridge mlp transformer_encoder" bash scripts/train_all_models.sh
PANEL_DIR=outputs/cache/full_20160104_20260520 RUN_PREFIX=exp01 bash scripts/train_all_models.sh
```

脚本日志写入 `outputs/logs/`，训练产物仍写入 `outputs/runs/{run_id}/`。

## 配置说明

默认配置在 `src/config/default.yaml`。常用字段：

- `data.root`：原始数据目录，默认 `A股数据`。
- `data.start_date` / `data.train_end_date`：训练时间段。
- `data.valid_start_date` / `data.valid_end_date`：验证时间段。
- `data.universe_mode`：股票池，默认 `official`，即排除北交所和 ST 股票后的官方比赛股票池。
- `features.lookback_windows`：滚动特征窗口。
- `dataset.lookback`：Transformer 等窗口序列模型使用的历史交易日长度。
- `label.horizons`：标签收益周期，例如 1 日、3 日、5 日。
- `model.name`：默认模型名。
- `training.epochs`：训练轮数。
- `outputs.root`：输出目录，默认 `outputs`。

命令行可以用 `--override key=value` 临时覆盖配置，例如：

```powershell
python train.py `
  --train-panel outputs/cache/train_panel.csv `
  --valid-panel outputs/cache/valid_panel.csv `
  --model-name mlp `
  --override training.epochs=10 `
  --override training.learning_rate=0.0005 `
  --run-id mlp_lr_test
```

## 主要文件和目录用途

| 路径 | 用途 |
| --- | --- |
| `大作业.md` | 课程原始要求。 |
| `CLAUDE.md` | 给代码助手看的项目约束和注意事项。 |
| `README.md` | 项目说明和运行流程。 |
| `requirements.txt` | Python 依赖。 |
| `pyproject.toml` | Ruff、mypy、pytest 等工具配置。 |
| `prepare_panels.py` | 从 `A股数据/` 生成 `train_panel.csv` 和 `valid_panel.csv`。 |
| `train.py` | 从已生成的面板训练模型，并保存模型、预处理器、预测和训练曲线。 |
| `evaluate.py` | 读取验证集预测文件，输出 IC、RankIC、ICIR、分组收益等评估结果。 |
| `scripts/train_all_models.sh` | 批量生成面板并训练多个模型，适合正式实验。 |
| `src/config/default.yaml` | 默认数据、特征、模型、训练、回测配置。 |
| `src/data/` | CSV 数据读取、交易日历、股票池过滤。 |
| `src/features/` | 价格、估值、资金流、新闻等特征构造和预处理。 |
| `src/datasets/` | 标签构造、时间切分、表格 / 窗口数据集。 |
| `src/models/` | Ridge、ElasticNet、GBDT、MLP、Transformer 等模型封装。 |
| `src/training/` | 训练流程、早停、损失函数和训练日志。 |
| `src/evaluation/` | 预测评价指标和报告输出。 |
| `src/backtest/` | 投组合、交易执行、策略和回测指标。 |
| `src/predict/daily_signal.py` | 使用训练好的模型生成每日股票打分信号。 |
| `src/predict/daily_order.py` | 把每日信号转换为模拟盘订单建议。 |
| `tests/` | 单元测试和流程级冒烟测试。 |
| `doc/` | 课程设计文档（proposal、detailed-design）。 |
| `A股数据/` | 原始课程数据，不应提交到代码仓库。 |
| `outputs/` | 训练、评估、信号、订单和缓存输出，不应提交到代码仓库。 |

## 数据目录

默认数据目录为项目根目录下的 `A股数据/`：

- `basic.csv`：股票基础信息，用于排除北交所。
- `trade_cal.csv`：交易日历。
- `daily/`：日频 OHLCV 行情。
- `metric/`：估值和财务指标。
- `moneyflow/`：资金流数据。
- `market/`：指数行情。
- `index_weight/`：指数成分股权重。
- `stock_st/`：ST 股票列表。
- `news/`：新闻文本。

## 时间与标签口径

主标签严格使用：

```text
label_1d(T) = close(T+2) / close(T+1) - 1
```

其中 `T` 是信号日，模型只使用 `T` 日收盘后及以前可得数据；策略在 `T+1` 下单，主标签以 `T+2` 可卖出价格衡量收益。辅助标签支持 3 日和 5 日。

预处理器、截尾阈值、填充值、标准化参数和 TF-IDF 词表只在训练期 `fit`，验证、回测和每日预测只调用 `transform`。

## 输出目录

- `outputs/cache/`：特征和面板缓存，可删除重建。
- `outputs/logs/`：长流程脚本日志，可删除重建。
- `outputs/runs/{run_id}/`：训练配置、日志、损失曲线、验证预测、模型摘要、模型文件。
- `outputs/evaluation/{run_id}/`：`prediction_metrics.json`、`daily_ic.csv`、`group_return.csv`。
- `outputs/backtest/{run_id}/`：`nav.csv`、`trades.csv`、`positions.csv`、`orders.csv`、`backtest_metrics.json`、`nav_curve.png`。
- `outputs/signals/`：每日信号 `YYYYMMDD_signal.csv`。
- `outputs/orders/`：每日订单建议 `YYYYMMDD_orders.csv`。

## 最小复现检查清单

1. 安装依赖并确认质量检查通过。
2. 运行 `python prepare_panels.py` 生成训练 / 验证面板。
3. 训练 `ridge` baseline，确认数据和评估流程正常。
4. 训练 1 个 epoch 的 `mlp` 或 `transformer_encoder`，满足深度学习模型要求并确认模型链路正常。
5. 运行 `evaluate.py` 输出 IC、RankIC、ICIR 和方向准确率。
6. 使用最新可用日生成下一交易日信号和订单建议。

报告阶段仍需补充组员姓名、学号、具体分工，以及同花顺模拟盘截图和实际成交偏差记录。
