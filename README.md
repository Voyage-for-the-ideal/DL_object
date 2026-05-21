# A 股短期趋势预测与模拟交易

本项目实现课程大作业所需的可复现 Python 工程：读取 `A股数据/`，构造无未来信息泄露的特征和标签，训练基线模型与神经网络模型，评估 IC / RankIC / ICIR / 方向准确率，并把每日预测分数转成模拟盘订单建议。

## 环境

推荐使用已有的 conda 环境：

```bash
conda activate d2l
pip install -r requirements.txt
```

如使用 Codex / 脚本方式运行，可将命令写成：

```bash
conda run -n d2l python -m pytest
```

## 数据路径

默认数据目录为项目根目录下的 `A股数据/`，配置在 `src/config/default.yaml` 的 `data.root`。原始数据、训练权重、大缓存和 `outputs/` 下的大体积产物不应提交。

主要输入包括：

- `basic.csv`：股票基础信息，用于排除北交所。
- `trade_cal.csv`：交易日历。
- `daily/`、`metric/`、`moneyflow/`：日频行情、估值、资金流。
- `index_weight/`：沪深 300 等指数成分，回退时只使用目标日期之前的文件。
- `stock_st/`：按日 ST 股票列表。
- `news/`：市场级新闻文本，第一版以 TF-IDF 处理。

## 时间与标签口径

默认训练期为 `2019-01-01` 至 `2025-12-31`，验证 / 回测期为 `2026-01-01` 至 `2026-05-20`。

主标签严格使用：

```text
label_1d(T) = close(T+2) / close(T+1) - 1
```

其中 `T` 是信号日，模型只使用 `T` 日收盘后及以前可得数据；策略在 `T+1` 下单，主标签以 `T+2` 可卖出价格衡量收益。辅助标签支持 3 日和 5 日。

预处理器、截尾阈值、填充值、标准化参数和 TF-IDF 词表只在训练期 `fit`，验证、回测和每日预测只调用 `transform`。

## 示例命令

质量检查：

```bash
python -m ruff check .
python -m mypy src tests
python -m pytest
```

从已准备好的训练 / 验证面板训练 Ridge 或 MLP：

```bash
python train.py --train-panel outputs/cache/train_panel.csv --valid-panel outputs/cache/valid_panel.csv --model-name ridge --run-id demo_ridge
python train.py --train-panel outputs/cache/train_panel.csv --valid-panel outputs/cache/valid_panel.csv --model-name mlp --override training.epochs=1 --run-id demo_mlp
```

评估验证集预测：

```bash
python evaluate.py --predictions outputs/runs/demo_ridge/valid_predictions.csv --output-dir outputs/evaluation/demo_ridge --label-column label_1d
```

生成每日信号：

```bash
python -m src.predict.daily_signal --feature-file outputs/cache/latest_features.csv --preprocessor outputs/runs/demo_ridge/preprocessor.joblib --model outputs/runs/demo_ridge/model.joblib --model-name ridge
```

生成模拟盘订单：

```bash
python -m src.predict.daily_order --signal-file outputs/signals/20260520_signal.csv --quotes-file A股数据/daily/20260520.csv --cash 1000000
```

## 输出目录

- `outputs/runs/{run_id}/`：训练配置、日志、损失曲线、验证预测、模型摘要、轻量 artifact。
- `outputs/evaluation/{run_id}/`：`prediction_metrics.json`、`daily_ic.csv`、`group_return.csv`、图表。
- `outputs/backtest/{run_id}/`：`nav.csv`、`trades.csv`、`positions.csv`、`orders.csv`、`backtest_metrics.json`、`nav_curve.png`。
- `outputs/signals/`：每日信号 `YYYYMMDD_signal.csv`。
- `outputs/orders/`：每日订单建议 `YYYYMMDD_orders.csv`。
- `outputs/cache/`：特征缓存，可删除重建，不应提交。

## 最小复现检查清单

1. 安装依赖并确认 `python -m pytest` 通过。
2. 使用小股票池生成特征面板，并检查训练 / 验证按日期切分。
3. 训练 1 个 epoch 的 MLP 或一个 Ridge 基线。
4. 评估验证预测，输出 IC、RankIC、ICIR 和方向准确率。
5. 用预测分数运行短区间回测，检查 T+1、手续费、滑点、涨跌停和 100 股整数倍。
6. 使用最新可用日生成下一交易日信号和订单建议。

报告阶段仍需补充组员姓名、学号、具体分工，以及同花顺模拟盘截图和实际成交偏差记录。

