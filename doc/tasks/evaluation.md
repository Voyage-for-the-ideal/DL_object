# evaluation 模块任务

## 模块目标

评估模型预测能力，输出 IC、ICIR、方向准确率、分组收益和可视化图表，为模型选择和报告分析提供依据。

## 最小任务清单

- [x] E001 实现基础回归指标：MSE、MAE。
- [x] E002 实现日度 IC，同一交易日内计算 `score` 与 `label` 的 Pearson 相关。
- [x] E003 实现日度 RankIC，同一交易日内计算 `score` 与 `label` 的 Spearman 相关。
- [x] E004 实现 IC 均值、IC 标准差和 ICIR，单日股票数量过少时跳过该日。
- [x] E005 实现方向准确率：`sign(score) == sign(label)`。
- [x] E006 实现按预测分数分 5 组或 10 组的分组收益统计。
- [x] E007 实现 `daily_ic.csv`、`group_return.csv`、`prediction_metrics.json` 输出。
- [x] E008 实现 `ic_curve.png` 和 `group_return_plot.png`。
- [x] E009 实现评估 CLI，读取 `valid_predictions.csv` 和标签后生成完整评估目录。
- [x] E010 为完全正相关样本写测试，IC 应为 1。
- [x] E011 为完全负相关样本写测试，IC 应为 -1。
- [x] E012 为单日样本过少写测试，指标计算应跳过而不是污染整体结果。
- [x] E013 实现基线对比汇总表，至少支持 Ridge、LightGBM、MLP、Transformer 的指标并列展示。

## 完成标准

- [x] 指标输入统一为 `trade_date, ts_code, score, label`。
- [x] 评估结果可以直接用于报告中的模型预测能力分析。
- [x] IC、ICIR 和方向准确率与需求文档口径一致。
- [x] 评估模块不重新训练模型，也不重新拟合预处理器。


