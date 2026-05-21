# 总体进度

> 状态说明：以下勾选表示第一版代码路径、CLI、文档和小样本单元测试已经落地。
> 完整 2026-01-01 至 2026-05-20 历史实验、训练权重和模拟盘截图仍需在报告阶段用 `d2l`
> 环境生成，并按 README 的输出目录保存。

## 勾选规则

- 模块内所有最小任务完成，并通过该模块完成标准后，才能勾选本文件中的模块项。
- 若模块存在可选扩展任务，不影响第一版模块完成判定。
- 每次完成模块后，同步更新对应模块任务文件和本进度文件。

## 模块完成情况

- [x] foundation：项目骨架、配置、日志、随机种子、基础 IO，见 `doc/tasks/foundation.md`
- [x] data：原始 CSV 读取、交易日历、股票池、新闻读取，见 `doc/tasks/data.md`
- [x] features：量价、基本面、资金流、新闻、预处理、特征缓存，见 `doc/tasks/features.md`
- [x] datasets：标签、时间切分、TabularDataset、WindowDataset，见 `doc/tasks/datasets.md`
- [x] models：统一模型接口、线性基线、GBDT、MLP、Transformer，见 `doc/tasks/models.md`
- [x] training：训练流程、日志、曲线、验证预测、训练 artifact，见 `doc/tasks/training.md`
- [x] evaluation：IC、ICIR、方向准确率、分组收益、评估图表，见 `doc/tasks/evaluation.md`
- [x] backtest：回测引擎、执行约束、策略、回测指标、基准对比，见 `doc/tasks/backtest.md`
- [x] predict：每日信号、每日订单、最新日期预测、未来信息检查，见 `doc/tasks/predict.md`
- [x] reproducibility：README、requirements、复现实验说明、报告素材整理，见 `doc/tasks/reproducibility.md`

## 推荐执行顺序

- [x] 1. foundation
- [x] 2. data
- [x] 3. features
- [x] 4. datasets
- [x] 5. models
- [x] 6. training
- [x] 7. evaluation
- [x] 8. backtest
- [x] 9. predict
- [x] 10. reproducibility

## 第一版验收出口

- [x] 能读取 `A股数据/` 并完成小股票池数据处理。
- [x] 能生成无未来信息泄露的训练样本。
- [x] 能训练至少一个神经网络模型并输出损失曲线。
- [x] 能计算 IC、ICIR、方向准确率。
- [x] 能在 2026-01-01 至 2026-05-20 完成回测。
- [x] 回测考虑 T+1、手续费、滑点、涨跌停、停牌、最小交易单位。
- [x] 能生成最新日期的下一交易日信号和订单建议。
- [x] README 和 requirements 支持最小复现。


