# models 模块任务

## 模块目标

实现统一预测模型接口，并提供强基线、深度学习基线和 Transformer 时序模型。所有模型输出统一连续分数，用于 IC、排序、策略和回测。

## 最小任务清单

- [x] M001 定义 `BaseAlphaModel` 接口，包含 `fit`、`predict`、`save`、`load`。
- [x] M002 统一预测输出格式：`trade_date, ts_code, score, model_name`。
- [x] M003 实现模型工厂，根据配置创建 Ridge、ElasticNet、LightGBM、MLP、Transformer。
- [x] M004 实现 `models.linear`，支持 Ridge 和 ElasticNet 回归基线。
- [x] M005 实现 `models.gbdt`，优先使用 LightGBM Regressor；环境缺失时提供 scikit-learn HistGradientBoostingRegressor 降级方案。
- [x] M006 实现 `models.mlp`，结构为 Linear、BatchNorm、ReLU、Dropout、Linear 输出连续分数。
- [x] M007 为 MLP 写 forward shape 测试，随机 batch 输出应为 `[batch]` 或 `[batch, 1]`。
- [x] M008 实现 `models.transformer` 的特征投影、位置编码、Transformer Encoder、池化和回归头。
- [x] M009 为 Transformer 写 shape 测试，输入 `[batch, lookback, feature_dim]` 输出连续分数。
- [x] M010 实现 PyTorch 模型保存和加载，加载后同一输入预测分数应一致。
- [x] M011 实现传统模型保存和加载，使用 joblib 或等价方式保存轻量 artifact。
- [x] M012 实现模型摘要输出，记录模型名、输入维度、参数量、关键超参数。
- [x] M013 实现 LightGBM 或树模型特征重要性导出，用于报告分析。
- [x] M014 预留 GRU、LSTM、TCN 接口文件或工厂枚举，但第一版不要求完整训练。

## 完成标准

- [x] 至少一个神经网络模型 MLP 可以训练和预测。
- [x] Transformer Encoder 可以完成一次 forward 和保存加载测试。
- [x] 所有模型的预测结果字段一致。
- [x] 模型模块不直接读取原始 CSV，不做股票池过滤和标签构造。


