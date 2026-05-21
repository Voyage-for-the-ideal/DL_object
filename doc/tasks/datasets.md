# datasets 模块任务

## 模块目标

把特征面板转换为可训练样本，完成标签构造、时间切分、表格数据集和历史窗口数据集。该模块必须明确区分信号日、买入日和标签结束日。

## 最小任务清单

- [x] DS001 实现标签构造函数：`label_1d(T) = close(T+2) / close(T+1) - 1`。
- [x] DS002 实现辅助标签：`label_3d(T)` 和 `label_5d(T)`，并允许通过配置选择训练目标。
- [x] DS003 为标签构造写人工测试：4 至 7 个交易日价格序列能验证 `T+1` 和标签结束日对齐。
- [x] DS004 实现样本过滤：缺少 `T+1` 或标签结束日行情时，不生成对应 horizon 样本。
- [x] DS005 实现时间切分函数，默认训练期为 2019-01-01 至 2025-12-31，验证期为 2026-01-01 至 2026-05-20。
- [x] DS006 禁止随机把验证期日期混入训练集；如需 DataLoader shuffle，只允许在训练集内部按样本打乱。
- [x] DS007 实现 `TabularDataset`，输出 `X[num_samples, num_features]`、`y[num_samples]`、`index[trade_date, ts_code]`。
- [x] DS008 实现 `WindowDataset`，对信号日 `T` 输出 `[T-lookback+1, ..., T]` 的历史窗口。
- [x] DS009 实现窗口缺失处理策略，至少支持缺失历史剔除或填充加 mask 中的一种。
- [x] DS010 为窗口数据集写泄露测试：任意样本窗口不得包含 `T+1` 或之后日期。
- [x] DS011 实现特征列选择，排除标签列、未来价格列、非数值标识列。
- [x] DS012 实现 PyTorch DataLoader 构造函数，用于 MLP 和 Transformer 训练。
- [x] DS013 实现 scikit-learn 风格数据导出，用于 Ridge、ElasticNet 和 LightGBM。
- [x] DS014 实现小样本 dataset smoke test，输出样本数、特征数、日期范围和标签缺失率。

## 完成标准

- [x] 主标签与需求文档定义完全一致。
- [x] 表格模型和时序模型可以共享同一份特征与标签语义。
- [x] 时间切分由交易日决定，不使用随机样本划分。
- [x] 数据集模块不拟合 scaler、imputer 或 TF-IDF。


