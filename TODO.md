# 改进清单

## 第一层：低投入、高回报

- [ ] 1. 换 label_5d 重新训练
  - 改 `label.main=label_5d`，无需改代码
  - 预期 IC 从 0.02 提升到 0.04+

- [ ] 2. 减小模型容量
  - MLP: `hidden_dim: 256 → 128`, `num_layers: 3 → 2`
  - Transformer: `hidden_dim: 256 → 128`, `num_layers: 3 → 2`, `num_heads: 8 → 4`

- [ ] 3. 加 L2 正则化（weight decay）
  - 训练代码加 `weight_decay=1e-4`
  - 只对 MLP 和 Transformer 生效

## 第二层：中等投入、稳定收益

- [ ] 4. 提高 dropout（0.15 → 0.3）
- [ ] 5. 目标工程：波动率标准化（label / trailing_volatility_20d）
- [ ] 6. 目标工程：cross-sectional rank normalize
- [ ] 7. GBDT 特征选择（选 top-32 特征喂神经网络）

## 第三层：值得探索但投入较大

- [ ] 8. 训练标签从回归改为分类（涨/跌/平）
- [ ] 9. 模型集成（GBDT + MLP stacking）
- [ ] 10. 添加更多 alpha 因子（行业中性化、基本面因子等）
- [ ] 11. 学习率调度（CosineAnnealing / ReduceLROnPlateau）
