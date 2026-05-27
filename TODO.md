# 改进清单

## 第一层：低投入、高回报

- [x] 1. 换 label_5d 重新训练
  - 改 `label.main=label_5d`，无需改代码 → 实际需要接通 `train.py` 中 `config["label"]["main"]` → `label_column` 链路
  - 预期 IC 从 0.02 提升到 0.04+

- [x] 2. 减小模型容量
  - MLP: `hidden_dim: 256 → 128`, `num_layers: 3 → 2`
  - Transformer: `hidden_dim: 256 → 128`, `num_layers: 3 → 2`, `num_heads: 8 → 4`

- [x] 3. 加 L2 正则化（weight decay）
  - 训练代码加 `weight_decay=1e-4`
  - 只对 MLP 和 Transformer 生效

## 第二层：中等投入、稳定收益

- [x] 4. 提高 dropout（0.15 → 0.3）
- [x] 5. 目标工程：波动率标准化（label / trailing_volatility_20d）
- [x] 6. 目标工程：cross-sectional rank normalize
- [ ] 7. GBDT 特征选择（选 top-32 特征喂神经网络）

## 第三层：值得探索但投入较大

- [ ] 8. 训练标签从回归改为分类（涨/跌/平）
- [ ] 9. 模型集成（GBDT + MLP stacking）
- [ ] 10. 添加更多 alpha 因子（行业中性化、基本面因子等）
- [ ] 11. 学习率调度（CosineAnnealing / ReduceLROnPlateau）

## 大作业硬性要求缺口

- [ ] 12. 回测引擎缺少 CLI 入口
  - `src/backtest/engine.py` 中 `BacktestEngine.run()` 已完整实现（NAV 跟踪、年化收益、夏普比率、最大回撤、NAV 曲线图）
  - 但项目中没有 `backtest.py` 或等效命令行入口，无法对历史区间运行完整回测模拟
  - 大作业要求 "推荐实现历史回测；不实现会影响分数"
  - README 提到了 `outputs/backtest/{run_id}/` 目录但没给出生成命令

- [ ] 13. 回测缺少与市场指数的对比
  - 大作业提到 "可以和上证指数，沪深300指数做对比"
  - `src/backtest/metrics.py:38-53` 已有 `benchmark_nav()` 函数，但未被任何地方调用
  - 回测流程没有加载指数基准数据并做对比

- [ ] 14. config 标志未生效（小问题）
- [ ] 15. 接入 FinBERT 新闻特征
  - NewsFinbertFeatureGenerator (FinBERT + TruncatedSVD 768→16)
  - 集成到 prepare_panels.py 和 daily_signal.py
  - `default.yaml` 中 `official_universe_exclude_st` 和 `official_universe_exclude_bse` 已定义
  - 但 `src/data/universe.py` 中硬编码了两种排除，未读取这些配置标志
  - 功能正确但不优雅
