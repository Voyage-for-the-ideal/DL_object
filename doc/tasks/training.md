# training 模块任务

## 模块目标

串联配置、数据集、预处理器、模型和评估指标，完成可复现训练流程，并输出训练日志、验证预测和损失曲线。

## 最小任务清单

- [x] TR001 实现 `training.trainer` 主流程：加载配置、构造数据集、拟合预处理器、构造模型、训练、评估、保存 artifact。
- [x] TR002 实现训练期预处理拟合，确保 scaler、imputer、winsorize 阈值只在训练期 `fit`。
- [x] TR003 实现 scikit-learn 模型训练入口，支持 Ridge、ElasticNet 和 GBDT。
- [x] TR004 实现 PyTorch 模型训练循环，支持 MLP 和 Transformer。
- [x] TR005 实现 MSELoss，预留 HuberLoss 和 RankIC loss 配置。
- [x] TR006 实现早停逻辑，基于验证集 loss 或验证 IC 保存最佳轻量 artifact。
- [x] TR007 每个 epoch 记录 train loss、valid loss、valid IC、valid RankIC、学习率。
- [x] TR008 保存 `outputs/runs/{run_id}/config.yaml`，记录实际运行配置。
- [x] TR009 保存 `train_log.csv`、`metrics.json`、`loss_curve.png`、`valid_predictions.csv`、`model_summary.txt`。
- [x] TR010 实现训练中断后的最小可读日志，不要求自动恢复训练，但已有日志不能损坏。
- [x] TR011 实现 `train.py` 命令行入口，支持指定配置文件、模型名、run_id。
- [x] TR012 实现小样本训练 smoke test：MLP 训练 1 个 epoch 并输出日志和验证预测。
- [x] TR013 实现重复 seed smoke test，同一配置下关键指标在可接受范围内稳定。

## 完成标准

- [x] 训练流程可以在小股票池上跑通。
- [x] 训练和验证按时间切分，不随机跨日期混合。
- [x] 训练输出足够支持报告中的损失曲线和模型对比。
- [x] 大体积模型权重只保存在本地输出目录，不作为提交内容。


