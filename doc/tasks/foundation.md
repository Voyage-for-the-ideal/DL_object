# foundation 模块任务

## 模块目标

建立项目最小骨架、统一配置、日志、随机种子和基础 IO 约定。该模块完成后，后续数据、训练、回测和每日预测模块都能复用同一套路径、配置和输出规范。

## 最小任务清单

- [x] F001 创建 `src/` 包结构和必要的 `__init__.py`，包含 `config`、`data`、`features`、`datasets`、`models`、`training`、`evaluation`、`backtest`、`predict`、`utils`。
- [x] F002 创建 `src/config/default.yaml`，覆盖数据区间、股票池、标签、特征、预处理、数据集、模型、训练、回测、策略和输出目录配置。
- [x] F003 实现配置加载函数，支持读取 YAML、合并命令行覆盖项、将相对路径解析到项目根目录。
- [x] F004 实现配置字段校验，至少检查日期顺序、数据根目录、股票池模式、模型名、输出根目录是否有效。
- [x] F005 实现 `utils.seed`，统一设置 Python、NumPy、PyTorch 随机种子，并处理 CUDA 可复现选项。
- [x] F006 实现 `utils.logging`，支持控制台日志、文件日志、run_id 和关键配置摘要输出。
- [x] F007 实现 `utils.io`，提供安全创建目录、读写 JSON、读写 CSV、保存 YAML、生成 run 输出路径的函数。
- [x] F008 约定输出目录结构：`outputs/runs/`、`outputs/evaluation/`、`outputs/backtest/`、`outputs/signals/`、`outputs/orders/`、`outputs/cache/`。
- [x] F009 实现一个最小 smoke 脚本或测试，验证默认配置可以加载、输出目录可以创建、随机种子函数可以执行。
- [x] F010 在模块文档或代码注释中明确：`outputs/` 下的大体积缓存、模型权重和运行产物不作为提交内容。

## 完成标准

- [x] 运行配置加载 smoke test 不报错。
- [x] 后续模块可以通过统一配置读取数据根目录、日期范围、模型参数和输出路径。
- [x] 项目根目录与工作目录变化时，配置路径仍能稳定解析。


