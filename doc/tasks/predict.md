# predict 模块任务

## 模块目标

在每日盘后读取最新可用数据，生成下一交易日可使用的预测信号和模拟盘订单建议。该模块必须验证不读取未来日期文件。

## 最小任务清单

- [x] P001 实现最新可用交易日识别，从 `daily/` 文件和交易日历中确定信号日 `T`。
- [x] P002 实现运行前检查：最新 `daily` 存在、模型可加载、预处理器可加载、特征字段与训练期一致。
- [x] P003 实现每日特征构造，最多读取 `T` 及以前数据，不需要 `T+1` 文件。
- [x] P004 实现每日股票池过滤，排除 ST、北交所、停牌和不可交易股票。
- [x] P005 加载训练期保存的预处理器，只调用 `transform`。
- [x] P006 加载模型并输出连续预测分数。
- [x] P007 实现信号排名，输出字段：`signal_date, next_trade_date, ts_code, score, rank, model_name`。
- [x] P008 保存 `outputs/signals/YYYYMMDD_signal.csv`。
- [x] P009 实现 `daily_order` 输入读取，至少支持当前持仓 CSV 和当前现金配置。
- [x] P010 调用主策略生成模拟盘订单建议。
- [x] P011 输出字段：`signal_date, trade_date, ts_code, action, target_weight, target_shares, estimated_price, estimated_amount, reason`。
- [x] P012 保存 `outputs/orders/YYYYMMDD_orders.csv`。
- [x] P013 实现 `predict/daily_signal.py` 命令行入口，支持指定信号日或自动最新日期。
- [x] P014 实现 `predict/daily_order.py` 命令行入口，支持指定信号文件、持仓文件和现金。
- [x] P015 写泄露检查测试：删除 `T+1` 之后文件后，仍能为 `T` 生成信号。
- [x] P016 写字段完整性测试，信号和订单 CSV 必须包含设计文档要求字段。

## 完成标准

- [x] 最新日期可以生成下一交易日信号。
- [x] 每日预测不需要未来行情文件。
- [x] 订单建议考虑当前持仓、现金和交易约束。
- [x] 输出文件路径和字段满足需求文档。


