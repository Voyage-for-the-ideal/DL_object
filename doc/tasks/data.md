# data 模块任务

## 模块目标

从 `A股数据/` 读取原始 CSV，提供交易日历、股票池过滤、新闻读取和按日数据访问能力。该模块只负责加载、对齐和过滤，不做模型特征计算。

## 最小任务清单

- [x] D001 定义 `data.schema`，集中维护常用字段名：`trade_date`、`ts_code`、价格字段、成交量字段、新闻字段、指数权重字段。
- [x] D002 实现 `CsvDataLoader.load_basic()`，读取 `basic.csv`，统一股票代码、市场、上市日期字段格式。
- [x] D003 实现 `CsvDataLoader.load_trade_calendar()`，读取 `trade_cal.csv` 并只保留有效交易日。
- [x] D004 实现按日期读取 `daily/`、`metric/`、`moneyflow/`、`stock_st/` 的方法；缺失文件返回空 DataFrame 并记录日志。
- [x] D005 实现 `load_index_weight(trade_date=None)`，支持读取沪深 300 成分和权重数据。
- [x] D006 实现 `load_market(index_code)`，读取上证指数、沪深 300、创业板等基准行情。
- [x] D007 实现 `load_news(date)`，读取新闻标题、正文和时间字段，缺失日期返回空 DataFrame。
- [x] D008 实现日期标准化工具，保证内部交易日格式在字符串 `YYYYMMDD` 和 `datetime64` 之间可控转换。
- [x] D009 实现 `TradingCalendar.is_trading_day()`、`previous_trade_date()`、`next_trade_date()`、`trade_dates_between()`。
- [x] D010 为交易日历写人工测试，覆盖周末、节假日、区间首尾和 `T+n` 偏移。
- [x] D011 实现 `UniverseBuilder.get_hs300_universe(trade_date)`，当日没有权重文件时只向过去查找最近一期成分股。
- [x] D012 实现 `UniverseBuilder.get_official_universe(trade_date)`，从 `basic.csv` 排除北交所股票。
- [x] D013 实现按日 ST 过滤，确保只使用当日或历史已知 ST 状态，不用未来状态过滤历史样本。
- [x] D014 实现可交易股票池过滤，排除停牌、成交量为 0、成交额异常低或当日无行情的股票。
- [x] D015 实现一个小样本数据读取 smoke test：给定一个交易日，能读取行情、指标、资金流、股票池并输出行数统计。

## 完成标准

- [x] 数据读取方法在文件存在和缺失两种情况下都有稳定返回。
- [x] 沪深 300 成分回退只向历史查找。
- [x] 官方股票池始终排除 ST 股票和北交所股票。
- [x] 数据模块不计算未来收益标签，也不拟合任何预处理统计量。


