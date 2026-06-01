# FT-Transformer n_blocks 选择分析

## 背景

FT-Transformer (Gorishniy et al. 2021) 将每个数值特征映射为 d_token 维 embedding，通过 Transformer blocks 建模特征间交互。当前配置为 `n_blocks=4`，训练中观察到严重过拟合（train_loss 单调下降，valid_loss 在 epoch 6-12 后持续上升），需要分析最优层数。

## 数据规模

- 训练样本：2454（时间序列切分，不可 shuffle）
- 特征维度：128（经 GBDT 特征选择后）或 273（全量）
- 128 特征下 GBDT 选择的 top-128 已能达到 Rank IC 0.172，优于 273 特征的 0.163

## 各 n_blocks 参数规模对比（128 特征, d_token=128）

单个 Transformer Block 参数量：

| 组件 | 参数 |
|------|------|
| MHA in_proj (QKV) | 128 × 128 × 3 = 49,152 |
| MHA out_proj | 128 × 128 = 16,384 |
| FFN fc1 (d_token → 2×d_token) | 128 × 256 = 32,768 |
| FFN fc2 (2×d_token → d_token) | 256 × 128 = 32,768 |
| LayerNorm × 2 | 2 × 256 = 512 |
| **单 Block 合计** | **~132K** |

总参数量（含 Feature Tokenizer：273 × 128 = 34,944 + CLS token 128 + positional encoding 129 × 128 = 16,512，约 52K）：

| n_blocks | Transformer 参数 | 总参数 | params:samples |
|----------|-----------------|--------|---------------|
| 1 | 132K | ~190K | 1:12.9 |
| **2** | **264K** | **~322K** | **1:7.6** |
| 3 | 396K | ~454K | 1:5.4 |
| 4 (当前) | 528K | ~588K | 1:4.2 |

## 经验法则

对于表格数据（tabular data）上的深度学习，推荐参数:样本比 ≥ 1:10 以防止过拟合。n_blocks=2 时 ~322K 参数对应 2454 样本，比值为 1:7.6，在可接受范围内（结合适当正则化如 dropout、weight decay）。n_blocks=4 时 1:4.2 的比值明显过拟合风险高。

## 各层数分析

### n_blocks=1（190K, 1:12.9）

- **优点**：参数:样本比最健康，过拟合风险最低
- **缺点**：仅能建模成对特征交互（pairwise interactions）；无法捕捉二阶交互（interaction of interactions），表达能力有限

### n_blocks=2（322K, 1:7.6）★ 推荐

- **优点**：参数:样本比合理；能建模 pairwise + second-order 特征交互；两层已足够覆盖表格数据的交互模式
- **推理**：表格数据不像文本/图像具有层次化结构（字符→词→句子→段落；边缘→纹理→部件→物体），每增加一层 transformer block 的边际收益递减。两层足以让每个特征在两次 self-attention 后与其他所有特征建立间接交互路径

### n_blocks=3（454K, 1:5.4）

- 参数:样本比开始偏危险，但配合强正则化（dropout ≥ 0.3, weight_decay ≥ 0.005）仍可控制

### n_blocks=4（588K, 1:4.2）— 当前配置

- **问题**：参数:样本比仅 1:4.2，明显不足。训练日志验证了这一点：
  - seed=42：最优 epoch 6/16，Rank IC 0.163（epoch 6 后 valid_loss 持续上升）
  - seed=123：最优 epoch 11/21，Rank IC 0.164
  - seed=456：最优 epoch 12/22，Rank IC 0.163
  - 三个 seed 均在 epoch 6-12 达到最优，之后过拟合

## 为什么表格数据不需要深层 Transformer

1. **无层次结构**：文本/图像的层次化语义需要多层抽象，表格特征的交互是"扁平"的——特征 A 与特征 B 的相关性不需要多层变换来表达
2. **特征数量有限**：128 个特征的成对交互空间为 C(128,2) ≈ 8000 种组合，两层 self-attention 已能充分覆盖
3. **FT-Transformer 原论文**（Gorishniy et al., 2021）：在大多数表格 benchmark 上，3-4 层已是最优配置的上限，更深的模型从未显著优于 2-3 层

## 结论

**推荐 n_blocks=2**，配合 dropout=0.35、drop_path=0.2、weight_decay=0.005、CosineAnnealingLR 调度器。这将参数:样本比从 1:4.2 改善至 1:7.6，同时保留足够的模型表达能力。
