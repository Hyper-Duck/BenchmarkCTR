# BenchmarkCTR

Baseline CTR Model Comparison

数据集为 Criteo Uplift Modeling Dataset, 大约 2500 万行, 每行包含 11 个特征、一个处理指标和两个标签（访问与转化）。

## 实验目标
在相同数据集 Criteo Uplift Modeling Dataset 上, 采用统一的预处理与评估流程, 对比以下八种模型的点击率预测性能:

1. FTRL (Follow-The-Regularized-Leader)
2. FFM (Field-aware Factorization Machine)
3. Wide & Deep
4. DeepFM
5. Deep & Cross Network (DCN)
6. DMR (Deep Match to Rank)
7. DIN (Deep Interest Network)
8. CTNet (Continual Transfer Network)

训练脚本 `experiments/train.py` 提供 `--model` 参数。脚本支持指定学习率、L2 正则化和 Dropout，并可通过 `--seed` 设置随机种子以复现结果，训练过程中会在 `--checkpoint-dir` 指定目录下按 epoch 保存模型，并将每个 epoch 的验证指标写入 `--log-file` 指定的 CSV。

## 数据预处理
- **连续特征缺失**: 统一填充为 0, 并增加二元指示特征。
- **类别特征缺失**: 填充为特殊字符串 `"unknown"`。
- **特征编码**:
  - 连续特征进行 Z-score 标准化。
  - 类别特征采用维度 8 的 embedding。
  - 对超长尾类别, 出现次数 < 100 的类别合并为 `"rare"`。
- **数据划分**: 随机打乱后按 70/15/15 划分为训练集、验证集和测试集。

## 实验实现
- 框架: Python + PyTorch
- 训练参数:
  - `batch_size = 1024`
  - `embedding_dim = 8`
  - MLP 3 层, 隐藏单元 [256, 128, 64], 激活函数 ReLU
- 超参数范围:
  - 学习率 `lr ∈ {1e-3, 5e-4, 1e-4}`
  - L2 正则化系数 `λ ∈ {1e-3, 1e-4, 1e-5}`
  - Dropout `p ∈ {0.0, 0.2, 0.5}`
- 通过验证集使用网格搜索进行超参数调优。

## 评估指标
- 主要指标: AUC, LogLoss, PR-AUC
- 次要指标: Calibration (Brier score), 训练时间、推理时间

## 结果呈现
- 在测试集上统计 AUC、LogLoss、PR-AUC, 并列出超参数设定与模型复杂度表。
- 对比各模型的训练与推理时间。
- 绘制 ROC 曲线、PR 曲线及 Calibration 曲线。


## 项目结构
```
BenchmarkCTR/
├─ data/           # 原始数据存放位置
├─ preprocess/     # 数据预处理模块
├─ models/         # 额外自定义模型
├─ experiments/    # 训练脚本
├─ logs/           # 训练日志
├─ outputs/        # 模型权重与评估结果
```

### 快速开始
1. 在 `data/` 目录放入原始 `criteo.csv` 文件。
2. 首次运行时脚本会将 `criteo.csv` 转换为同目录下的 `criteo.pt` 以加速后续加载。
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 运行示例训练脚本（以 DeepFM 为例）：
   ```bash
   python experiments/train.py --data data/criteo.csv --epochs 1 --model DeepFM --lr 1e-3 --l2 1e-5 --dropout 0.5 --output outputs/result.csv --seed 2025 --checkpoint-dir outputs/checkpoints --log-file logs/train_metrics.csv
   ```
   训练结束后，所有传入的超参数（如 `seed`、`dnn_hidden_units` 等）会与评估指标一起
   追加写入 `--output` 指定的 CSV 文件，便于后续比较。
5. 若需从已有模型继续训练，可传入 `--start-from-checkpoint` 并将 `--epochs`
   设为额外训练的轮数（以 DeepFM 从 checkpointe poch_2 开始为例）：
   ```bash
   python experiments/train.py --data data/criteo.csv --epochs 1 --model DeepFM --lr 1e-3 --l2 1e-5 --dropout 0.5 --output outputs/result.csv --seed 2025 --checkpoint-dir outputs/checkpoints --log-file logs/train_metrics.csv --start-from-checkpoint outputs/checkpoints/DeepFM_epoch_2.pt
   ```
6. FTLR 模型与其他模型略有不同，其使用 `alpha`、`beta`、`l1` 和 `l2` 4个参数。示例训练脚本：
   ```bash
   python experiments/train.py --data data/criteo.csv --epochs 1 --model FTRL --alpha 0.05 --beta 1.0 --l1 1.0 --l2 1e-5 --output outputs/result.csv --seed 2025 --checkpoint-dir outputs/checkpoints --log-file logs/ftrl_log.csv
   ```
7. 实验的超参数优化运行代码示例可以在 `run.experiments.txt` 中找到。
