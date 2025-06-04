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

示例训练脚本 `experiments/train.py` 提供 `--model` 参数，可直接选择 `DeepFM`、`FFM`、`WideDeep` 或 `DCN` 进行实验，并支持指定学习率、L2 正则化和 Dropout。

## 数据预处理
- **连续特征缺失**: 统一填充为 0, 并增加二元指示特征。
- **类别特征缺失**: 填充为特殊字符串 `"unknown"`。
- **特征编码**:
  - 连续特征进行 Z-score 标准化。
  - 类别特征采用维度 8 的 embedding。
  - 对超长尾类别, 出现次数 < 100 的类别合并为 `"rare"`。
- **数据划分**: 随机打乱后按 70/15/15 划分为训练集、验证集和测试集。

## 实验实现
- 框架: Python + PyTorch + DeepCTR
- 训练参数:
  - `batch_size = 1024`
  - `embedding_dim = 8`
  - MLP 3 层, 隐藏单元 [256, 128, 64], 激活函数 ReLU
- 超参数范围:
  - 学习率 `lr ∈ {1e-3, 5e-4, 1e-4}`
  - L2 正则化系数 `λ ∈ {1e-3, 1e-4, 1e-5}`
  - Dropout `p ∈ {0.0, 0.2, 0.5}`
- 通过验证集使用网格搜索或贝叶斯优化进行超参数调优。

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
├─ models/         # 额外自定义模型(可选)
├─ experiments/    # 训练脚本
├─ logs/           # 训练日志
├─ outputs/        # 模型权重与评估结果
```

### 快速开始
1. 在 `data/` 目录放入原始 `criteo.csv` 文件。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行示例训练脚本（以 DeepFM 为例）：
   ```bash
   python experiments/train.py --data data/criteo.csv --epochs 1 \
       --model DeepFM --lr 1e-3 --l2 1e-5 --dropout 0.5 \
       --output outputs/result.csv
   ```
