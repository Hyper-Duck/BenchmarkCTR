# BenchmarkCTR

Baseline CTR Model Comparison

The dataset used is the **Criteo Uplift Modeling Dataset**, containing about 25 million rows. Each row includes 11 features, one treatment indicator, and two labels (visit and conversion).

## Experiment Objective

On the same dataset (Criteo Uplift Modeling Dataset), using a unified preprocessing and evaluation pipeline, we compare the click-through rate (CTR) prediction performance of the following eight models:

1. FTRL (Follow-The-Regularized-Leader)
2. FFM (Field-aware Factorization Machine)
3. Wide & Deep
4. DeepFM
5. Deep & Cross Network (DCN)
6. DMR (Deep Match to Rank)
7. DIN (Deep Interest Network)
8. CTNet (Continual Transfer Network)

The training script `experiments/train.py` provides the `--model` parameter. The script supports specifying learning rate, L2 regularization, and Dropout. Random seeds can be set using `--seed` for reproducibility. During training, models are saved per epoch in the directory specified by `--checkpoint-dir`, and validation metrics per epoch are written into the CSV file specified by `--log-file`.

## Data Preprocessing

* **Missing continuous features**: filled with 0, with an additional binary indicator feature.
* **Missing categorical features**: filled with a special string `"unknown"`.
* **Feature encoding**:

  * Continuous features: Z-score normalization.
  * Categorical features: embeddings with dimension 8.
  * Rare categories: categories appearing fewer than 100 times are merged into `"rare"`.
* **Data split**: after shuffling, the dataset is divided into 70/15/15 for training, validation, and test.

## Experiment Implementation

* Framework: Python + PyTorch
* Training parameters:

  * `batch_size = 1024`
  * `embedding_dim = 8`
  * MLP: 3 layers, hidden units \[256, 128, 64], activation = ReLU
* Hyperparameter search space:

  * Learning rate `lr ∈ {1e-3, 5e-4, 1e-4}`
  * L2 regularization `λ ∈ {1e-3, 1e-4, 1e-5}`
  * Dropout `p ∈ {0.0, 0.2, 0.5}`
* Hyperparameter tuning is performed via grid search on the validation set.

## Evaluation Metrics

* Primary: AUC, LogLoss, PR-AUC
* Secondary: Calibration (Brier score), training time, inference time

## Results Presentation

* Report AUC, LogLoss, PR-AUC on the test set, along with hyperparameter settings and model complexity.
* Compare training and inference times across models.
* Plot ROC curves, PR curves, and calibration curves.

## Project Structure

```
BenchmarkCTR/
├─ data/           # Raw dataset
├─ preprocess/     # Data preprocessing modules
├─ models/         # Custom models
├─ experiments/    # Training scripts
├─ logs/           # Training logs
├─ outputs/        # Model checkpoints and evaluation results
```

### Quick Start

1. Place the raw `criteo.csv` file in the `data/` directory.
2. On first run, the script will convert `criteo.csv` into `criteo.pt` in the same directory for faster loading.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run an example training script (DeepFM as example):

   ```bash
   python experiments/train.py --data data/criteo.csv --epochs 1 --model DeepFM --lr 1e-3 --l2 1e-5 --dropout 0.5 --output outputs/result.csv --seed 2025 --checkpoint-dir outputs/checkpoints --log-file logs/train_metrics.csv
   ```

   After training, all input hyperparameters (e.g., `seed`, `dnn_hidden_units`) will be appended with evaluation metrics into the CSV specified by `--output` for easy comparison.
5. To continue training from an existing checkpoint, provide `--start-from-checkpoint` and set `--epochs` to the additional number of epochs. Example (continue DeepFM from checkpoint epoch 2):

   ```bash
   python experiments/train.py --data data/criteo.csv --epochs 1 --model DeepFM --lr 1e-3 --l2 1e-5 --dropout 0.5 --output outputs/result.csv --seed 2025 --checkpoint-dir outputs/checkpoints --log-file logs/train_metrics.csv --start-from-checkpoint outputs/checkpoints/DeepFM_epoch_2.pt
   ```
6. The FTRL model uses four parameters: `alpha`, `beta`, `l1`, and `l2`, which differ slightly from the others. Example training script:

   ```bash
   python experiments/train.py --data data/criteo.csv --epochs 1 --model FTRL --alpha 0.05 --beta 1.0 --l1 1.0 --l2 1e-5 --output outputs/result.csv --seed 2025 --checkpoint-dir outputs/checkpoints --log-file logs/ftrl_log.csv
   ```
7. Example scripts for hyperparameter optimization can be found in `run.experiments.txt`.