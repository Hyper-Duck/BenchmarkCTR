"""Training script skeleton for CTR models using DeepCTR-torch."""
import argparse
import pandas as pd
import torch
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.utils import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

from preprocess.utils import preprocess_criteo


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.data)
    df = df.sample(frac=1.0, random_state=2025).reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]

    numeric_cols = [c for c in df.columns if c.startswith("I")]
    categorical_cols = [c for c in df.columns if c.startswith("C")]

    df_train, scaler = preprocess_criteo(df_train, numeric_cols, categorical_cols)
    df_val, _ = preprocess_criteo(df_val, numeric_cols, categorical_cols)
    df_test, _ = preprocess_criteo(df_test, numeric_cols, categorical_cols)

    # simple vocab size assumption
    vocab_size = {col: df_train[col].nunique() + 1 for col in categorical_cols}

    feature_columns = [DenseFeat(col, 1) for col in numeric_cols]
    feature_columns += [SparseFeat(col, vocabulary_size=vocab_size[col], embedding_dim=8) for col in categorical_cols]

    feature_names = get_feature_names(feature_columns)

    train_dataset = Dataset(df_train, feature_columns, label_name="click")
    val_dataset = Dataset(df_val, feature_columns, label_name="click")

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(linear_feature_columns=feature_columns,
                   dnn_feature_columns=feature_columns,
                   task="binary",
                   dnn_hidden_units=[256, 128, 64],
                   l2_reg_embedding=1e-5,
                   dnn_dropout=0.5,
                   device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCELoss()

    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            y_pred = model(x)
            loss = loss_fn(y_pred.squeeze(-1), y.float().to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                y_pred = model(x)
                preds.append(y_pred.cpu())
                labels.append(y)
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        auc = roc_auc_score(labels, preds)
        ll = log_loss(labels, preds)
        pr_auc = average_precision_score(labels, preds)
        print(f"Epoch {epoch+1} - AUC {auc:.4f} LogLoss {ll:.4f} PR-AUC {pr_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTR model training")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=1)
    main(parser.parse_args())
