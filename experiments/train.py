"""Training script skeleton for CTR models using DeepCTR-torch."""
import argparse
import os
import pandas as pd
import torch
from deepctr_torch.models import DeepFM, FFM, WideDeep, DCN
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.utils import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

from preprocess.utils import preprocess_criteo


def get_model(name: str, feature_columns, device):
    if name.lower() == "deepfm":
        return DeepFM(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            task="binary",
            dnn_hidden_units=[256, 128, 64],
            dnn_dropout=0.5,
            l2_reg_embedding=1e-5,
            device=device,
        )
    if name.lower() == "ffm":
        return FFM(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            task="binary",
            dnn_hidden_units=[256, 128, 64],
            device=device,
        )
    if name.lower() == "widedeep":
        return WideDeep(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            task="binary",
            dnn_hidden_units=[256, 128, 64],
            dnn_dropout=0.5,
            device=device,
        )
    if name.lower() == "dcn":
        return DCN(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            task="binary",
            dnn_hidden_units=[256, 128, 64],
            cross_num=2,
            device=device,
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate(model, data_loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            y_pred = model(x)
            preds.append(y_pred.cpu())
            labels.append(y)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    auc = roc_auc_score(labels, preds)
    ll = log_loss(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    return auc, ll, pr_auc


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
    test_dataset = Dataset(df_test, feature_columns, label_name="click")

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, feature_columns, device)
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

        val_auc, val_ll, val_pr = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1} - AUC {val_auc:.4f} LogLoss {val_ll:.4f} PR-AUC {val_pr:.4f}"
        )

    test_auc, test_ll, test_pr = evaluate(model, test_loader, device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame([
        {
            "model": args.model,
            "epochs": args.epochs,
            "val_auc": val_auc,
            "val_logloss": val_ll,
            "val_pr_auc": val_pr,
            "test_auc": test_auc,
            "test_logloss": test_ll,
            "test_pr_auc": test_pr,
        }
    ]).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTR model training")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model", type=str, default="DeepFM", help="Model name: DeepFM/FFM/WideDeep/DCN")
    parser.add_argument(
        "--output", type=str, default="outputs/result.csv", help="Path to save metrics CSV"
    )
    main(parser.parse_args())
