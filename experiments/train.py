"""Training script for CTR models with custom PyTorch implementations."""
import argparse
import os
import sys
import random
import pandas as pd
import torch
import time
import numpy as np
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.features import DenseFeat, SparseFeat
from models import (
    CTRDataset,
    CSVDataset,
    CTNetModel,
    DINModel,
    DMRModel,
    FTRLModel,
    FTRLProximal,
    FFMModel,
    DeepFMModel,
    WideDeepModel,
    DCNModel,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, IterableDataset

from preprocess.utils import apply_preprocess, fit_preprocess, split_dataframe


def _to_model_input(model, features, device):
    """Move batch tensors to the target device."""

    for k, v in features.items():
        features[k] = v.to(device)
    return features


def get_model(
    name: str,
    feature_columns,
    device,
    dropout: float,
    l2: float,
    hidden_units: list[int],
    cross_num: int,
    memory_hops: int,
    conv_layers: int,
    attention_hidden_size: int | None,
    embed_dim: int,
):
    if name.lower() == "deepfm":
        return DeepFMModel(
            feature_columns,
            embedding_dim=embed_dim,
            hidden_units=hidden_units,
            dropout=dropout,
        )
    if name.lower() == "ffm":
        return FFMModel(feature_columns, embedding_dim=embed_dim)
    if name.lower() == "widedeep":
        return WideDeepModel(
            feature_columns,
            embedding_dim=embed_dim,
            hidden_units=hidden_units,
            dropout=dropout,
        )
    if name.lower() == "dcn":
        return DCNModel(
            feature_columns,
            embedding_dim=embed_dim,
            hidden_units=hidden_units,
            cross_num=cross_num,
            dropout=dropout,
        )
    if name.lower() == "ftrl":
        return FTRLModel(feature_columns)
    if name.lower() == "dmr":
        return DMRModel(feature_columns, hidden_units=hidden_units, memory_hops=memory_hops, dropout=dropout)
    if name.lower() == "din":
        return DINModel(feature_columns, hidden_units=hidden_units, attention_hidden_size=attention_hidden_size, dropout=dropout)
    if name.lower() == "ctnet":
        return CTNetModel(feature_columns, hidden_units=hidden_units, conv_layers=conv_layers, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")


def evaluate(model, data_loader, device, desc: str = "Eval"):
    """Evaluate model on data loader and return metrics and inference time."""

    model.eval()
    preds, labels = [], []
    start = time.time()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False, dynamic_ncols=True, file=sys.stdout):
            x, y = batch
            x = _to_model_input(model, x, device)
            y_pred = model(x)
            preds.append(y_pred.cpu())
            labels.append(y)
    infer_time = time.time() - start

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    auc = roc_auc_score(labels, preds)
    ll = log_loss(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    brier = brier_score_loss(labels, preds)
    return auc, ll, pr_auc, brier, infer_time


def main(args: argparse.Namespace, explicit: set[str]) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.train_data and args.lazy_chunk_size > 0:
        sample_df = pd.read_csv(args.train_data, nrows=100)
        df_train = df_val = df_test = None
    else:
        pt_path = os.path.splitext(args.data)[0] + ".pt"
        if os.path.exists(pt_path):
            df = torch.load(pt_path, weights_only=False)
        else:
            df = pd.read_csv(args.data)
            torch.save(df, pt_path)
        df_train, df_val, df_test = split_dataframe(df, random_state=args.seed)

    df_cols = sample_df if args.train_data and args.lazy_chunk_size > 0 else df
    numeric_cols = [c for c in df_cols.columns if c.startswith("I")]
    categorical_cols = [c for c in df_cols.columns if c.startswith("C")]
    if not numeric_cols and not categorical_cols:
        exclude_cols = {"click", "label", "treatment", "conversion", "visit"}
        for col in df_cols.columns:
            if col.lower() in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(df_cols[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    label_candidates = ["click", "label", "conversion", "visit"]
    label_col = None
    for name in label_candidates:
        if name in df_cols.columns:
            label_col = name
            break
    if label_col is None:
        raise ValueError(
            "Could not determine label column. Expected one of: "
            + ", ".join(label_candidates)
        )

    if args.train_data and args.lazy_chunk_size > 0:
        vocab_size = {col: sample_df[col].nunique() + 1 for col in categorical_cols}
        feature_columns = [DenseFeat(col, 1) for col in numeric_cols]
        hidden_units = [int(x) for x in str(args.dnn_hidden_units).replace(",", " ").split() if x]
        feature_columns += [
            SparseFeat(col, vocabulary_size=vocab_size[col], embedding_dim=args.embed_dim)
            for col in categorical_cols
        ]

        train_dataset = CSVDataset(args.train_data, feature_columns, label_name=label_col, chunksize=args.lazy_chunk_size)
        val_dataset = CSVDataset(
            args.val_data,
            feature_columns,
            label_name=label_col,
            cate_mapping=train_dataset.cate_maps,
            chunksize=args.lazy_chunk_size,
        )
        test_dataset = CSVDataset(
            args.test_data,
            feature_columns,
            label_name=label_col,
            cate_mapping=train_dataset.cate_maps,
            chunksize=args.lazy_chunk_size,
        )
    else:
        df_train, scaler, rare_maps = fit_preprocess(df_train, numeric_cols, categorical_cols)
        df_val = apply_preprocess(df_val, numeric_cols, categorical_cols, scaler, rare_maps)
        df_test = apply_preprocess(df_test, numeric_cols, categorical_cols, scaler, rare_maps)

        vocab_size = {col: df_train[col].nunique() + 1 for col in categorical_cols}

        feature_columns = [DenseFeat(col, 1) for col in numeric_cols]
        hidden_units = [int(x) for x in str(args.dnn_hidden_units).replace(",", " ").split() if x]

        feature_columns += [
            SparseFeat(col, vocabulary_size=vocab_size[col], embedding_dim=args.embed_dim)
            for col in categorical_cols
        ]

        train_dataset = CTRDataset(df_train, feature_columns, label_name=label_col)
        val_dataset = CTRDataset(
            df_val, feature_columns, label_name=label_col, cate_mapping=train_dataset.cate_maps
        )
        test_dataset = CTRDataset(
            df_test, feature_columns, label_name=label_col, cate_mapping=train_dataset.cate_maps
        )

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    num_workers = min(16, os.cpu_count())
    pin_memory = True
    batch_size = 2048

    shuffle_train = not isinstance(train_dataset, IterableDataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=loader_gen,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=loader_gen,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=loader_gen,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU for training")
    model = get_model(
        args.model,
        feature_columns,
        device,
        args.dropout,
        args.l2,
        hidden_units,
        args.cross_num,
        args.memory_hops,
        args.conv_layers,
        args.attention_hidden_size,
        args.embed_dim,
    )
    model.to(device)

    if args.model.lower() == "ftrl":
        optimizer = FTRLProximal(
            model.parameters(),
            alpha=args.alpha,
            beta=args.beta,
            l1=args.l1,
            l2=args.l2,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    loss_fn = torch.nn.BCELoss()

    start_epoch = 0
    if args.start_from_checkpoint:
        ckpt = torch.load(args.start_from_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0)
        else:
            model.load_state_dict(ckpt)
            import re

            m = re.search(r"_epoch_(\d+)\.pt$", os.path.basename(args.start_from_checkpoint))
            if m:
                start_epoch = int(m.group(1))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    model.train()
    total_train_time = 0.0
    total_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch, total_epochs):
        start_train = time.time()
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{total_epochs}",
            unit="batch",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for batch in progress:
            optimizer.zero_grad()
            x, y = batch
            x = _to_model_input(model, x, device)
            y_pred = model(x)
            loss = loss_fn(y_pred.squeeze(-1), y.float().to(device))
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        train_time = time.time() - start_train
        total_train_time += train_time

        val_auc, val_ll, val_pr, val_brier, val_infer = evaluate(model, val_loader, device, desc="Val")
        print(
            f"Epoch {epoch+1} - AUC {val_auc:.4f} LogLoss {val_ll:.4f} PR-AUC {val_pr:.4f}"
        )
        pd.DataFrame(
            [
                {
                    "epoch": epoch + 1,
                    "val_auc": val_auc,
                    "val_logloss": val_ll,
                    "val_pr_auc": val_pr,
                }
            ]
        ).to_csv(
            args.log_file,
            mode="a",
            index=False,
            header=not os.path.exists(args.log_file),
        )
        ckpt_path = os.path.join(
            args.checkpoint_dir, f"{args.model}_epoch_{epoch+1}.pt"
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            ckpt_path,
        )

    test_auc, test_ll, test_pr, test_brier, test_infer = evaluate(model, test_loader, device, desc="Test")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.model.lower() == "ftrl":
        result = {"model": args.model, "epochs": total_epochs}
        for name in ["alpha", "beta", "l1", "l2", "seed"]:
            if name in explicit:
                result[name] = getattr(args, name)
        result.update(
            {
                "val_auc": val_auc,
                "val_logloss": val_ll,
                "val_pr_auc": val_pr,
                "val_brier": val_brier,
                "val_infer_time": val_infer,
                "test_auc": test_auc,
                "test_logloss": test_ll,
                "test_pr_auc": test_pr,
                "test_brier": test_brier,
                "test_infer_time": test_infer,
                "train_time": total_train_time,
            }
        )
        results = [result]
    else:
        result = {"model": args.model, "epochs": total_epochs}
        for name in [
            "lr",
            "l2",
            "dropout",
            "dnn_hidden_units",
            "embed_dim",
            "cross_num",
            "memory_hops",
            "conv_layers",
            "attention_hidden_size",
            "seed",
        ]:
            if name in explicit:
                result[name] = getattr(args, name)
        result.update(
            {
                "val_auc": val_auc,
                "val_logloss": val_ll,
                "val_pr_auc": val_pr,
                "val_brier": val_brier,
                "val_infer_time": val_infer,
                "test_auc": test_auc,
                "test_logloss": test_ll,
                "test_pr_auc": test_pr,
                "test_brier": test_brier,
                "test_infer_time": test_infer,
                "train_time": total_train_time,
            }
        )
        results = [result]
    pd.DataFrame(results).to_csv(
        args.output,
        mode="a",
        index=False,
        header=not os.path.exists(args.output),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTR model training")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name: DeepFM/WideDeep/DCN/FFM/FTRL/DMR/DIN/CTNet",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="DNN dropout")
    parser.add_argument("--l2", type=float, default=1e-5, help="L2 regularization")

    parser.add_argument(
        "--dnn-hidden-units",
        type=str,
        default="256 128 64",
        help="Space separated list for DNN hidden units",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=8,
        help="Embedding dimension for categorical features",
    )
    parser.add_argument(
        "--cross-num",
        type=int,
        default=2,
        help="Number of cross layers for DCN",
    )
    parser.add_argument(
        "--memory-hops",
        type=int,
        default=1,
        help="Number of memory hops for DMR",
    )
    parser.add_argument(
        "--conv-layers",
        type=int,
        default=3,
        help="Number of convolutional layers for CTNet",
    )
    parser.add_argument(
        "--attention-hidden-size",
        type=int,
        default=None,
        help="Attention hidden size for DIN",
    )
    parser.add_argument(
        "--lazy-chunk-size",
        type=int,
        default=0,
        help="If > 0, read CSV lazily with this chunk size",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to preprocessed training CSV for lazy mode",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation CSV for lazy mode",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test CSV for lazy mode",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="FTRL alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="FTRL beta")
    parser.add_argument("--l1", type=float, default=1.0, help="FTRL L1 regularization")
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--start-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/train_metrics.csv",
        help="Path to save per-epoch validation metrics",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/result.csv", help="Path to save metrics CSV"
    )
    args = parser.parse_args()
    argv = sys.argv[1:]
    explicit: set[str] = set()
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        for option in action.option_strings:
            if option in argv:
                explicit.add(action.dest)
                break
    if args.model.lower() == "ftrl" and args.output == "outputs/result.csv":
        args.output = "outputs/ftrl.csv"
    main(args, explicit)
