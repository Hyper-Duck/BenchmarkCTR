"""Preprocess Criteo Uplift Modeling Dataset for DeepCTR models."""

import argparse
import os
import pickle

import pandas as pd

from .utils import apply_preprocess, fit_preprocess, split_dataframe


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)

    numeric_cols = [c for c in df.columns if c.startswith("I")]
    categorical_cols = [c for c in df.columns if c.startswith("C")]
    if not numeric_cols and not categorical_cols:
        # fallback for datasets without I/C prefixed columns
        exclude_cols = {"click", "label", "treatment", "conversion", "visit"}
        for col in df.columns:
            if col.lower() in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    df_train, df_val, df_test = split_dataframe(df)
    df_train, scaler, rare_maps = fit_preprocess(df_train, numeric_cols, categorical_cols)
    df_val = apply_preprocess(df_val, numeric_cols, categorical_cols, scaler, rare_maps)
    df_test = apply_preprocess(df_test, numeric_cols, categorical_cols, scaler, rare_maps)

    os.makedirs(args.output_dir, exist_ok=True)
    df_train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    with open(os.path.join(args.output_dir, "preprocess.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "rare_maps": rare_maps}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Criteo dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to raw Criteo CSV")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed splits",
    )
    main(parser.parse_args())
