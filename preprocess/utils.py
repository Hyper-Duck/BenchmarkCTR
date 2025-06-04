"""Utility functions for data preprocessing."""

from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_dataframe(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_state: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split dataframe into train/val/test subsets."""

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train : n_train + n_val]
    df_test = df.iloc[n_train + n_val :]

    return df_train, df_val, df_test

def _apply_missing_flags(
    df: pd.DataFrame, numeric_cols: Iterable[str], categorical_cols: Iterable[str]
) -> pd.DataFrame:
    """Handle missing values and add indicator columns."""

    for col in numeric_cols:
        df[col + "_miss"] = df[col].isnull().astype("int8")
        df[col] = df[col].fillna(0)
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")
    return df


def fit_preprocess(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    freq_threshold: int = 100,
) -> Tuple[pd.DataFrame, StandardScaler, Dict[str, Iterable[str]]]:
    """Fit preprocessing objects on training dataframe and transform it."""

    df = _apply_missing_flags(df, numeric_cols, categorical_cols)

    # compute rare category sets
    rare_maps: Dict[str, Iterable[str]] = {}
    for col in categorical_cols:
        freq = df[col].value_counts()
        rare_maps[col] = freq[freq < freq_threshold].index.tolist()
        df[col] = df[col].where(~df[col].isin(rare_maps[col]), "rare")

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler, rare_maps


def apply_preprocess(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: StandardScaler,
    rare_maps: Dict[str, Iterable[str]],
) -> pd.DataFrame:
    """Apply fitted preprocessing to new dataframe."""

    df = _apply_missing_flags(df, numeric_cols, categorical_cols)
    for col in categorical_cols:
        df[col] = df[col].where(~df[col].isin(rare_maps[col]), "rare")

    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

