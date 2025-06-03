import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


def preprocess_criteo(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """Preprocess Criteo dataset.

    Args:
        df: Raw dataframe.
        numeric_cols: List of numerical feature names.
        categorical_cols: List of categorical feature names.

    Returns:
        Tuple of processed dataframe and fitted StandardScaler.
    """
    for col in numeric_cols:
        df[col + "_miss"] = df[col].isnull().astype("int8")
        df[col] = df[col].fillna(0)
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler
