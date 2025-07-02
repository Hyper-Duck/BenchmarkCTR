import torch
from torch.utils.data import Dataset
from .features import SparseFeat, DenseFeat

class CTRDataset(Dataset):
    """Simple Dataset turning dataframe rows into feature dicts for models."""

    def __init__(self, df, feature_columns, label_name="click", cate_mapping=None):
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.label = label_name
        self.sparse_cols = [f.name for f in feature_columns if isinstance(f, SparseFeat)]
        self.dense_cols = [f.name for f in feature_columns if isinstance(f, DenseFeat)]

        self.cate_maps = {}
        for col in self.sparse_cols:
            if cate_mapping and col in cate_mapping:
                self.cate_maps[col] = cate_mapping[col]
            else:
                cats = self.df[col].astype("category").cat.categories
                self.cate_maps[col] = {v: i + 1 for i, v in enumerate(cats)}
            self.df[col] = self.df[col].map(self.cate_maps[col]).fillna(0).astype("int64")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = {}
        for name in self.sparse_cols:
            features[name] = torch.tensor(row[name], dtype=torch.long)
        for name in self.dense_cols:
            features[name] = torch.tensor(row[name], dtype=torch.float32)
        label = torch.tensor(row[self.label], dtype=torch.float32)
        return features, label
