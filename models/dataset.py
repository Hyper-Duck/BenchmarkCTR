import torch
from torch.utils.data import Dataset
from deepctr_torch.inputs import SparseFeat, DenseFeat

class CTRDataset(Dataset):
    """Simple Dataset turning dataframe rows into feature dicts for models."""

    def __init__(self, df, feature_columns, label_name="click"):
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.label = label_name
        self.sparse_cols = [f.name for f in feature_columns if isinstance(f, SparseFeat)]
        self.dense_cols = [f.name for f in feature_columns if isinstance(f, DenseFeat)]

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
