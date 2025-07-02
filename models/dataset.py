import os
import torch
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
from .features import SparseFeat, DenseFeat

class CTRDataset(Dataset):
    """Dataset that stores all feature columns as tensors for fast loading."""

    def __init__(self, df, feature_columns, label_name="click", cate_mapping=None):
        df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.label = label_name
        self.sparse_cols = [f.name for f in feature_columns if isinstance(f, SparseFeat)]
        self.dense_cols = [f.name for f in feature_columns if isinstance(f, DenseFeat)]

        self.cate_maps = {}
        for col in self.sparse_cols:
            if cate_mapping and col in cate_mapping:
                self.cate_maps[col] = cate_mapping[col]
            else:
                cats = df[col].astype("category").cat.categories
                self.cate_maps[col] = {v: i + 1 for i, v in enumerate(cats)}
            df[col] = df[col].map(self.cate_maps[col]).fillna(0).astype("int64")

        self.sparse_tensors = {
            col: torch.tensor(df[col].values, dtype=torch.long) for col in self.sparse_cols
        }
        self.dense_tensors = {
            col: torch.tensor(df[col].values, dtype=torch.float32) for col in self.dense_cols
        }
        self.labels = torch.tensor(df[self.label].values, dtype=torch.float32)
        self.df = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = {name: tensor[idx] for name, tensor in self.sparse_tensors.items()}
        for name, tensor in self.dense_tensors.items():
            features[name] = tensor[idx]
        label = self.labels[idx]
        return features, label

    def save(self, path: str) -> None:
        """Serialize tensors and category maps to a file."""
        torch.save(
            {
                "sparse_tensors": self.sparse_tensors,
                "dense_tensors": self.dense_tensors,
                "labels": self.labels,
                "cate_maps": self.cate_maps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, feature_columns, label_name: str = "click"):
        """Load dataset previously saved with :meth:`save`."""
        data = torch.load(path, map_location="cpu")
        obj = cls.__new__(cls)
        obj.feature_columns = feature_columns
        obj.label = label_name
        obj.sparse_cols = [f.name for f in feature_columns if isinstance(f, SparseFeat)]
        obj.dense_cols = [f.name for f in feature_columns if isinstance(f, DenseFeat)]
        obj.cate_maps = data["cate_maps"]
        obj.sparse_tensors = data["sparse_tensors"]
        obj.dense_tensors = data["dense_tensors"]
        obj.labels = data["labels"]
        obj.df = None
        return obj


class CSVDataset(IterableDataset):
    """Iterable dataset that lazily reads samples from a CSV file."""

    def __init__(
        self,
        file_path: str,
        feature_columns,
        label_name: str = "click",
        cate_mapping: dict[str, dict[str, int]] | None = None,
        chunksize: int = 10000,
    ) -> None:
        self.file_path = file_path
        self.feature_columns = feature_columns
        self.label = label_name
        self.chunksize = max(1, int(chunksize))

        self.sparse_cols = [f.name for f in feature_columns if isinstance(f, SparseFeat)]
        self.dense_cols = [f.name for f in feature_columns if isinstance(f, DenseFeat)]

        self.cate_maps: dict[str, dict[str, int]] = cate_mapping or {
            col: {} for col in self.sparse_cols
        }
        self._next_id = {
            col: max(self.cate_maps[col].values(), default=0) + 1 for col in self.sparse_cols
        }

    def __iter__(self):
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunksize):
            for _, row in chunk.iterrows():
                features = {}
                for name in self.sparse_cols:
                    val = row[name]
                    mapping = self.cate_maps.setdefault(name, {})
                    if val not in mapping:
                        mapping[val] = self._next_id[name]
                        self._next_id[name] += 1
                    features[name] = torch.tensor(mapping[val], dtype=torch.long)
                for name in self.dense_cols:
                    features[name] = torch.tensor(row[name], dtype=torch.float32)
                label = torch.tensor(row[self.label], dtype=torch.float32)
                yield features, label
