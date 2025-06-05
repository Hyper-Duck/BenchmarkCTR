from typing import List
import torch
from torch import nn
from deepctr_torch.inputs import SparseFeat, DenseFeat


class DMRModel(nn.Module):
    """Simplified Deep Match to Rank model using an MLP."""

    def __init__(self, feature_columns, hidden_units: List[int] | None = None, dropout: float = 0.5):
        super().__init__()
        hidden_units = hidden_units or [256, 128, 64]
        self.sparse_feats = [c for c in feature_columns if isinstance(c, SparseFeat)]
        self.dense_feats = [c for c in feature_columns if isinstance(c, DenseFeat)]
        self.embeddings = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, c.embedding_dim) for c in self.sparse_feats}
        )
        input_dim = len(self.dense_feats) + len(self.sparse_feats) * self.sparse_feats[0].embedding_dim
        layers = []
        prev_dim = input_dim
        for u in hidden_units:
            layers.append(nn.Linear(prev_dim, u))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = u
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        dense = torch.cat([x[c.name].float() for c in self.dense_feats], dim=-1) if self.dense_feats else None
        sparse = [self.embeddings[c.name](x[c.name].long()).squeeze(1) for c in self.sparse_feats]
        concat = torch.cat(([dense] if dense is not None else []) + sparse, dim=-1)
        out = self.mlp(concat)
        return torch.sigmoid(out.squeeze(-1))
