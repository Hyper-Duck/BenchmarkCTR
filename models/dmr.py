from typing import List
import torch
from torch import nn
from deepctr_torch.inputs import SparseFeat, DenseFeat


class DMRModel(nn.Module):
    """Simplified Deep Match to Rank model using an MLP."""

    def __init__(
        self,
        feature_columns,
        hidden_units: List[int] | None = None,
        memory_hops: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        hidden_units = hidden_units or [256, 128, 64]
        self.sparse_feats = [c for c in feature_columns if isinstance(c, SparseFeat)]
        self.dense_feats = [c for c in feature_columns if isinstance(c, DenseFeat)]
        if not self.sparse_feats and not self.dense_feats:
            raise ValueError("DMRModel requires at least one feature")
        self.embeddings = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, c.embedding_dim) for c in self.sparse_feats}
        )
        embed_dim = self.sparse_feats[0].embedding_dim if self.sparse_feats else 1
        input_dim = len(self.dense_feats) + len(self.sparse_feats) * embed_dim
        layers = []
        prev_dim = input_dim
        for _ in range(max(1, int(memory_hops))):
            for u in hidden_units:
                layers.append(nn.Linear(prev_dim, u))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = u
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        dense = (
            torch.cat([x[c.name].float().unsqueeze(1) for c in self.dense_feats], dim=1)
            if self.dense_feats
            else None
        )
        sparse = [self.embeddings[c.name](x[c.name].long()) for c in self.sparse_feats]
        parts = ([] if dense is None else [dense]) + sparse
        concat = torch.cat(parts, dim=1)
        out = self.mlp(concat)
        return torch.sigmoid(out.squeeze(-1))
