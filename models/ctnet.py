from typing import List
import torch
from torch import nn
from deepctr_torch.inputs import SparseFeat, DenseFeat


class CTNetModel(nn.Module):
    """Simplified Continual Transfer Network with gating."""

    def __init__(
        self,
        feature_columns,
        hidden_units: List[int] | None = None,
        conv_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_units is None:
            hidden_units = [256] * conv_layers
        self.sparse_feats = [c for c in feature_columns if isinstance(c, SparseFeat)]
        self.dense_feats = [c for c in feature_columns if isinstance(c, DenseFeat)]
        if not self.sparse_feats and not self.dense_feats:
            raise ValueError("CTNetModel requires at least one feature")
        embed_dim = self.sparse_feats[0].embedding_dim if self.sparse_feats else 1
        self.embeddings = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, embed_dim) for c in self.sparse_feats}
        )
        input_dim = len(self.dense_feats) + len(self.sparse_feats) * embed_dim
        self.gate = nn.Linear(input_dim, input_dim)
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
        dense = (
            torch.cat([x[c.name].float().unsqueeze(1) for c in self.dense_feats], dim=1)
            if self.dense_feats
            else None
        )
        sparse = [self.embeddings[c.name](x[c.name].long()) for c in self.sparse_feats]
        parts = ([] if dense is None else [dense]) + sparse
        concat = torch.cat(parts, dim=1)
        gated = concat * torch.sigmoid(self.gate(concat))
        out = self.mlp(gated)
        return torch.sigmoid(out.squeeze(-1))
