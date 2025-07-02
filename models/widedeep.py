import torch
from torch import nn
from typing import List
from .features import SparseFeat, DenseFeat


class WideDeepModel(nn.Module):
    """Simple Wide & Deep model implemented in PyTorch."""

    def __init__(
        self,
        feature_columns: List,
        embedding_dim: int = 8,
        hidden_units: List[int] | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        hidden_units = hidden_units or [256, 128, 64]
        self.sparse_feats = [c for c in feature_columns if isinstance(c, SparseFeat)]
        self.dense_feats = [c for c in feature_columns if isinstance(c, DenseFeat)]
        if not self.sparse_feats and not self.dense_feats:
            raise ValueError("WideDeepModel requires at least one feature")

        # linear (wide) part
        self.linear_sparse = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, 1) for c in self.sparse_feats}
        )
        self.linear_dense = nn.ParameterDict(
            {c.name: nn.Parameter(torch.zeros(1)) for c in self.dense_feats}
        )

        # deep part embeddings
        self.embeddings = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, embedding_dim) for c in self.sparse_feats}
        )
        self.dense_params = nn.ParameterDict(
            {c.name: nn.Parameter(torch.zeros(embedding_dim)) for c in self.dense_feats}
        )
        for p in self.dense_params.values():
            nn.init.xavier_uniform_(p.view(1, -1))

        self.bias = nn.Parameter(torch.zeros(1))

        input_dim = (len(self.sparse_feats) + len(self.dense_feats)) * embedding_dim
        layers = []
        prev_dim = input_dim
        for u in hidden_units:
            layers.append(nn.Linear(prev_dim, u))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = u
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        if self.sparse_feats:
            batch_size = x[self.sparse_feats[0].name].shape[0]
        else:
            batch_size = x[self.dense_feats[0].name].shape[0]
        out = self.bias.expand(batch_size, 1)
        for c in self.sparse_feats:
            out = out + self.linear_sparse[c.name](x[c.name].long())
        for c in self.dense_feats:
            out = out + self.linear_dense[c.name] * x[c.name].float().unsqueeze(1)

        embeddings = [self.embeddings[c.name](x[c.name].long()) for c in self.sparse_feats]
        embeddings += [x[c.name].float().unsqueeze(1) * self.dense_params[c.name] for c in self.dense_feats]
        if embeddings:
            stack = torch.stack(embeddings, dim=1)
            dnn_input = stack.view(batch_size, -1)
            out = out + self.dnn(dnn_input)
        return torch.sigmoid(out.squeeze(-1))
