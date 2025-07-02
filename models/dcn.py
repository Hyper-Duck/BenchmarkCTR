import torch
from torch import nn
from typing import List
from .features import SparseFeat, DenseFeat


class CrossNet(nn.Module):
    """Cross network module used in DCN."""

    def __init__(self, input_dim: int, num_layers: int = 2):
        super().__init__()
        self.kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for w, b in zip(self.kernels, self.bias):
            xw = torch.matmul(xl, w)  # [batch, 1]
            xl = x0 * xw + b + xl
        return xl


class DCNModel(nn.Module):
    """Minimal Deep & Cross Network implementation."""

    def __init__(
        self,
        feature_columns: List,
        embedding_dim: int = 8,
        hidden_units: List[int] | None = None,
        cross_num: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        hidden_units = hidden_units or [256, 128, 64]
        self.sparse_feats = [c for c in feature_columns if isinstance(c, SparseFeat)]
        self.dense_feats = [c for c in feature_columns if isinstance(c, DenseFeat)]
        if not self.sparse_feats and not self.dense_feats:
            raise ValueError("DCNModel requires at least one feature")
        self.embeddings = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, embedding_dim) for c in self.sparse_feats}
        )
        self.dense_params = nn.ParameterDict(
            {c.name: nn.Parameter(torch.zeros(embedding_dim)) for c in self.dense_feats}
        )
        for p in self.dense_params.values():
            nn.init.xavier_uniform_(p.view(1, -1))
        input_dim = (len(self.sparse_feats) + len(self.dense_feats)) * embedding_dim
        self.cross = CrossNet(input_dim, cross_num)
        layers = []
        prev_dim = input_dim
        for u in hidden_units:
            layers.append(nn.Linear(prev_dim, u))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = u
        self.dnn = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_dim + input_dim, 1)

    def forward(self, x):
        if self.sparse_feats:
            batch_size = x[self.sparse_feats[0].name].shape[0]
        else:
            batch_size = x[self.dense_feats[0].name].shape[0]
        embeddings = [self.embeddings[c.name](x[c.name].long()) for c in self.sparse_feats]
        embeddings += [x[c.name].float().unsqueeze(1) * self.dense_params[c.name] for c in self.dense_feats]
        stack = torch.stack(embeddings, dim=1) if embeddings else torch.empty(batch_size, 0, device=x[next(iter(x))].device)
        x0 = stack.view(batch_size, -1)
        cross_out = self.cross(x0)
        deep_out = self.dnn(x0)
        out = self.fc(torch.cat([cross_out, deep_out], dim=1))
        return torch.sigmoid(out.squeeze(-1))
