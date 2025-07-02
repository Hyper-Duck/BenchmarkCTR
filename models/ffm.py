import math
from typing import List
import torch
from torch import nn
from .features import SparseFeat, DenseFeat


class FFMModel(nn.Module):
    """A minimal Field-aware Factorization Machine implementation."""

    def __init__(self, feature_columns: List, embedding_dim: int = 8):
        super().__init__()
        self.sparse_feats = [c for c in feature_columns if isinstance(c, SparseFeat)]
        self.dense_feats = [c for c in feature_columns if isinstance(c, DenseFeat)]
        if not self.sparse_feats and not self.dense_feats:
            raise ValueError("FFMModel requires at least one feature")
        self.fields = [c.name for c in self.sparse_feats + self.dense_feats]
        self.field_index = {name: i for i, name in enumerate(self.fields)}
        self.embed_dim = embedding_dim

        # linear terms
        self.linear_sparse = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, 1) for c in self.sparse_feats}
        )
        self.linear_dense = nn.ParameterDict(
            {c.name: nn.Parameter(torch.zeros(1)) for c in self.dense_feats}
        )

        # field-aware embeddings for sparse features
        self.femb_sparse = nn.ModuleDict()
        for fi in self.sparse_feats:
            sub = nn.ModuleDict()
            for fj in self.fields:
                if fi.name == fj:
                    continue
                sub[fj] = nn.Embedding(fi.vocabulary_size, embedding_dim)
            self.femb_sparse[fi.name] = sub

        # field-aware parameters for dense features (value * embedding)
        self.femb_dense = nn.ParameterDict()
        for fd in self.dense_feats:
            param = nn.Parameter(torch.zeros(len(self.fields), embedding_dim))
            nn.init.xavier_uniform_(param)
            self.femb_dense[fd.name] = param

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x[self.fields[0]].shape[0]
        out = self.bias.expand(batch_size, 1)

        # linear part
        for c in self.sparse_feats:
            out = out + self.linear_sparse[c.name](x[c.name].long())
        for c in self.dense_feats:
            out = out + x[c.name].float().unsqueeze(1) * self.linear_dense[c.name]

        # pairwise field-aware interactions
        inter = 0.0
        for i, fi in enumerate(self.fields[:-1]):
            xi = x[fi]
            for j, fj in enumerate(self.fields[i + 1 :], start=i + 1):
                xj = x[fj]
                if fi in self.femb_sparse:
                    vi = self.femb_sparse[fi][fj](xi.long())
                else:  # dense feature
                    vi = self.femb_dense[fi][self.field_index[fj]].unsqueeze(0).expand(batch_size, self.embed_dim)
                    vi = vi * xi.float().unsqueeze(1)
                if fj in self.femb_sparse:
                    vj = self.femb_sparse[fj][fi](xj.long())
                else:
                    vj = self.femb_dense[fj][self.field_index[fi]].unsqueeze(0).expand(batch_size, self.embed_dim)
                    vj = vj * xj.float().unsqueeze(1)
                inter = inter + torch.sum(vi * vj, dim=1, keepdim=True)
        out = out + inter
        return torch.sigmoid(out.squeeze(-1))
