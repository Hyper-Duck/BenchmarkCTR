from typing import List
import torch
from torch import nn
from .features import SparseFeat, DenseFeat

class CTNetModel(nn.Module):
    """Continual Transfer Network with CNN-based feature extractor."""

    def __init__(
        self,
        feature_columns: List,
        embedding_dim: int = 8,
        conv_layers: int = 2,
        conv_channels: int = 32,
        hidden_units: List[int] | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        hidden_units = hidden_units or [128, 64]
        self.sparse_feats = [c for c in feature_columns if isinstance(c, SparseFeat)]
        self.dense_feats = [c for c in feature_columns if isinstance(c, DenseFeat)]
        if not self.sparse_feats and not self.dense_feats:
            raise ValueError("CTNetModel requires at least one feature")

        # embeddings for sparse and dense features
        self.embeddings = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, embedding_dim) for c in self.sparse_feats}
        )
        self.dense_transform = nn.ModuleDict(
            {c.name: nn.Linear(1, embedding_dim) for c in self.dense_feats}
        )
        self.num_fields = len(self.sparse_feats) + len(self.dense_feats)

        convs = []
        in_ch = embedding_dim
        for _ in range(max(1, int(conv_layers))):
            convs.append(nn.Conv1d(in_ch, conv_channels, kernel_size=3, padding=1))
            convs.append(nn.ReLU())
            convs.append(nn.Dropout(dropout))
            in_ch = conv_channels
        self.extractor = nn.Sequential(*convs)

        flattened_dim = conv_channels * self.num_fields
        self.gate = nn.Linear(flattened_dim, flattened_dim)

        layers = []
        prev_dim = flattened_dim
        for u in hidden_units:
            layers.append(nn.Linear(prev_dim, u))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = u
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embeds = []
        for c in self.sparse_feats:
            embeds.append(self.embeddings[c.name](x[c.name].long()).unsqueeze(1))
        for c in self.dense_feats:
            value = x[c.name].float().unsqueeze(1).unsqueeze(1)
            embeds.append(self.dense_transform[c.name](value))
        stack = torch.cat(embeds, dim=1)  # [B, F, D]
        conv_in = stack.permute(0, 2, 1)   # [B, D, F]
        conv_out = self.extractor(conv_in)
        flat = conv_out.view(conv_out.size(0), -1)
        gated = flat * torch.sigmoid(self.gate(flat))
        out = self.mlp(gated)
        return torch.sigmoid(out.squeeze(-1))
