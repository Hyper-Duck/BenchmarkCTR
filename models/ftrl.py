import math
from typing import Iterable

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from deepctr_torch.inputs import SparseFeat, DenseFeat


class FTRLProximal(Optimizer):
    """Simplified FTRL-Proximal optimizer implementation."""

    def __init__(self, params: Iterable, lr: float = 1.0, alpha: float = 0.1, beta: float = 1.0, l1: float = 0.0, l2: float = 0.0):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, l1=l1, l2=l2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta = group["beta"]
            l1 = group["l1"]
            l2 = group["l2"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["n"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["z"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                n = state["n"]
                z = state["z"]
                n_old = n.clone()
                n.add_(grad.pow(2))
                sigma = (n.sqrt() - n_old.sqrt()) / alpha
                z.add_(grad - sigma * p.data)
                # update parameter
                mask = torch.abs(z) <= l1
                p.data[mask] = 0.0
                denom = (beta + n.sqrt()) / alpha + l2
                p.data[~mask] = - (z[~mask] - torch.sign(z[~mask]) * l1) / denom[~mask]
                p.data.mul_(lr)
        return loss


class FTRLModel(nn.Module):
    """Logistic regression style model trained with FTRL."""

    def __init__(self, feature_columns):
        super().__init__()
        self.sparse_feats = [c for c in feature_columns if isinstance(c, SparseFeat)]
        self.dense_feats = [c for c in feature_columns if isinstance(c, DenseFeat)]
        self.embeddings = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, 1) for c in self.sparse_feats}
        )
        if self.dense_feats:
            self.dense_layer = nn.Linear(len(self.dense_feats), 1, bias=False)
        else:
            self.dense_layer = None
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.sparse_feats:
            batch_size = x[self.sparse_feats[0].name].shape[0]
        elif self.dense_feats:
            batch_size = x[self.dense_feats[0].name].shape[0]
        else:
            raise ValueError("FTRLModel requires at least one feature")

        out = self.bias.expand(batch_size, 1)
        if self.dense_layer is not None:
            dense = torch.cat([x[c.name].float() for c in self.dense_feats], dim=-1)
            out = out + self.dense_layer(dense)
        for c in self.sparse_feats:
            emb = self.embeddings[c.name](x[c.name].long())
            out = out + emb
        return torch.sigmoid(out.squeeze(-1))
