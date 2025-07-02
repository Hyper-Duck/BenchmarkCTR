import math
from typing import List, Iterable
import torch
from torch import nn
from .features import SparseFeat, DenseFeat
from torch.optim.optimizer import Optimizer, required


class FTRLModel(nn.Module):
    """Simple logistic regression model for FTRL optimizer."""

    def __init__(self, feature_columns: List):
        super().__init__()
        self.sparse_feats = [c for c in feature_columns if isinstance(c, SparseFeat)]
        self.dense_feats = [c for c in feature_columns if isinstance(c, DenseFeat)]
        if not self.sparse_feats and not self.dense_feats:
            raise ValueError("FTRLModel requires at least one feature")
        self.linear_sparse = nn.ModuleDict(
            {c.name: nn.Embedding(c.vocabulary_size, 1) for c in self.sparse_feats}
        )
        self.linear_dense = nn.ParameterDict(
            {c.name: nn.Parameter(torch.zeros(1)) for c in self.dense_feats}
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.sparse_feats:
            batch_size = x[self.sparse_feats[0].name].shape[0]
        else:
            batch_size = x[self.dense_feats[0].name].shape[0]
        out = self.bias.expand(batch_size)
        for c in self.sparse_feats:
            out = out + self.linear_sparse[c.name](x[c.name].long()).squeeze(-1)
        for c in self.dense_feats:
            out = out + self.linear_dense[c.name] * x[c.name].float()
        return torch.sigmoid(out)


class FTRLProximal(Optimizer):
    """PyTorch implementation of FTRL-Proximal optimizer."""

    def __init__(self, params: Iterable, alpha=required, beta=1.0, l1=0.0, l2=0.0):
        defaults = dict(alpha=alpha, beta=beta, l1=l1, l2=l2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            l1 = group['l1']
            l2 = group['l2']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['n'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['z'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                n, z = state['n'], state['z']
                n_old = n.clone()
                n.add_(grad.pow(2))
                sigma = (n.sqrt() - n_old.sqrt()) / alpha
                z.add_(grad - sigma * p.data)
                # update weight
                sign = z.sign()
                mask = z.abs() < l1
                p.data = torch.where(
                    mask,
                    torch.zeros_like(p),
                    -(z - sign * l1) / ((beta + n.sqrt()) / alpha + l2)
                )
        return loss
