"""Distribution helpers for discrete, continuous, and MultiDiscrete actions."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn
from torch.distributions import Categorical, Normal


class FixedCategorical(Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        if actions.dim() > 1 and actions.size(-1) == 1:
            actions = actions.squeeze(-1)
        actions = actions.long()
        return super().log_prob(actions).unsqueeze(-1)

    def entropy(self):
        return super().entropy().unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(dim=-1, keepdim=True)

    def mode(self):
        return self.mean


class FixedMultiCategorical:
    def __init__(self, logits: Sequence[torch.Tensor]) -> None:
        self.dists = [FixedCategorical(logits=item) for item in logits]

    def sample(self):
        return torch.cat([dist.sample() for dist in self.dists], dim=-1)

    def log_probs(self, actions):
        parts = []
        for idx, dist in enumerate(self.dists):
            parts.append(dist.log_probs(actions[..., idx : idx + 1]))
        return torch.stack(parts, dim=0).sum(dim=0)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self.dists], dim=0).sum(dim=0)

    def mode(self):
        return torch.cat([dist.mode() for dist in self.dists], dim=-1)


def _mask_logits(logits: torch.Tensor, available_actions=None) -> torch.Tensor:
    if available_actions is None:
        return logits
    return logits.masked_fill(available_actions <= 0.0, -1e10)


class CategoricalLayer(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, gain: float = 0.01) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, action_dim)
        nn.init.orthogonal_(self.linear.weight, gain=gain)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, features, available_actions=None):
        logits = _mask_logits(self.linear(features), available_actions)
        return FixedCategorical(logits=logits)


class DiagGaussianLayer(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, gain: float = 0.01) -> None:
        super().__init__()
        self.fc_mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        nn.init.orthogonal_(self.fc_mean.weight, gain=gain)
        nn.init.constant_(self.fc_mean.bias, 0.0)

    def forward(self, features, available_actions=None):
        mean = self.fc_mean(features)
        std = self.log_std.exp().expand_as(mean)
        return FixedNormal(mean, std)


class MultiCategoricalLayer(nn.Module):
    def __init__(self, input_dim: int, action_dims: Sequence[int], gain: float = 0.01) -> None:
        super().__init__()
        self.action_dims = [int(dim) for dim in action_dims]
        self.branches = nn.ModuleList([nn.Linear(input_dim, dim) for dim in self.action_dims])
        for branch in self.branches:
            nn.init.orthogonal_(branch.weight, gain=gain)
            nn.init.constant_(branch.bias, 0.0)

    def _split_masks(self, available_actions) -> List:
        if available_actions is None:
            return [None] * len(self.action_dims)
        if isinstance(available_actions, (list, tuple)):
            return list(available_actions)
        return list(torch.split(available_actions, self.action_dims, dim=-1))

    def forward(self, features, available_actions=None):
        masks = self._split_masks(available_actions)
        logits = [_mask_logits(branch(features), mask) for branch, mask in zip(self.branches, masks)]
        return FixedMultiCategorical(logits)
