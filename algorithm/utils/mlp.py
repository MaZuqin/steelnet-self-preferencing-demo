"""MLP feature extractor for actor and critic networks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    return nn.ReLU()


class MLPBase(nn.Module):
    """A compact orthogonally-initialized feed-forward base network."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(h) for h in hidden_sizes]
        layers: list[nn.Module] = []
        act_name = str(activation)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(in_dim, out_dim)
            nn.init.orthogonal_(linear.weight, gain=nn.init.calculate_gain("relu" if act_name == "relu" else "tanh"))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(_activation(act_name))
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
