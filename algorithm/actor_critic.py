"""Actor and critic networks used by HAPPO/MAPPO policies."""

from __future__ import annotations

from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .utils.act import ACTLayer
from .utils.mlp import MLPBase
from .utils.rnn import GRULayer
from .utils.util import check

DimLike = Union[int, Sequence[int]]


def _flat_dim(dim: DimLike) -> int:
    if isinstance(dim, int):
        return dim
    return int(prod(dim))


class Actor(nn.Module):
    """Policy network.

    The actor receives a local observation and produces an action. Build one
    actor per role when the platform, LEs, and SMEs use different observation
    or action dimensions. SteelNet policies always keep the GRU branch enabled
    so local observations can be filtered through history before action output.
    """

    def __init__(
        self,
        obs_dim: DimLike,
        action_space=None,
        action_dim=None,
        action_type: Optional[str] = None,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
        use_layer_norm: bool = True,
        use_recurrent: bool = True,
        recurrent_hidden_size: Optional[int] = None,
        recurrent_layers: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = _flat_dim(obs_dim)
        # Kept as an argument for compatibility with older configs; this
        # project always uses GRU for the partially observable SteelNet setting.
        self.use_recurrent = True

        self.base = MLPBase(
            input_dim=self.obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

        feature_dim = self.base.output_dim
        rnn_hidden = recurrent_hidden_size or feature_dim
        self.rnn = GRULayer(
            input_dim=feature_dim,
            hidden_dim=rnn_hidden,
            num_layers=recurrent_layers,
        )
        feature_dim = rnn_hidden

        self.act = ACTLayer(
            input_dim=feature_dim,
            action_space=action_space,
            action_dim=action_dim,
            action_type=action_type,
        )
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _features(self, obs, rnn_states=None, masks=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        obs = check(obs, device=self.device, dtype=torch.float32)
        obs = obs.reshape(-1, self.obs_dim)
        features = self.base(obs)

        if self.rnn is not None:
            rnn_states = check(rnn_states, device=self.device, dtype=torch.float32)
            masks = check(masks, device=self.device, dtype=torch.float32)
            features, rnn_states = self.rnn(features, rnn_states, masks)

        return features, rnn_states

    def forward(
        self,
        obs,
        rnn_states=None,
        masks=None,
        deterministic: bool = False,
        available_actions=None,
    ):
        features, rnn_states = self._features(obs, rnn_states, masks)
        available_actions = check(available_actions, device=self.device, dtype=torch.float32)
        actions, action_log_probs = self.act(
            features,
            deterministic=deterministic,
            available_actions=available_actions,
        )
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, actions, rnn_states=None, masks=None, available_actions=None):
        features, _ = self._features(obs, rnn_states, masks)
        actions = check(actions, device=self.device, dtype=None)
        available_actions = check(available_actions, device=self.device, dtype=torch.float32)
        return self.act.evaluate_actions(features, actions, available_actions)

    def get_dist(self, obs, rnn_states=None, masks=None, available_actions=None):
        features, _ = self._features(obs, rnn_states, masks)
        available_actions = check(available_actions, device=self.device, dtype=torch.float32)
        return self.act.get_dist(features, available_actions)


class Critic(nn.Module):
    """Centralized value network.

    The critic receives the centralized/global state and outputs V(s). It also
    uses GRU by default to align value estimation with delayed inventory states.
    """

    def __init__(
        self,
        cent_obs_dim: DimLike,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
        use_layer_norm: bool = True,
        use_recurrent: bool = True,
        recurrent_hidden_size: Optional[int] = None,
        recurrent_layers: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.cent_obs_dim = _flat_dim(cent_obs_dim)
        # Kept as an argument for compatibility with older configs; this
        # project always uses GRU for the partially observable SteelNet setting.
        self.use_recurrent = True

        self.base = MLPBase(
            input_dim=self.cent_obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

        feature_dim = self.base.output_dim
        rnn_hidden = recurrent_hidden_size or feature_dim
        self.rnn = GRULayer(
            input_dim=feature_dim,
            hidden_dim=rnn_hidden,
            num_layers=recurrent_layers,
        )
        feature_dim = rnn_hidden

        self.value_head = nn.Linear(feature_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, cent_obs, rnn_states=None, masks=None):
        cent_obs = check(cent_obs, device=self.device, dtype=torch.float32)
        cent_obs = cent_obs.reshape(-1, self.cent_obs_dim)
        features = self.base(cent_obs)

        if self.rnn is not None:
            rnn_states = check(rnn_states, device=self.device, dtype=torch.float32)
            masks = check(masks, device=self.device, dtype=torch.float32)
            features, rnn_states = self.rnn(features, rnn_states, masks)

        values = self.value_head(features)
        return values, rnn_states
