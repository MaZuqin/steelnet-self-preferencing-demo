"""Action output layer that maps network features to action distributions."""

from __future__ import annotations

from typing import Optional, Sequence

from torch import nn

from .distributions import CategoricalLayer, DiagGaussianLayer, MultiCategoricalLayer


def infer_action_spec(action_space=None, action_dim=None, action_type: Optional[str] = None):
    if action_type is not None:
        kind = action_type.lower()
        if kind in {"discrete", "categorical"}:
            return "discrete", int(action_dim if action_dim is not None else action_space.n)
        if kind in {"continuous", "box", "gaussian"}:
            return "continuous", int(action_dim if action_dim is not None else action_space.shape[0])
        if kind in {"multidiscrete", "multi_discrete"}:
            dims = action_dim if action_dim is not None else action_space.nvec
            return "multidiscrete", [int(v) for v in dims]
        raise ValueError(f"Unsupported action_type: {action_type}")

    if action_space is None and action_dim is None:
        raise ValueError("Either action_space or action_dim/action_type must be provided.")

    if action_space is not None:
        name = action_space.__class__.__name__.lower()
        if hasattr(action_space, "nvec") or "multidiscrete" in name:
            return "multidiscrete", [int(v) for v in action_space.nvec]
        if hasattr(action_space, "n") or name == "discrete":
            return "discrete", int(action_space.n)
        if hasattr(action_space, "shape") or name == "box":
            return "continuous", int(action_space.shape[0])

    if isinstance(action_dim, Sequence) and not isinstance(action_dim, (str, bytes)):
        return "multidiscrete", [int(v) for v in action_dim]
    return "continuous", int(action_dim)


class ACTLayer(nn.Module):
    """Create discrete, continuous, or MultiDiscrete action distributions."""

    def __init__(
        self,
        input_dim: int,
        action_space=None,
        action_dim=None,
        action_type: Optional[str] = None,
        gain: float = 0.01,
    ) -> None:
        super().__init__()
        self.action_type, self.action_dim = infer_action_spec(action_space, action_dim, action_type)

        if self.action_type == "discrete":
            self.dist = CategoricalLayer(input_dim, self.action_dim, gain=gain)
        elif self.action_type == "continuous":
            self.dist = DiagGaussianLayer(input_dim, self.action_dim, gain=gain)
        elif self.action_type == "multidiscrete":
            self.dist = MultiCategoricalLayer(input_dim, self.action_dim, gain=gain)
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}")

    def get_dist(self, features, available_actions=None):
        return self.dist(features, available_actions=available_actions)

    def forward(self, features, deterministic: bool = False, available_actions=None):
        dist = self.get_dist(features, available_actions)
        actions = dist.mode() if deterministic else dist.sample()
        action_log_probs = dist.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, features, actions, available_actions=None):
        dist = self.get_dist(features, available_actions)
        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy()
        return action_log_probs, dist_entropy
