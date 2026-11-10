"""Factory helpers for SteelNet HAPPO/MAPPO policies.

The SteelNet action interface is fixed as:
- platform: ``[O_t^P, beta_t^S]``;
- LE supplier: ``[O_t^L]``;
- SME supplier: ``[O_t^S]``.

Order quantities are emitted as continuous policy samples and converted to
non-negative integers by ``envs.steelnet_env.Env.action_map``. ``beta_t^S``
remains continuous.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .happo_policy import HAPPOPolicy
from .mappo_policy import MAPPOPolicy
from .utils.steelnet_action import ACTION_DIMS_BY_ROLE, AGENT_NAMES

DimSpec = Union[int, Sequence[int]]

STEELNET_ACTION_DIMS = dict(ACTION_DIMS_BY_ROLE)


def _flat_dim(value) -> int:
    return int(np.asarray(value, dtype=np.float32).reshape(-1).shape[0])


def _as_role_mapping(value, label: str) -> Dict[str, DimSpec]:
    if isinstance(value, Mapping):
        return {name: value[name] for name in AGENT_NAMES}

    values = list(value)
    if len(values) != len(AGENT_NAMES):
        raise ValueError(f"{label} must provide dimensions for platform, LE, and SME.")
    return dict(zip(AGENT_NAMES, values))


def infer_steelnet_dims(env) -> Tuple[Dict[str, int], int]:
    """Infer local observation and centralized-state dimensions from an env."""
    if hasattr(env, "get_obs"):
        obs = env.get_obs()
    elif hasattr(env, "get_reset_obs"):
        normalize = getattr(env, "normalize", getattr(env, "default_normalize", None))
        obs = env.get_reset_obs(normalize)
    else:
        obs = env._build_obs(getattr(env, "normalize", None))

    if len(obs) != len(AGENT_NAMES):
        raise ValueError("SteelNet env must return three local observations.")

    obs_dims = {name: _flat_dim(obs[idx]) for idx, name in enumerate(AGENT_NAMES)}
    cent_obs_dim = _flat_dim(env.global_state())
    return obs_dims, cent_obs_dim


def build_steelnet_policies(
    obs_dims: Optional[Union[Mapping[str, DimSpec], Sequence[DimSpec]]] = None,
    cent_obs_dim: Optional[DimSpec] = None,
    env=None,
    algorithm: str = "happo",
    hidden_sizes: Sequence[int] = (128, 128),
    recurrent_hidden_size: Optional[int] = None,
    recurrent_layers: int = 1,
    lr: float = 3e-4,
    critic_lr: float = 3e-4,
    optimizer_eps: float = 1e-5,
    weight_decay: float = 0.0,
    device="cpu",
    **network_kwargs,
) -> Dict[str, HAPPOPolicy]:
    """Create GRU-enabled policies for platform, LE supplier, and SME supplier.

    Pass either ``env`` or explicit ``obs_dims`` and ``cent_obs_dim``. The
    returned mapping follows ``("platform", "le_supplier", "sme_supplier")``.
    """
    if env is not None:
        inferred_obs_dims, inferred_cent_obs_dim = infer_steelnet_dims(env)
        obs_dims = inferred_obs_dims if obs_dims is None else obs_dims
        cent_obs_dim = inferred_cent_obs_dim if cent_obs_dim is None else cent_obs_dim

    if obs_dims is None or cent_obs_dim is None:
        raise ValueError("Pass env, or provide both obs_dims and cent_obs_dim.")

    role_obs_dims = _as_role_mapping(obs_dims, "obs_dims")
    kind = algorithm.lower()
    if kind == "happo":
        policy_cls = HAPPOPolicy
    elif kind == "mappo":
        policy_cls = MAPPOPolicy
    else:
        raise ValueError("algorithm must be 'happo' or 'mappo'.")

    network_kwargs = dict(network_kwargs)
    network_kwargs["use_recurrent"] = True
    network_kwargs["recurrent_hidden_size"] = recurrent_hidden_size
    network_kwargs["recurrent_layers"] = recurrent_layers

    policies = {}
    for role in AGENT_NAMES:
        policies[role] = policy_cls(
            obs_dim=role_obs_dims[role],
            cent_obs_dim=cent_obs_dim,
            action_dim=STEELNET_ACTION_DIMS[role],
            action_type="continuous",
            hidden_sizes=hidden_sizes,
            lr=lr,
            critic_lr=critic_lr,
            optimizer_eps=optimizer_eps,
            weight_decay=weight_decay,
            device=device,
            **network_kwargs,
        )
    return policies


def build_happo_steelnet_policies(*args, **kwargs) -> Dict[str, HAPPOPolicy]:
    """Shortcut for GRU-enabled SteelNet HAPPO policies."""
    kwargs["algorithm"] = "happo"
    return build_steelnet_policies(*args, **kwargs)


def build_mappo_steelnet_policies(*args, **kwargs) -> Dict[str, MAPPOPolicy]:
    """Shortcut for GRU-enabled SteelNet MAPPO policies."""
    kwargs["algorithm"] = "mappo"
    return build_steelnet_policies(*args, **kwargs)
