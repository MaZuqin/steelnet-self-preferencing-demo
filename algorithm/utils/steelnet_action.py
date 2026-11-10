"""Action adapters for ``envs.steelnet_env.Env``.

The environment already owns the final action decoding:
``Env.action_map`` rounds order quantities to non-negative integers and clips
``beta_sme`` to ``[0, 1]``. This module only formats policy outputs into the
three-agent structure expected by ``env.step(..., one_hot=False)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import numpy as np
import torch

from envs.state import (
    ACTION_DIMS_BY_NAME,
    AGENT_NAMES as ENV_AGENT_NAMES,
    DEFAULT_BETA_SME,
    LE,
    PLATFORM as PLATFORM_INDEX,
    SME,
)

AGENT_NAMES = tuple(ENV_AGENT_NAMES)
PLATFORM = AGENT_NAMES[PLATFORM_INDEX]
LE_SUPPLIER = AGENT_NAMES[LE]
SME_SUPPLIER = AGENT_NAMES[SME]
ACTION_DIMS_BY_ROLE = dict(ACTION_DIMS_BY_NAME)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flat(value) -> np.ndarray:
    return np.asarray(_to_numpy(value), dtype=np.float32).reshape(-1)


def prepare_platform_action(action, default_beta_sme: float = DEFAULT_BETA_SME):
    """Format platform action as ``[O_t, beta_sme]``.

    ``O_t`` remains continuous here because ``steelnet_env`` is responsible for
    converting it to a non-negative integer. ``beta_sme`` remains continuous.
    """
    arr = _flat(action)
    order = arr[0] if arr.size >= 1 else 0.0
    beta_sme = arr[1] if arr.size >= 2 else default_beta_sme
    return [float(order), float(beta_sme)]


def prepare_supplier_action(action):
    """Format LE/SME action as ``[replenishment]`` for ``steelnet_env``."""
    arr = _flat(action)
    replenishment = arr[0] if arr.size >= 1 else 0.0
    return [float(replenishment)]


def prepare_steelnet_actions(actions, default_beta_sme: float = DEFAULT_BETA_SME):
    """Format actor outputs for ``Env.step(actions, one_hot=False)``.

    Accepted inputs:
    - mapping with keys ``platform``, ``le_supplier``, ``sme_supplier``;
    - sequence ``[platform_action, le_action, sme_action]``.
    """
    if isinstance(actions, Mapping):
        platform_action = actions.get(PLATFORM, actions.get(0, [0.0, default_beta_sme]))
        le_action = actions.get(LE_SUPPLIER, actions.get(1, [0.0]))
        sme_action = actions.get(SME_SUPPLIER, actions.get(2, [0.0]))
    else:
        if len(actions) != 3:
            raise ValueError("SteelNet actions must contain platform, LE, and SME actions.")
        platform_action, le_action, sme_action = actions

    return [
        prepare_platform_action(platform_action, default_beta_sme=default_beta_sme),
        prepare_supplier_action(le_action),
        prepare_supplier_action(sme_action),
    ]


def stack_agent_actions(
    platform_actions,
    le_actions,
    sme_actions,
    default_beta_sme: float = DEFAULT_BETA_SME,
) -> list:
    """Build batched environment actions from three policy output arrays."""
    platform_arr = _to_numpy(platform_actions)
    le_arr = _to_numpy(le_actions)
    sme_arr = _to_numpy(sme_actions)

    batch_size = int(np.asarray(platform_arr).shape[0]) if np.asarray(platform_arr).ndim > 1 else 1
    if batch_size == 1:
        return prepare_steelnet_actions(
            [platform_arr, le_arr, sme_arr],
            default_beta_sme=default_beta_sme,
        )

    env_actions = []
    for idx in range(batch_size):
        env_actions.append(
            prepare_steelnet_actions(
                [platform_arr[idx], le_arr[idx], sme_arr[idx]],
                default_beta_sme=default_beta_sme,
            )
        )
    return env_actions
