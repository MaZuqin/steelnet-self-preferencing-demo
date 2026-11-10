"""Small tensor utilities used by the HAPPO/MAPPO implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def check(value: Any, device=None, dtype=None):
    """Convert numpy/list inputs to tensors and move tensors to device/dtype.

    ``None`` is returned unchanged so callers can pass optional RNN states,
    action masks, or available-action masks without special-casing them.
    """

    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor
