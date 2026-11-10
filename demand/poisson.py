"""Independent stationary Poisson paths for the manuscript-facing demo.

The 20 path rates are a demo assumption: linspace(92, 132, 20). The manuscript
does not specify the Poisson rates. This generator does not replace the
archived stationary generator or alter its samples.
"""
from __future__ import annotations

import operator

import numpy as np


PATH_LAMBDAS = tuple(float(value) for value in np.linspace(92.0, 132.0, 20))
RATE_PROVENANCE = "Demo assumption: 20 fixed rates linspace(92, 132, 20); manuscript rates unspecified."


def generate(n_paths: int = 20, horizon: int = 366, seed: int = 2026) -> np.ndarray:
    """Return the first ``n_paths`` of 20 reproducible Poisson paths.

    Each path has its own RNG stream and its fixed grid rate. Requesting fewer
    paths or fewer periods preserves the corresponding prefix of a larger
    request with the same seed. Values are integer counts stored as float32,
    matching the other demand loaders.
    """
    n_paths = operator.index(n_paths)
    horizon = operator.index(horizon)
    if not 1 <= n_paths <= len(PATH_LAMBDAS):
        raise ValueError("n_paths must be between 1 and 20 for the fixed rate grid")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    streams = np.random.SeedSequence(seed).spawn(n_paths)
    return np.stack([
        np.random.default_rng(stream).poisson(lam=PATH_LAMBDAS[index], size=horizon)
        for index, stream in enumerate(streams)
    ]).astype(np.float32)
