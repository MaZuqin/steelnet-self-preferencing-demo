from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent / "data" / "stochastic_jump.npz"


def generate(n_paths: int = 20, horizon: int = 366, seed: int = 2027) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 30.0
    base = np.linspace(92.0, 132.0, n_paths, dtype=np.float64)
    x = np.zeros(n_paths, dtype=np.float64)
    v = np.full(n_paths, 0.020, dtype=np.float64)
    demand = np.zeros((n_paths, horizon), dtype=np.float32)
    jump_flag = np.zeros((n_paths, horizon), dtype=np.uint8)
    jump_size = np.zeros((n_paths, horizon), dtype=np.float32)

    for t in range(horizon):
        z1 = rng.standard_normal(n_paths)
        z2 = -0.35 * z1 + np.sqrt(1.0 - 0.35**2) * rng.standard_normal(n_paths)
        v += 2.2 * (0.030 - v) * dt + 0.22 * np.sqrt(np.maximum(v, 1e-8)) * np.sqrt(dt) * z2
        v = np.maximum(v, 1e-8)
        x += 1.7 * (0.0 - x) * dt + np.sqrt(v) * np.sqrt(dt) * z1
        season = 0.045 * np.sin(2.0 * np.pi * t / 52.0)
        value = base * np.exp(x + season + 0.03 * t / max(horizon - 1, 1))

        jumps = rng.random(n_paths) < 0.025
        multiplier = np.where(jumps, np.exp(rng.normal(0.22, 0.10, n_paths)), 1.0)
        jumped = np.maximum(value * multiplier, 1.0)
        demand[:, t] = jumped.astype(np.float32)
        jump_flag[:, t] = jumps.astype(np.uint8)
        jump_size[:, t] = (jumped - value).astype(np.float32)

    return demand, jump_flag, jump_size


def main() -> None:
    demand, jump_flag, jump_size = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        demand=demand,
        jump_flag=jump_flag,
        jump_size=jump_size,
        generator=np.array(["stochastic_jump_heston"]),
        seed=np.array([2027]),
        params_json=np.array([json.dumps({"type": "stochastic_jump", "model": "heston", "jump_prob": 0.025})]),
    )
    print(f"saved: {OUT}")
    print(f"shape={demand.shape}, jumps={int(jump_flag.sum())}, min={demand.min():.2f}, mean={demand.mean():.2f}, max={demand.max():.2f}")


if __name__ == "__main__":
    main()
