"""Discounted manuscript objective and inventory/backlog cost decomposition.

This module extends the existing episode aggregator without changing archived
metric definitions. Input rows are period records from ``_step_record`` and
must belong to one episode, ordered from decision period zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from steelnet.metrics import aggregate_episode_metrics

NODES = ("platform", "le", "sme")


def _coefficients(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(
            f"{name} must contain three finite nonnegative coefficients in platform, LE, SME order"
        )
    return result


def aggregate(
    rows: Iterable[dict[str, Any]],
    gamma: float = 0.99,
    *,
    holding_cost: Sequence[float] = (100.0, 20.0, 20.0),
    backlog_cost: Sequence[float] = (300.0, 200.0, 200.0),
) -> dict[str, float]:
    """Return original metrics plus discounted costs and their decomposition.

    ``gamma**0`` weights the first period. If ``t`` is present, it must equal
    the zero-based row index. Cost coefficients follow platform, LE, SME order.
    ``*_cost_<node>`` and ``discounted_*_cost_<node>`` are episode totals;
    ``holding_cost``/``backlog_cost`` also sum over all three nodes. The recorded
    system cost must match the inventory and backlog decomposition each period.
    """
    rows = list(rows)
    if not rows:
        return {}
    gamma = float(gamma)
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and between zero and one")
    holding = _coefficients(holding_cost, "holding_cost")
    backlog = _coefficients(backlog_cost, "backlog_cost")
    for index, row in enumerate(rows):
        if "t" in row and float(row["t"]) != index:
            raise ValueError(
                "Episode rows must be ordered by consecutive periods starting at t=0"
            )
    inventory = np.asarray(
        [[float(row[f"inventory_{node}"]) for node in NODES] for row in rows]
    )
    pending = np.asarray(
        [[float(row[f"backlog_{node}"]) for node in NODES] for row in rows]
    )
    recorded_cost = np.asarray([float(row["system_cost"]) for row in rows])
    if not all(
        np.isfinite(values).all() for values in (inventory, pending, recorded_cost)
    ):
        raise ValueError("Inventory, backlog and system costs must be finite")
    holding_by_period = inventory * holding
    backlog_by_period = pending * backlog
    total_by_period = holding_by_period + backlog_by_period
    decomposed = total_by_period.sum(axis=1)
    if not np.allclose(decomposed, recorded_cost, rtol=1e-9, atol=1e-6):
        difference = np.abs(decomposed - recorded_cost)
        period = int(np.argmax(difference))
        raise AssertionError(
            f"Inventory/backlog cost does not match system_cost at t={period}: "
            f"{decomposed[period]} vs {recorded_cost[period]}; check cost coefficients and purchase-cost inclusion"
        )

    weights = np.power(gamma, np.arange(len(rows), dtype=np.float64))
    metrics = aggregate_episode_metrics(rows)
    metrics["discount_gamma"] = gamma
    metrics["discounted_total_cost"] = float(np.dot(weights, recorded_cost))
    for component, values in (
        ("holding_cost", holding_by_period),
        ("backlog_cost", backlog_by_period),
        ("total_cost", total_by_period),
    ):
        ordinary = values.sum(axis=0)
        discounted = weights @ values
        for index, node in enumerate(NODES):
            metrics[f"{component}_{node}"] = float(ordinary[index])
            metrics[f"discounted_{component}_{node}"] = float(discounted[index])
        # Preserve the original total_cost and the directly discounted system
        # objective instead of replacing them with a different reduction order.
        if component != "total_cost":
            metrics[component] = float(ordinary.sum())
            metrics[f"discounted_{component}"] = float(discounted.sum())
    return metrics
