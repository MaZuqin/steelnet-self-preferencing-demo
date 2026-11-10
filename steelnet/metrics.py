"""Paper-facing metrics for total-order and beta mechanism analysis."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return (
        float(numerator / denominator)
        if abs(float(denominator)) > 1e-12
        else float(default)
    )


def bullwhip_ratio(order: np.ndarray, demand: np.ndarray) -> float:
    order = np.asarray(order, dtype=float)
    demand = np.asarray(demand, dtype=float)
    if order.size < 3 or demand.size < 3:
        return float("nan")
    demand_var = float(np.var(demand, ddof=1))
    if not np.isfinite(demand_var) or demand_var <= 1e-12:
        return float("nan")
    return float(np.var(order, ddof=1) / demand_var)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return float("nan")
    left = left[mask]
    right = right[mask]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def aggregate_episode_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Aggregate one episode into efficiency, routing, and coupling metrics."""

    rows = list(rows)
    if not rows:
        return {}

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=float)

    demand = values("demand")
    total_order = values("platform_order")
    le_order = values("platform_order_le")
    sme_order = values("platform_order_sme")
    # Manuscript Eq. (8): upstream replenishment variance relative to the
    # orders received by that supplier, not relative to customer demand.
    # Missing upstream actions make this metric unavailable, rather than zero.
    le_replenishment = np.asarray(
        [float(row.get("replenishment_le", float("nan"))) for row in rows], dtype=float
    )
    sme_replenishment = np.asarray(
        [float(row.get("replenishment_sme", float("nan"))) for row in rows], dtype=float
    )
    beta = values("beta_S")
    effective_beta = values("effective_beta_S")
    le_shipment = values("shipment_le")
    sme_shipment = values("shipment_sme")
    total_shipment = le_shipment + sme_shipment
    total_order_sum = float(np.sum(total_order))
    sme_order_sum = float(np.sum(sme_order))
    ending_platform_backlog = float(rows[-1].get("backlog_platform", 0.0))
    ending_sme_backlog = float(rows[-1].get("backlog_sme", 0.0))
    demand_sum = float(np.sum(demand))

    metrics = {
        "n_steps": float(len(rows)),
        "total_cost": float(np.sum(values("system_cost"))),
        "mean_system_cost": float(np.mean(values("system_cost"))),
        "mean_platform_backlog": float(np.mean(values("backlog_platform"))),
        "ending_platform_backlog": ending_platform_backlog,
        "cumulative_service_level": float(
            np.clip(
                safe_div(demand_sum - ending_platform_backlog, demand_sum), 0.0, 1.0
            )
        ),
        "BW_total": bullwhip_ratio(total_order, demand),
        "BW_LE": bullwhip_ratio(le_replenishment, le_order),
        "BW_SME": bullwhip_ratio(sme_replenishment, sme_order),
        "allocation_to_customer_variance_ratio_LE": bullwhip_ratio(le_order, demand),
        "allocation_to_customer_variance_ratio_SME": bullwhip_ratio(sme_order, demand),
        "beta_mean": float(np.mean(beta)),
        "beta_std": float(np.std(beta, ddof=1)) if beta.size > 1 else 0.0,
        "beta_S_mean": float(np.mean(beta)),
        "beta_S_std": float(np.std(beta, ddof=1)) if beta.size > 1 else 0.0,
        "routing_volatility": (
            float(np.mean(np.abs(np.diff(beta)))) if beta.size > 1 else 0.0
        ),
        "aggregate_sme_order_share": safe_div(sme_order_sum, total_order_sum),
        "order_weighted_beta": safe_div(
            float(np.sum(beta * total_order)), total_order_sum
        ),
        "order_weighted_beta_S": safe_div(
            float(np.sum(beta * total_order)), total_order_sum
        ),
        "effective_beta_mean": float(np.mean(effective_beta)),
        "effective_beta_S_mean": float(np.mean(effective_beta)),
        "beta_order_coupling_mae": float(
            np.mean(np.abs(sme_order - beta * total_order))
        ),
        "beta_order_coupling_max_error": float(
            np.max(np.abs(sme_order - beta * total_order))
        ),
        "beta_S_order_coupling_mae": float(
            np.mean(np.abs(sme_order - beta * total_order))
        ),
        "beta_S_order_coupling_max_error": float(
            np.max(np.abs(sme_order - beta * total_order))
        ),
        "allocation_balance_max_error": float(
            np.max(np.abs(le_order + sme_order - total_order))
        ),
        "QS3_realized_fulfillment_share": safe_div(
            float(np.sum(sme_shipment)), float(np.sum(total_shipment))
        ),
        "QS3_legacy_ship_per_total_order": safe_div(
            float(np.sum(sme_shipment)), total_order_sum
        ),
        "FR3_sme_order_fulfillment_rate": float(
            np.clip(
                safe_div(sme_order_sum - ending_sme_backlog, sme_order_sum), 0.0, 1.0
            )
        ),
        "corr_total_order_demand": safe_corr(total_order, demand),
        "corr_beta_demand": safe_corr(beta, demand),
        "corr_beta_total_order": safe_corr(beta, total_order),
        "corr_beta_S_demand": safe_corr(beta, demand),
        "corr_beta_S_total_order": safe_corr(beta, total_order),
    }
    metrics.update(
        {
            "C_total_episode": metrics["total_cost"],
            "BW1": metrics["BW_LE"],
            "BW3": metrics["BW_SME"],
            "QS3": metrics["QS3_realized_fulfillment_share"],
            "FR3": metrics["FR3_sme_order_fulfillment_rate"],
        }
    )
    return metrics


def build_jump_event_rows(
    rows: Iterable[dict[str, Any]],
    *,
    window: int = 3,
    positive_quantile: float = 0.95,
    negative_quantile: float = 0.05,
) -> list[dict[str, Any]]:
    """Align demand-jump windows for O_t, beta, and beta*O_t analysis."""

    rows = list(rows)
    if len(rows) < 2 * int(window) + 2:
        return []
    demand = np.asarray([float(row["demand"]) for row in rows], dtype=float)
    previous = demand[:-1]
    rates = (demand[1:] - previous) / (np.abs(previous) + 1e-8)
    finite = rates[np.isfinite(rates)]
    if finite.size == 0:
        return []
    low = float(np.quantile(finite, negative_quantile))
    high = float(np.quantile(finite, positive_quantile))

    event_rows: list[dict[str, Any]] = []
    for rate_index, rate in enumerate(rates):
        event_step = rate_index + 1
        jump_type = "positive" if rate >= high else ("negative" if rate <= low else "")
        if not jump_type:
            continue
        if event_step - window < 0 or event_step + window >= len(rows):
            continue
        event_id = f"{jump_type}-t{event_step}"
        for rel_step in range(-window, window + 1):
            source = rows[event_step + rel_step]
            event_rows.append(
                {
                    "event_id": event_id,
                    "jump_type": jump_type,
                    "event_step": int(event_step),
                    "rel_step": int(rel_step),
                    "jump_rate": float(rate),
                    "demand": float(source["demand"]),
                    "platform_order": float(source["platform_order"]),
                    "beta_sme": float(source["beta_S"]),
                    "beta_S": float(source["beta_S"]),
                    "platform_order_sme": float(source["platform_order_sme"]),
                    "beta_times_total_order": float(
                        source["beta_S"] * source["platform_order"]
                    ),
                    "beta_S_times_total_order": float(
                        source["beta_S"] * source["platform_order"]
                    ),
                }
            )
    return event_rows


def summarize_metric_rows(metric_rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Aggregate episode metrics across evaluation demand paths."""

    rows = list(metric_rows)
    if not rows:
        return {}
    result: dict[str, float] = {"n_episodes": float(len(rows))}
    metadata = {
        "episode",
        "scenario",
        "method",
        "run_id",
        "seed",
        "checkpoint",
        "reward_alpha",
    }
    keys = sorted({key for row in rows for key in row if key not in metadata})
    for key in keys:
        vals = []
        for row in rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                vals.append(value)
        if vals:
            result[f"{key}_mean"] = float(np.mean(vals))
            result[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return result


def summarize_jump_event_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize event-aligned demand, O_t, beta, and beta*O_t trajectories."""

    rows = list(rows)
    if not rows:
        return []
    group_keys = ("scenario", "method", "jump_type", "rel_step")
    metric_keys = (
        "demand",
        "platform_order",
        "beta_S",
        "platform_order_sme",
        "beta_S_times_total_order",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        result = dict(zip(group_keys, key))
        result["n_events"] = len(
            {(item["episode"], item["event_id"]) for item in group}
        )
        for metric in metric_keys:
            values = np.asarray([float(item[metric]) for item in group], dtype=float)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        summaries.append(result)
    return summaries
