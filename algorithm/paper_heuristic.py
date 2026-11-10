"""Causal base-stock implementation of Appendix C.3.

The manuscript specifies the target-stock formula, capped order and gap-based
routing, but leaves the numerical parameters and inventory-position conventions
open. Defaults here are explicit demo assumptions, not recovered paper values.
They are fixed before comparing policies: window 14, z=1.645, fallback SME share
0.5 and no minimum SME quota. Neither future demand nor private SME inventory
or backlog is used by the platform decision.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from envs.state import LE, PLATFORM, SME


@dataclass
class PaperBaseStockHeuristic:
    """Periodic-review base-stock rule with observable routing proxies.

    Review periods include one decision interval plus replenishment lead time.
    Platform replenishment traverses both links of a branch; supplier
    replenishment traverses only its manufacturer link. Explicit overrides
    permit a different documented convention without changing the formulas.

    For routing, desired branch stock uses the fallback share of the platform
    forecast over one review interval plus supplier-to-platform lead time.
    LE ready stock is observed. SME ready stock is estimated from recent actual
    shipments over its manufacturing review period, less outstanding orders
    reconstructed from past allocations and shipments. This is a supply proxy,
    not an assertion that the SME's unobserved on-hand stock is known.
    """

    platform_order_max: float = 220.0
    supplier_order_max: float = 220.0
    forecast_window: int = 14
    safety_z: float = 1.645
    beta_fallback: float = 0.5
    sme_min_share: float = 0.0
    platform_review_period: float | None = None
    le_review_period: float | None = None
    sme_review_period: float | None = None

    def __post_init__(self):
        if not isinstance(self.forecast_window, int) or self.forecast_window < 1:
            raise ValueError("forecast_window must be a positive integer")
        for name in ("platform_order_max", "supplier_order_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.safety_z) or self.safety_z < 0:
            raise ValueError("safety_z must be nonnegative and finite")
        for name in ("beta_fallback", "sme_min_share"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("platform_review_period", "le_review_period", "sme_review_period"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite when set")
        self.reset()

    def reset(self):
        """Clear all causal history at an episode boundary."""
        self._demand_history = deque(maxlen=self.forecast_window)
        self._allocation_history = {j: deque(maxlen=self.forecast_window) for j in (LE, SME)}
        self._sme_shipment_history = deque(maxlen=self.forecast_window)
        self._sme_outstanding = 0.0
        self._env = None
        self._last_step = -1
        self.last_diagnostics = {}

    @staticmethod
    def _moments(history):
        if not history:
            return 0.0, 0.0
        values = np.asarray(history, dtype=float)
        return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

    def _observe(self, env):
        step = int(env.step_num)
        if self._env is not env or step < self._last_step:
            self.reset()
            self._env = env
        if step == self._last_step:
            return  # Repeated requests at the same decision do not duplicate data.
        if step > 0:
            for j in (LE, SME):
                self._allocation_history[j].append(float(env.last_platform_orders[j]))
            shipped = float(env.last_shipments[SME])
            self._sme_shipment_history.append(shipped)
            self._sme_outstanding = max(
                0.0, self._sme_outstanding + float(env.last_platform_orders[SME]) - shipped
            )
        self._demand_history.append(float(env._get_demand(step)))
        self._last_step = step

    def _supplier_review(self, env, supplier):
        override = self.le_review_period if supplier == LE else self.sme_review_period
        return float(override if override is not None else 1 + env.m_to_supplier_lead_times[supplier])

    def _target(self, mean, std, review):
        return max(0.0, mean * review + self.safety_z * std * math.sqrt(review))

    def _allocation_weight(self, env, mean, std):
        """Platform routing reads no SME private inventory/backlog field."""
        fallback = max(self.beta_fallback, self.sme_min_share)
        shares = {LE: 1.0 - fallback, SME: fallback}
        target = {
            j: self._target(shares[j] * mean, shares[j] * std,
                            1.0 + env.supplier_to_platform_lead_times[j])
            for j in (LE, SME)
        }
        le_inbound = env.m_to_supplier[LE]
        le_due = float(le_inbound[0]) if len(le_inbound) else 0.0
        le_ready = max(0.0, float(env.inventory[LE]) + le_due - float(env.backlog[LE]))
        sme_ship_rate, _ = self._moments(self._sme_shipment_history)
        sme_ready_proxy = max(
            0.0, sme_ship_rate * self._supplier_review(env, SME) - self._sme_outstanding
        )
        position = {
            LE: float(sum(env.supplier_to_platform[LE])) + le_ready,
            SME: float(sum(env.supplier_to_platform[SME])) + sme_ready_proxy,
        }
        gaps = {j: max(0.0, target[j] - position[j]) for j in (LE, SME)}
        total_gap = gaps[LE] + gaps[SME]
        beta = gaps[SME] / total_gap if total_gap > 1e-12 else fallback
        beta = min(1.0, max(self.sme_min_share, beta))
        self.last_diagnostics.update(
            routing_gap_le=gaps[LE], routing_gap_sme=gaps[SME],
            sme_ready_proxy=sme_ready_proxy, sme_outstanding_proxy=self._sme_outstanding,
        )
        return beta

    def act_from_env(self, env):
        self._observe(env)
        mean, std = self._moments(self._demand_history)
        beta = self._allocation_weight(env, mean, std)
        full_lead = {
            j: env.m_to_supplier_lead_times[j] + env.supplier_to_platform_lead_times[j]
            for j in (LE, SME)
        }
        review = float(self.platform_review_period) if self.platform_review_period is not None else (
            1.0 + (1.0 - beta) * full_lead[LE] + beta * full_lead[SME]
        )
        platform_ip = (
            float(env.inventory[PLATFORM]) - float(env.backlog[PLATFORM])
            + sum(env.supplier_to_platform[LE]) + sum(env.supplier_to_platform[SME])
        )
        order = float(np.clip(self._target(mean, std, review) - platform_ip, 0, self.platform_order_max))
        replenishments = {}
        for j in (LE, SME):
            # Each supplier can use its own state, but only past allocated orders.
            branch_mean, branch_std = self._moments(self._allocation_history[j])
            supplier_ip = float(env.inventory[j]) - float(env.backlog[j]) + sum(env.m_to_supplier[j])
            supplier_review = self._supplier_review(env, j)
            replenishments[j] = float(np.clip(
                self._target(branch_mean, branch_std, supplier_review) - supplier_ip,
                0, self.supplier_order_max,
            ))
        self.last_diagnostics.update(
            forecast_mean=mean, forecast_std=std, platform_inventory_position=float(platform_ip),
            platform_review_period=review, beta_sme=beta,
        )
        # The environment performs all rounding once, using manuscript Eq. (2).
        return [[order, beta], [replenishments[LE]], [replenishments[SME]]]

    def act(self, obs, env=None):
        if env is None:
            raise ValueError("PaperBaseStockHeuristic.act requires env")
        return self.act_from_env(env)
