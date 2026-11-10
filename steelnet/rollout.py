"""Final policy adapters and unchanged environment step records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from algorithm.policy_factory import build_steelnet_policies
from envs.state import AGENT_NAMES, LE, PLATFORM, SME
from envs.steelnet_env import Env


class ActionProvider(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(
        self, observations: Sequence[np.ndarray], env: Env
    ) -> list[list[float]]: ...


def make_env(
    *,
    horizon: int = 366,
    lead_time: Sequence[int] = (1, 2, 0),
    initial_inventory: Sequence[float] = (0.0, 0.0, 0.0),
    holding_cost: Sequence[float] = (100.0, 20.0, 20.0),
    backlog_cost: Sequence[float] = (300.0, 200.0, 200.0),
    purchase_price: Sequence[float] = (0.0, 12.0, 10.8),
    purchase_discount: Any = None,
    purchase_discount_tiers: Any = None,
    include_purchase_cost_in_reward: bool = False,
    alpha: float = 0.0,
    random_shipping_loss: bool = False,
    branch_lost_rate: Sequence[float] = (0.05, 0.10),
    transport_lost_rate: Any = None,
    lost_rate: float | None = None,
    integer_shipping_loss: bool = True,
    obs_scale: float = 1000.0,
    paper_observation: bool = False,
) -> Env:
    return Env(
        episode_len=int(horizon),
        lead_time=list(lead_time),
        initial_inventory=list(initial_inventory),
        holding_cost=list(holding_cost),
        backlog_cost=list(backlog_cost),
        purchase_price=list(purchase_price),
        purchase_discount=purchase_discount,
        purchase_discount_tiers=purchase_discount_tiers,
        include_purchase_cost_in_reward=include_purchase_cost_in_reward,
        alpha=float(alpha),
        random_shipping_loss=random_shipping_loss,
        branch_lost_rate=list(branch_lost_rate),
        transport_lost_rate=transport_lost_rate,
        lost_rate=lost_rate,
        integer_shipping_loss=integer_shipping_loss,
        normalize_obs=True,
        obs_scale=float(obs_scale),
        paper_observation=paper_observation,
    )


@dataclass
class HeuristicAdapter:
    heuristic: Any
    name: str = "base_stock"

    def reset(self) -> None:
        if hasattr(self.heuristic, "reset"):
            self.heuristic.reset()
        if hasattr(self.heuristic, "_last_beta"):
            self.heuristic._last_beta = None

    def act(self, observations: Sequence[np.ndarray], env: Env) -> list[list[float]]:
        return self.heuristic.act(observations, env=env)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


@dataclass
class LearnedPolicyAdapter:
    """Recurrent HAPPO/MAPPO adapter using the paper's [O_t, beta_S] action."""

    policies: Mapping[str, Any]
    order_max: float
    supplier_order_max: float
    hidden_size: int
    name: str
    deterministic: bool = True

    def __post_init__(self) -> None:
        self._rnn_states: list[np.ndarray] = []
        self.reset()

    def reset(self) -> None:
        self._rnn_states = [
            np.zeros((1, int(self.hidden_size)), dtype=np.float32) for _ in AGENT_NAMES
        ]

    def act(self, observations: Sequence[np.ndarray], env: Env) -> list[list[float]]:
        latent: list[np.ndarray] = []
        masks = np.ones((1, 1), dtype=np.float32)
        for index, role in enumerate(AGENT_NAMES):
            policy = self.policies[role]
            policy.prep_rollout()
            with torch.no_grad():
                action, next_rnn = policy.act(
                    np.asarray(observations[index], dtype=np.float32).reshape(1, -1),
                    self._rnn_states[index],
                    masks,
                    deterministic=bool(self.deterministic),
                )
            latent.append(action.detach().cpu().numpy().reshape(-1))
            self._rnn_states[index] = next_rnn.detach().cpu().numpy()

        platform = _sigmoid(latent[PLATFORM])
        le = _sigmoid(latent[LE])
        sme = _sigmoid(latent[SME])
        total_order = float(platform[0] * self.order_max)
        beta_sme = float(platform[1]) if platform.size >= 2 else 0.5
        return [
            [total_order, beta_sme],
            [float(le[0] * self.supplier_order_max)],
            [float(sme[0] * self.supplier_order_max)],
        ]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        env: Env,
        algorithm: str,
        order_max: float,
        supplier_order_max: float,
        hidden_size: int = 32,
        device: str = "cpu",
    ) -> "LearnedPolicyAdapter":
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=device, weights_only=True
        )
        train_row = dict(checkpoint.get("train_row", {}) or {})
        action_mode = str(train_row.get("platform_action_mode", "unknown"))
        if action_mode != "order_beta":
            raise ValueError(
                f"checkpoint action mode is {action_mode!r}; Exp2 requires 'order_beta'"
            )
        env.reset(
            train=False,
            normalize=True,
            demand_list=[0.0 for _ in range(int(env.episode_max_steps))],
        )
        policies = build_steelnet_policies(
            env=env,
            algorithm=algorithm,
            hidden_sizes=(int(hidden_size), int(hidden_size)),
            recurrent_hidden_size=int(hidden_size),
            recurrent_layers=1,
            device=device,
            use_layer_norm=True,
        )
        states = checkpoint.get("policies", checkpoint)
        for role, policy in policies.items():
            role_state = states[role]
            policy.actor.load_state_dict(role_state.get("actor", role_state))
            if "critic" in role_state:
                if checkpoint.get("clock_critic"):
                    from algorithm.time_critic import TimeAwareCritic

                    clock = checkpoint["clock_critic"]
                    policy.critic = TimeAwareCritic(
                        policy.critic,
                        horizon=clock["horizon"],
                        gamma=clock["gamma"],
                        state_skip=clock.get("state_time_skip", False),
                    )
                    policy.critic_optimizer = torch.optim.Adam(
                        policy.critic.parameters(), lr=3e-4, eps=1e-5
                    )
                policy.critic.load_state_dict(role_state["critic"])
        return cls(
            policies=policies,
            order_max=float(order_max),
            supplier_order_max=float(supplier_order_max),
            hidden_size=int(hidden_size),
            name=str(algorithm).lower(),
        )


def _step_record(
    *,
    scenario: str,
    method: str,
    episode: int,
    info: dict[str, Any],
    env: Env,
) -> dict[str, Any]:
    platform_orders = info["platform_orders"]
    shipments = info["shipments_to_platform"]
    purchase = dict(info.get("purchase", {}) or {})
    shipping_loss = dict(info.get("shipping_loss", {}) or {})
    shipping_lost_qty = float(
        sum(float((item or {}).get("lost_qty", 0.0)) for item in shipping_loss.values())
    )
    platform_order = float(info["platform_order"])
    beta_sme = float(info["beta_sme"])
    platform_order_le = float(platform_orders[LE])
    platform_order_sme = float(platform_orders[SME])
    effective_beta_sme = (
        platform_order_sme / platform_order if platform_order > 1e-12 else 0.0
    )
    return {
        "scenario": scenario,
        "method": method,
        "episode": int(episode),
        "t": int(info["t"]),
        "demand": float(info["demand"]),
        "system_cost": float(info["system_cost"]),
        "purchase_order_net_amount": float(purchase.get("total_order_net_amount", 0.0)),
        "purchase_order_gross_amount": float(
            purchase.get("total_order_gross_amount", 0.0)
        ),
        "purchase_discount_saving": float(
            purchase.get("total_order_discount_saving", 0.0)
        ),
        "reward_includes_purchase_cost": bool(
            purchase.get("reward_includes_purchase_cost", False)
        ),
        "shipping_loss_enabled": bool(getattr(env, "random_shipping_loss", False)),
        "shipping_lost_qty": shipping_lost_qty,
        "random_shipping_loss": bool(getattr(env, "random_shipping_loss", False)),
        "platform_order": platform_order,
        "beta_sme": beta_sme,
        "beta_S": beta_sme,
        "effective_beta_sme": effective_beta_sme,
        "effective_beta_S": effective_beta_sme,
        "platform_order_le": platform_order_le,
        "platform_order_sme": platform_order_sme,
        "beta_S_times_total_order": beta_sme * platform_order,
        "shipment_le": float(shipments[LE]),
        "shipment_sme": float(shipments[SME]),
        "replenishment_le": float(info["replenishments"][LE]),
        "replenishment_sme": float(info["replenishments"][SME]),
        "customer_shipment": float(info["customer_shipment"]),
        "backlog_platform": float(env.backlog[PLATFORM]),
        "backlog_le": float(env.backlog[LE]),
        "backlog_sme": float(env.backlog[SME]),
        "inventory_platform": float(env.inventory[PLATFORM]),
        "inventory_le": float(env.inventory[LE]),
        "inventory_sme": float(env.inventory[SME]),
        "allocation_balance_error": platform_order_le
        + platform_order_sme
        - platform_order,
        "beta_order_rounding_error": platform_order_sme - beta_sme * platform_order,
    }


def run_episode(provider, demand, config, method, scenario, index):
    env = make_env(
        **{**config["environment"], "horizon": len(demand)},
        alpha=config["policy"]["alpha"].get(method, 0.0),
    )
    provider.reset()
    obs = env.reset(train=False, normalize=True, demand_list=list(demand))
    rows = []
    for _ in range(len(demand)):
        action = provider.act(obs, env)
        obs, _, done, info = env.step(action, one_hot=False)
        rows.append(
            _step_record(
                scenario=scenario, method=method, episode=index, info=info[0], env=env
            )
        )
        if all(done):
            break
    return rows
