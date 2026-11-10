"""Policy interface layer for HAPPO."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch.optim import Adam

from .actor_critic import Actor, Critic


class HAPPOPolicy:
    """Wrap actor, critic, and optimizers behind a stable policy interface."""

    def __init__(
        self,
        obs_dim=None,
        cent_obs_dim=None,
        action_space=None,
        action_dim=None,
        action_type: Optional[str] = None,
        actor: Optional[Actor] = None,
        critic: Optional[Critic] = None,
        hidden_sizes: Sequence[int] = (128, 128),
        lr: float = 3e-4,
        critic_lr: float = 3e-4,
        optimizer_eps: float = 1e-5,
        weight_decay: float = 0.0,
        device: Union[str, torch.device] = "cpu",
        **network_kwargs,
    ) -> None:
        if actor is None:
            if obs_dim is None:
                raise ValueError("obs_dim is required when actor is not provided.")
            actor = Actor(
                obs_dim=obs_dim,
                action_space=action_space,
                action_dim=action_dim,
                action_type=action_type,
                hidden_sizes=hidden_sizes,
                device=device,
                **network_kwargs,
            )

        if critic is None:
            if cent_obs_dim is None:
                raise ValueError("cent_obs_dim is required when critic is not provided.")
            critic = Critic(
                cent_obs_dim=cent_obs_dim,
                hidden_sizes=hidden_sizes,
                device=device,
                **network_kwargs,
            )

        self.actor = actor
        self.critic = critic
        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=lr,
            eps=optimizer_eps,
            weight_decay=weight_decay,
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=critic_lr,
            eps=optimizer_eps,
            weight_decay=weight_decay,
        )

    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device

    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def prep_rollout(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def get_actions(
        self,
        obs,
        cent_obs,
        actor_rnn_states=None,
        critic_rnn_states=None,
        masks=None,
        deterministic: bool = False,
        available_actions=None,
    ):
        """Return values, sampled actions, log-probs, and next RNN states."""
        actions, action_log_probs, actor_rnn_states = self.actor(
            obs,
            rnn_states=actor_rnn_states,
            masks=masks,
            deterministic=deterministic,
            available_actions=available_actions,
        )
        values, critic_rnn_states = self.critic(
            cent_obs,
            rnn_states=critic_rnn_states,
            masks=masks,
        )
        return values, actions, action_log_probs, actor_rnn_states, critic_rnn_states

    def get_values(self, cent_obs, critic_rnn_states=None, masks=None):
        values, _ = self.critic(cent_obs, rnn_states=critic_rnn_states, masks=masks)
        return values

    def evaluate_actions(
        self,
        obs,
        cent_obs,
        actions,
        actor_rnn_states=None,
        critic_rnn_states=None,
        masks=None,
        available_actions=None,
    ):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            actions,
            rnn_states=actor_rnn_states,
            masks=masks,
            available_actions=available_actions,
        )
        values, _ = self.critic(
            cent_obs,
            rnn_states=critic_rnn_states,
            masks=masks,
        )
        return values, action_log_probs, dist_entropy

    def act(self, obs, actor_rnn_states=None, masks=None, deterministic: bool = True, available_actions=None):
        actions, _, actor_rnn_states = self.actor(
            obs,
            rnn_states=actor_rnn_states,
            masks=masks,
            deterministic=deterministic,
            available_actions=available_actions,
        )
        return actions, actor_rnn_states

    def lr_decay(self, episode: int, total_episodes: int, initial_lr: float, initial_critic_lr: Optional[float] = None):
        frac = max(0.0, 1.0 - episode / float(total_episodes))
        for group in self.actor_optimizer.param_groups:
            group["lr"] = initial_lr * frac
        for group in self.critic_optimizer.param_groups:
            group["lr"] = (initial_critic_lr if initial_critic_lr is not None else initial_lr) * frac
