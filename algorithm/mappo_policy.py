"""Policy interface layer for MAPPO."""

from __future__ import annotations

import torch

from .happo_policy import HAPPOPolicy


class MAPPOPolicy(HAPPOPolicy):
    """MAPPO policy wrapper.

    The interface mirrors HAPPOPolicy. Parameter sharing can be done by using
    one MAPPOPolicy instance for homogeneous agents, or by calling sync_from()
    between role-specific policies after an update.
    """

    @torch.no_grad()
    def sync_from(self, other: "MAPPOPolicy", tau: float = 1.0) -> None:
        """Synchronize this policy from another policy.

        tau=1.0 copies parameters exactly; lower values perform Polyak updates.
        """
        for target, source in zip(self.actor.parameters(), other.actor.parameters()):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
        for target, source in zip(self.critic.parameters(), other.critic.parameters()):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
