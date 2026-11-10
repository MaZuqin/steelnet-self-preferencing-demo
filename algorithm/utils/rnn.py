"""GRU wrapper compatible with feed-forward and recurrent PPO batches."""

from __future__ import annotations

import torch
from torch import nn


class GRULayer(nn.Module):
    """Single interface for rollout batches and flattened recurrent chunks.

    Inputs are usually ``[N, D]`` during rollout and ``[T*N, D]`` during PPO
    updates. When the number of feature rows is a multiple of the RNN-state
    batch size, this layer reconstructs the ``[T, N, D]`` sequence layout.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(int(input_dim), self.hidden_dim, self.num_layers)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def _init_states(self, batch_size: int, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(self.num_layers, batch_size, self.hidden_dim)

    def _format_states(self, rnn_states, batch_size: int, x: torch.Tensor) -> torch.Tensor:
        if rnn_states is None:
            return self._init_states(batch_size, x)
        h = rnn_states.to(device=x.device, dtype=x.dtype)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        elif h.dim() == 3 and h.size(0) != self.num_layers and h.size(1) == self.num_layers:
            h = h.transpose(0, 1).contiguous()
        return h

    def forward(self, x: torch.Tensor, rnn_states=None, masks=None):
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])

        if rnn_states is None:
            batch_size = x.size(0)
        else:
            batch_size = int(rnn_states.size(0) if rnn_states.dim() == 2 else rnn_states.size(1))
        batch_size = max(1, batch_size)

        h = self._format_states(rnn_states, batch_size, x)

        if masks is None:
            masks = x.new_ones(x.size(0), 1)
        else:
            masks = masks.to(device=x.device, dtype=x.dtype).reshape(-1, 1)

        # Rollout path: one feature row per hidden state.
        if x.size(0) == batch_size:
            h = h * masks.view(1, batch_size, 1)
            out, h = self.gru(x.unsqueeze(0), h)
            return out.squeeze(0), h[-1]

        # Recurrent update path: flattened [T*N, D], where N is hidden batch.
        if x.size(0) % batch_size != 0:
            # Fallback: treat the whole tensor as one-step samples.
            h = self._init_states(x.size(0), x)
            out, h = self.gru(x.unsqueeze(0), h)
            return out.squeeze(0), h[-1]

        time_steps = x.size(0) // batch_size
        x_seq = x.view(time_steps, batch_size, -1)
        mask_seq = masks.view(time_steps, batch_size, 1)
        outputs = []
        for t in range(time_steps):
            h = h * mask_seq[t].view(1, batch_size, 1)
            out, h = self.gru(x_seq[t : t + 1], h)
            outputs.append(out)
        return torch.cat(outputs, dim=0).reshape(time_steps * batch_size, -1), h[-1]
