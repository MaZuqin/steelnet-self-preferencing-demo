"""Finite-horizon critic; the execution-time actor receives no new information."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from .utils.util import check


class TimeAwareCritic(nn.Module):
    """Add known remaining time to the state encoder and a linear value skip.

    Input is [original centralized state, remaining periods]. Zero-initialized
    time weights exactly preserve the source critic before training. The two
    time features are remaining/H and the normalized discounted remaining mass.
    """
    def __init__(self, state_critic, horizon=366, gamma=.99, state_skip=False):
        super().__init__()
        if int(horizon) != horizon or horizon < 1 or not 0 < gamma <= 1:
            raise ValueError('horizon must be positive integer; gamma in (0, 1]')
        self.state_critic=state_critic
        self.horizon=int(horizon);self.gamma=float(gamma)
        self.cent_obs_dim=state_critic.cent_obs_dim+1
        self.use_recurrent=True
        first=state_critic.base.net[0]
        self.time_input=nn.Linear(2,first.out_features,bias=False)
        self.time_head=nn.Linear(2,1,bias=False)
        nn.init.zeros_(self.time_input.weight);nn.init.zeros_(self.time_head.weight)
        self.state_time_head=None
        if state_skip:self.enable_state_skip()
        self.to(state_critic.device)

    @property
    def device(self):return self.state_critic.device

    def time_features(self,remaining):
        remaining=remaining.clamp(0,self.horizon)
        fraction=remaining/self.horizon
        mass=fraction if self.gamma==1 else -torch.expm1(remaining*math.log(self.gamma))/(-math.expm1(self.horizon*math.log(self.gamma)))
        return torch.cat((fraction,mass),dim=-1)

    def enable_state_skip(self):
        """Linear state/time interaction, initialized without changing values."""
        if self.state_time_head is None:
            self.state_time_head=nn.Linear(self.state_critic.cent_obs_dim*2,1,bias=False).to(self.device)
            nn.init.zeros_(self.state_time_head.weight)

    def state_time_features(self,inputs):
        mass=self.time_features(inputs[:,-1:])[:,-1:]
        state=inputs[:,:-1]
        return torch.cat((state,state*mass),dim=-1)

    def forward(self,cent_obs,rnn_states=None,masks=None):
        inputs=check(cent_obs,device=self.device,dtype=torch.float32).reshape(-1,self.cent_obs_dim)
        clock=self.time_features(inputs[:,-1:])
        state=self.state_critic
        features=state.base.net[0](inputs[:,:-1])+self.time_input(clock)
        for layer in list(state.base.net.children())[1:]:features=layer(features)
        features,next_hidden=state.rnn(features,check(rnn_states,device=self.device,dtype=torch.float32),
                                       check(masks,device=self.device,dtype=torch.float32))
        values=state.value_head(features)+self.time_head(clock)
        if self.state_time_head is not None:values=values+self.state_time_head(self.state_time_features(inputs))
        return values,next_hidden


def critic_observation(env,critic):
    """Known clock only; never consult future demand or change local observations."""
    state=np.asarray(env.global_state(),dtype=np.float32)
    if isinstance(critic,TimeAwareCritic):
        return np.concatenate((state,np.asarray([max(0,env.episode_max_steps-env.step_num)],np.float32)))
    return state
