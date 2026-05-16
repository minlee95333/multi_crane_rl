from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Categorical


class CandidateActor(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, mask):
        # obs: [B,K,F], mask: [B,K]
        logits = self.net(obs).squeeze(-1)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        return logits

    def distribution(self, obs, mask):
        logits = self.forward(obs, mask)
        return Categorical(logits=logits)


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, state):
        return self.net(state).squeeze(-1)
