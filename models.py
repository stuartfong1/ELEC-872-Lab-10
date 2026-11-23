
import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)            # logit for y


class Adversary(nn.Module):
    def __init__(self, mode="eq_odds"):
        super().__init__()
        assert mode in ("parity", "eq_odds")
        self.mode = mode
        self.c = nn.Parameter(torch.ones(1))
        n_in = 4 if mode == "eq_odds" else 1
        self.lin = nn.Linear(n_in, 1)

    def forward(self, y_logit, y_true):
        s = torch.sigmoid((1 + self.c.abs()) * y_logit)
        if self.mode == "eq_odds":
            feats = torch.stack([s, s * y_true, s * (1 - y_true), y_true], dim=-1)
        else:
            feats = s.unsqueeze(-1)
        return self.lin(feats).squeeze(-1)         # logit for z
