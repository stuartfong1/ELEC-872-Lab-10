
import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, n_in, hidden=64, *, input_last_hidden_layer=False):
        super().__init__()
        self.input_last_hidden_layer = input_last_hidden_layer
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))
        self.out_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, output_last_hidden_state=False):
        x = self.net(x)
        if output_last_hidden_state:
            return self.out_proj(x).squeeze(-1), x.squeeze(-1)  # logit for y
        return self.out_proj(x).squeeze(-1)            # logit for y


class Adversary(nn.Module):
    def __init__(self, mode="eq_odds", *, add_hidden_layer=False, input_last_hidden_layer=False):
        super().__init__()
        assert mode in ("parity", "eq_odds", "eq_opp")
        self.mode = mode
        if not input_last_hidden_layer:
            self.c = nn.Parameter(torch.ones(1))
        n_in = 4 if mode in {"eq_odds", "eq_opp"} else 1
        self.input_last_hidden_layer = input_last_hidden_layer
        if add_hidden_layer:
            self.lin = nn.Sequential(
                nn.Linear(n_in, 4),
                nn.ReLU(),
                nn.Linear(4, 1),
            )
        elif input_last_hidden_layer:
            self.lin = nn.Linear(64, 1)
        else:
            self.lin = nn.Linear(n_in, 1)

    def forward(self, y_logit, y_true):
        if self.input_last_hidden_layer:
            last_hidden_state = y_logit
            return self.lin(last_hidden_state).squeeze(-1)

        s = torch.sigmoid((1 + self.c.abs()) * y_logit)
        if self.mode in {"eq_odds", "eq_opp"}:
            feats = torch.stack([s, s * y_true, s * (1 - y_true), y_true], dim=-1)
        else:
            feats = s.unsqueeze(-1)
        return self.lin(feats).squeeze(-1)         # logit for z
