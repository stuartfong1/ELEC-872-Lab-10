
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import set_seed, SEED
from models import Predictor, Adversary
from metrics import evaluate


def train(data, args, mode="eq_odds", alpha=1.0, debias=True, epochs=20, batch_size=128,
          lr_p=1e-3, lr_a=1e-2, seed=SEED, verbose=True):
    set_seed(seed)
    device = data["device"]
    P = Predictor(data["n_features"], input_last_hidden_layer=args.q11_input_last_hidden_layer).to(device)
    A = Adversary(mode, add_hidden_layer=args.q6_adversary_add_hidden_layer, input_last_hidden_layer=args.q11_input_last_hidden_layer).to(device)
    opt_p = torch.optim.Adam(P.parameters(), lr=lr_p)
    opt_a = torch.optim.Adam(A.parameters(), lr=lr_a)
    sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=epochs)  # decaying LR helps stability
    loader = DataLoader(data["train_ds"], batch_size=batch_size, shuffle=True)
    p_params = list(P.parameters())
    history = []

    total_num_steps = epochs * len(loader)
    cur_step = 1

    for epoch in range(1, epochs + 1):
        P.train(); A.train()
        tot_lp = tot_la = 0.0
        for xb, yb, zb in loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)

            if args.q10_gradient_reversal and debias:
                opt_p.zero_grad()
                opt_a.zero_grad()
                y_logit = P(xb)
                loss_p = F.binary_cross_entropy_with_logits(y_logit, yb)

                y_logit = GradientReversalLayer.apply(y_logit, alpha)
                z_logit = A(y_logit, yb)
                loss_a = F.binary_cross_entropy_with_logits(z_logit, zb)

                total_loss = loss_p + loss_a
                total_loss.backward()
                opt_p.step()
                opt_a.step()

                tot_lp += loss_p.item() * len(xb)
                tot_la += loss_a.item() * len(xb)
                continue

            # ---- 1. Forward pass ----
            if args.q11_input_last_hidden_layer:
                y_logit, last_hidden_state = P(xb, output_last_hidden_state=True)
            else:
                y_logit = P(xb)
            loss_p = F.binary_cross_entropy_with_logits(y_logit, yb)

            # ---- 2. Predictor update ----
            opt_p.zero_grad()
            if debias:
                if args.q11_input_last_hidden_layer:
                    z_logit = A(last_hidden_state, yb)
                else:
                    z_logit = A(y_logit, yb)
                loss_a = F.binary_cross_entropy_with_logits(z_logit, zb)
                grad_p = torch.autograd.grad(loss_p, p_params, retain_graph=True)
                grad_a = torch.autograd.grad(loss_a, p_params, retain_graph=True, allow_unused=True)
                for w, gp, ga in zip(p_params, grad_p, grad_a):
                    if ga is None:
                        continue
                    unit_ga = ga / (ga.norm() + 1e-8)
                    proj = (gp * unit_ga).sum() * unit_ga   # part of gp that helps the adversary
                    if args.q7_anneal_adversary:
                        cur_alpha = (cur_step / total_num_steps) ** 0.5  # Anneal by sqrt(t)
                        w.grad = gp - proj - cur_alpha * ga
                        cur_step += 1
                    else:
                        w.grad = gp - proj - alpha * ga         # the paper's update rule
            else:
                loss_p.backward()
                loss_a = torch.tensor(0.0)
            opt_p.step()

            # ---- 3. Adversary update (on detached predictor output) ----
            if debias:
                opt_a.zero_grad()
                if mode == "eq_opp":
                    y_logit = y_logit[yb == 1]
                    zb = zb[yb == 1]
                    yb = yb[yb == 1]
                if args.q11_input_last_hidden_layer:
                    z_logit = A(last_hidden_state.detach(), yb)
                else:
                    z_logit = A(y_logit.detach(), yb)
                F.binary_cross_entropy_with_logits(z_logit, zb).backward()
                opt_a.step()

            tot_lp += loss_p.item() * len(xb)
            tot_la += loss_a.item() * len(xb)

        sched_p.step()
        n = len(data["train_ds"])
        m = evaluate(P, data["X_test"], data["y_test"], data["z_test"])
        history.append({"epoch": epoch, "loss_P": tot_lp / n, "loss_A": tot_la / n, **m})
        # if verbose and (epoch % 5 == 0 or epoch == 1):
        if verbose:
            print(f"epoch {epoch:3d} | L_P {tot_lp/n:.4f} | L_A {tot_la/n:.4f} | "
                  f"acc {m['accuracy']:.4f} | DP {m['DP gap']:.3f} | "
                  f"TPR {m['TPR gap']:.3f} | FPR {m['FPR gap']:.3f}")
    return P, pd.DataFrame(history)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
