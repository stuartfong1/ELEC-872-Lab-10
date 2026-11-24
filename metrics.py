
import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def evaluate(predictor, X, y, z, threshold=0.5, *, separate_thresholds=False):
    predictor.eval()
    p = torch.sigmoid(predictor(X)).cpu().numpy()
    y, z = y.numpy(), z.numpy()
    yhat = (p >= threshold).astype(float)

    # male_threshold = 0.5
    # female_threshold = 0.09
    # yhat[z == 1] = (p >= male_threshold)[z == 1]
    # yhat[z == 0] = (p >= female_threshold)[z == 0]

    def rate(mask):
        return yhat[mask].mean() if mask.any() else np.nan

    return {
        "accuracy":  (yhat == y).mean(),
        "DP gap":    abs(rate(z == 1) - rate(z == 0)),
        "TPR gap":   abs(rate((z == 1) & (y == 1)) - rate((z == 0) & (y == 1))),
        "FPR gap":   abs(rate((z == 1) & (y == 0)) - rate((z == 0) & (y == 0))),
        "P(yhat=1|M)": rate(z == 1),
        "P(yhat=1|F)": rate(z == 0),
    }


@torch.no_grad()
def q2_scores(predictor, X, y, z, out_title="Scores", out_filename="scores.png"):
    predictor.eval()
    p = torch.sigmoid(predictor(X)).cpu().numpy()
    y, z = y.numpy(), z.numpy()

    scores_male = p[z==1]
    scores_female = p[z==0]

    plt.hist(scores_male, label="Male")
    plt.hist(scores_female, label="Female")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.ylim(0, 4000)
    plt.title(out_title)
    plt.legend()
    plt.savefig(out_filename)
    plt.clf()
