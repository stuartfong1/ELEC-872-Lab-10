
import numpy as np
import torch


@torch.no_grad()
def evaluate(predictor, X, y, z, threshold=0.5):
    predictor.eval()
    p = torch.sigmoid(predictor(X)).cpu().numpy()
    y, z = y.numpy(), z.numpy()
    yhat = (p >= threshold).astype(float)

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
