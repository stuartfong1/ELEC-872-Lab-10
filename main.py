import matplotlib
matplotlib.use("Agg")                      # save figures to files (works with !python)
import matplotlib.pyplot as plt
import pandas as pd

from data import load_adult, set_seed
from metrics import evaluate
from train import train

EPOCHS = 20
ALPHA = 1.0

if __name__ == "__main__":
    set_seed()
    data = load_adult()
    print("Using device:", data["device"])
    test = (data["X_test"], data["y_test"], data["z_test"])

    print("\n=== 1. Baseline (no debiasing) ===")
    P_base, hist_base = train(data, debias=False, epochs=EPOCHS)

    print("\n=== 2. Debiased: demographic parity ===")
    P_dp, hist_dp = train(data, mode="parity", alpha=ALPHA, epochs=EPOCHS)

    print("\n=== 3. Debiased: equalized odds ===")
    P_eo, hist_eo = train(data, mode="eq_odds", alpha=ALPHA, epochs=EPOCHS)

    results = pd.DataFrame({
        "Baseline":              evaluate(P_base, *test),
        "Debiased: parity":      evaluate(P_dp, *test),
        "Debiased: equal. odds": evaluate(P_eo, *test),
    }).T
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print("\n=== Results on the test set (gaps: lower = fairer) ===")
    print(results.round(4))
    results.to_csv("results.csv")

    metrics = ["accuracy", "DP gap", "TPR gap", "FPR gap"]
    ax = results[metrics].T.plot(kind="bar", figsize=(9, 4), rot=0)
    ax.set_title("Baseline vs adversarial debiasing")
    ax.set_ylabel("value  (gaps: lower = fairer)")
    plt.tight_layout(); plt.savefig("comparison.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for h, name in [(hist_base, "Baseline"), (hist_dp, "Parity"), (hist_eo, "Equal. odds")]:
        for ax, m in zip(axes, metrics):
            ax.plot(h["epoch"], h[m], label=name)
    for ax, m in zip(axes, metrics):
        ax.set_title(f"Test {m}"); ax.set_xlabel("epoch"); ax.legend()
    plt.tight_layout(); plt.savefig("curves.png", dpi=120); plt.close()

    print("\nSaved: results.csv, comparison.png, curves.png")
