
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_adult(seed=SEED, device=None, verbose=True):
    """Returns a dict with the training dataset, test tensors and the number of features."""
    device = device or get_device()
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame.dropna().reset_index(drop=True)

    y = df["class"].astype(str).str.strip().isin([">50K", ">50K."]).astype(np.float32).values
    z = (df["sex"].astype(str).str.strip() == "Male").astype(np.float32).values

    X_df = df.drop(columns=["class", "sex", "fnlwgt"])
    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    X_df = pd.get_dummies(X_df, columns=cat_cols, dtype=np.float32)

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X_df.values.astype(np.float32), y, z,
        test_size=0.3, random_state=seed, stratify=y)

    # Standardize numeric columns (fit on train only!)
    num_idx = [X_df.columns.get_loc(c) for c in num_cols]
    scaler = StandardScaler().fit(X_train[:, num_idx])
    X_train[:, num_idx] = scaler.transform(X_train[:, num_idx])
    X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])

    if verbose:
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"P(y=1) = {y.mean():.3f}   P(z=Male) = {z.mean():.3f}")
        print(f"P(y=1 | Male) = {y[z == 1].mean():.3f}   "
              f"P(y=1 | Female) = {y[z == 0].mean():.3f}")

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {
        "train_ds": TensorDataset(t(X_train), t(y_train), t(z_train)),
        "X_test": t(X_test).to(device),
        "y_test": t(y_test),
        "z_test": t(z_test),
        "n_features": X_train.shape[1],
        "device": device,
    }
