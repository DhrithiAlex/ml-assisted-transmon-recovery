import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from .data import generate_dataset
from .models import MLPHamiltonianRecovery

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, labels=("EJ", "EC", "g")) -> dict:
    results = {}
    for k, lab in enumerate(labels):
        mae = mean_absolute_error(y_true[:, k], y_pred[:, k])
        r2 = r2_score(y_true[:, k], y_pred[:, k])
        results[lab] = {"MAE": mae, "R2": r2}
    return results

def noise_robustness_scan(n_samples: int = 200, noise_levels=None, seed: int = 99) -> dict:
    if noise_levels is None:
        noise_levels = [0.002, 0.005, 0.01, 0.02, 0.05]
    results = {s: {"EJ": 0.0, "EC": 0.0, "g": 0.0} for s in noise_levels}
    for sigma in noise_levels:
        print(f" noise_sigma = {sigma:.3f}")
        X, y = generate_dataset(n_samples=n_samples, noise_sigma=sigma, seed=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
        mlp = MLPHamiltonianRecovery()
        mlp.fit(X_tr, y_tr)
        y_pred = mlp.predict(X_te)
        for k, lab in enumerate(["EJ", "EC", "g"]):
            results[sigma][lab] = mean_absolute_error(y_te[:, k], y_pred[:, k])
    return results