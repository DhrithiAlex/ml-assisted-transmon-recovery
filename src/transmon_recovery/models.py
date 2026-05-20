import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class GPHamiltonianRecovery:
    def __init__(self):
        kernel = (ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3))
        self.models = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True) for _ in range(3)]
        self.scaler = StandardScaler()
        self.labels = ["EJ", "EC", "g"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_sc = self.scaler.fit_transform(X)
        for k, model in enumerate(self.models):
            model.fit(X_sc, y[:, k])

    def predict(self, X: np.ndarray, return_std: bool = False):
        X_sc = self.scaler.transform(X)
        preds = []
        stds = []
        for model in self.models:
            mu, sigma = model.predict(X_sc, return_std=True)
            preds.append(mu)
            stds.append(sigma)
        preds = np.column_stack(preds)
        stds = np.column_stack(stds)
        return (preds, stds) if return_std else preds

class MLPHamiltonianRecovery:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation="relu",
                                  solver="adam", max_iter=2000, early_stopping=True,
                                  validation_fraction=0.1, random_state=0)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_sc = self.scaler_X.fit_transform(X)
        y_sc = self.scaler_y.fit_transform(y)
        self.model.fit(X_sc, y_sc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler_X.transform(X)
        y_sc = self.model.predict(X_sc)
        return self.scaler_y.inverse_transform(y_sc)