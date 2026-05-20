import numpy as np
from .spectroscopy import noisy_feature_vector

def generate_dataset(n_samples: int = 600,
                     noise_sigma: float = 0.01,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    EJ_samples = rng.uniform(5.0, 25.0, n_samples)
    EC_samples = rng.uniform(0.10, 0.5, n_samples)
    g_samples = rng.uniform(0.05, 0.3, n_samples)
    X_list = []
    y_list = []
    print(f"Generating {n_samples} synthetic samples …")
    for i, (EJ, EC, g) in enumerate(zip(EJ_samples, EC_samples, g_samples)):
        feats = noisy_feature_vector(EJ, EC, g, noise_sigma=noise_sigma, rng=rng)
        X_list.append(feats)
        y_list.append([EJ, EC, g])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")
    return np.array(X_list), np.array(y_list)