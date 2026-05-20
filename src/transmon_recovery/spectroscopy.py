import numpy as np
from scipy.optimize import curve_fit
from .physics import transmon_f01

def lorentzian(f: np.ndarray, f0: float, gamma: float, A: float, offset: float) -> np.ndarray:
    """Lorentzian absorption lineshape."""
    return offset + A * (gamma / 2.0) ** 2 / ((f - f0) ** 2 + (gamma / 2.0) ** 2)

def noisy_spectroscopy_features(EJ: float, EC: float, ng: float = 0.0,
                                 noise_sigma: float = 0.01,
                                 n_freq_pts: int = 200,
                                 N_max: int = 10,
                                 rng: np.random.Generator | None = None) -> tuple:
    if rng is None:
        rng = np.random.default_rng()
    f0_true = transmon_f01(EJ, EC, ng, N_max)
    gamma = 0.02 * f0_true
    f_min = f0_true * 0.97
    f_max = f0_true * 1.03
    freqs = np.linspace(f_min, f_max, n_freq_pts)
    signal = lorentzian(freqs, f0_true, gamma, A=1.0, offset=0.0)
    signal += rng.normal(0.0, noise_sigma, size=n_freq_pts)
    return freqs, signal, f0_true

def extract_peak(freqs: np.ndarray, signal: np.ndarray, f0_guess: float) -> float:
    """Fit Lorentzian and return peak frequency."""
    gamma_guess = 0.02 * f0_guess
    try:
        popt, _ = curve_fit(
            lorentzian, freqs, signal,
            p0=[f0_guess, gamma_guess, signal.max(), 0.0],
            maxfev=5000,
            bounds=([freqs[0], 0, 0, -np.inf], [freqs[-1], freqs[-1]-freqs[0], np.inf, np.inf])
        )
        return popt[0]
    except RuntimeError:
        return freqs[np.argmax(signal)]

def noisy_feature_vector(EJ: float, EC: float, g: float,
                         noise_sigma: float = 0.01,
                         ng_points: int = 7,
                         N_max: int = 10,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    ng_sweep = np.linspace(0.0, 0.5, ng_points)
    f01_fitted = []
    for ng in ng_sweep:
        _, _, f0_true = noisy_spectroscopy_features(EJ, EC, ng, noise_sigma, N_max, rng)
        freqs, signal, _ = noisy_spectroscopy_features(EJ, EC, ng, noise_sigma, N_max, rng)  # re-sample
        f01_fitted.append(extract_peak(freqs, signal, f0_true))
    f01_fitted = np.array(f01_fitted)
    g_meas = g + rng.normal(0.0, noise_sigma * g)
    return np.array([
        f01_fitted.mean(), f01_fitted.std(), f01_fitted.min(), f01_fitted.max(),
        f01_fitted[0], f01_fitted[ng_points//2], f01_fitted[-1], g_meas
    ])