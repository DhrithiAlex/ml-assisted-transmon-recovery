import numpy as np
from scipy.linalg import eigh

def build_transmon_hamiltonian(EJ: float, EC: float, ng: float, N_max: int = 10) -> np.ndarray:
    """
    Build the transmon Hamiltonian in the charge basis.
    """
    dim = 2 * N_max + 1
    n_vals = np.arange(-N_max, N_max + 1, dtype=float)
    H = np.diag(4.0 * EC * (n_vals - ng) ** 2)
    for i in range(dim - 1):
        H[i, i + 1] -= EJ / 2.0
        H[i + 1, i] -= EJ / 2.0
    return H

def transmon_f01(EJ: float, EC: float, ng: float = 0.0, N_max: int = 10) -> float:
    """Return the 0→1 transition frequency [GHz]."""
    H = build_transmon_hamiltonian(EJ, EC, ng, N_max)
    energies = eigh(H, eigvals_only=True, subset_by_index=[0, 1])
    return float(energies[1] - energies[0])