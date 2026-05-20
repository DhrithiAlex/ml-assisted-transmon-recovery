import os
import numpy as np
from transmon_recovery.data import generate_dataset
from transmon_recovery.models import GPHamiltonianRecovery, MLPHamiltonianRecovery
from transmon_recovery.evaluation import evaluate_model
from transmon_recovery.visualization import plot_example_spectroscopy, plot_dispersion
from sklearn.model_selection import train_test_split

def main():
    OUT = "outputs/"
    os.makedirs(OUT, exist_ok=True)
    print("="*65)
    print(" Transmon Hamiltonian Recovery – ML Pipeline")
    print("="*65)

    plot_example_spectroscopy(save_path=OUT + "fig1_spectroscopy_trace.png")
    plot_dispersion(save_path=OUT + "fig2_charge_dispersion.png")

    print("\nBuilding training dataset …")
    X, y = generate_dataset(n_samples=600, noise_sigma=0.01, seed=42)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    # GP on smaller set
    idx_gp = np.random.default_rng(0).choice(len(X_tr), 200, replace=False)
    X_tr_gp = X_tr[idx_gp]
    y_tr_gp = y_tr[idx_gp]

    print("Training models …")
    mlp = MLPHamiltonianRecovery()
    mlp.fit(X_tr, y_tr)
    y_pred_mlp = mlp.predict(X_te)

    gp = GPHamiltonianRecovery()
    gp.fit(X_tr_gp, y_tr_gp)
    y_pred_gp = gp.predict(X_te)

    print("\nEvaluation:")
    res_mlp = evaluate_model(y_te, y_pred_mlp)
    res_gp = evaluate_model(y_te, y_pred_gp)
    print(res_mlp)
    print(res_gp)

    print("\n✓ Pipeline completed! Figures saved to", OUT)

if __name__ == "__main__":
    main()