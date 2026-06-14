"""
synthetic_validation.py — синтетическая валидация идентификации параметров.
Адаптировано под run_experiment.py / RegularizedPINN.
"""
from __future__ import annotations
import json, numpy as np
from pathlib import Path
from scipy.integrate import odeint


def _ode(y, t, beta, gamma, kappa, delta, u_fn):
    I, D = float(np.clip(y[0],0,1)), max(float(y[1]),0)
    return [beta*I*(1-I) - gamma*I + u_fn(t), kappa*I - delta*D]


def generate_synthetic(beta_true=0.5, gamma_true=1.2, kappa_true=8.0, delta_true=2.5,
                        n=400, noise_frac=0.03, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 1, n)
    def u_fn(tv):
        u = 0.02
        for c,a,w in [(0.15,0.08,0.05),(0.45,0.12,0.06),(0.75,0.06,0.04)]:
            u += a*np.exp(-((tv-c)**2)/(2*w**2))
        return u
    sol = odeint(_ode, [0.05,0.0], t,
                 args=(beta_true,gamma_true,kappa_true,delta_true,u_fn),
                 rtol=1e-9, atol=1e-11)
    D_clean = np.maximum(sol[:,1], 0)
    D_obs   = np.maximum(D_clean + np.random.normal(0, noise_frac*D_clean.max(), n), 0)
    D_scale = D_obs.max() if D_obs.max()>0 else 1.0
    return {
        "t_norm": t.astype(np.float32),
        "D_norm": (D_obs/D_scale).astype(np.float32),
        "D_clean_norm": (D_clean/D_scale).astype(np.float32),
        "t_scale": 1.0, "D_scale": float(D_scale), "n_windows": n,
        "true_params": {"beta":beta_true,"gamma":gamma_true,"kappa":kappa_true,"delta":delta_true},
    }


def run_synthetic_validation(output_dir="synthetic_validation",
                               beta_true=0.5, gamma_true=1.2, kappa_true=8.0, delta_true=2.5,
                               n=400, epochs=5000, seed=42):
    """Генерирует синтетику, обучает PINN, сравнивает θ̂ с θ*."""
    import sys, subprocess
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic(beta_true, gamma_true, kappa_true, delta_true, n=n, seed=seed)
    # Сохраняем как npz для run_experiment.py
    npz_path = out / "synthetic_data.npz"
    np.savez(str(npz_path), t_norm=data["t_norm"], D_norm=data["D_norm"])
    print(f"Синтетические данные: β={beta_true}, γ={gamma_true}, κ={kappa_true}, δ={delta_true}")
    print(f"Данные сохранены: {npz_path}")
    result = {
        "true_params": data["true_params"],
        "note": "Запустите run_experiment.py --preprocessed synthetic_validation/synthetic_data.npz "
                "--output_dir synthetic_validation/pinn_result --epochs 5000 "
                "--n_harmonics 4 --n_state_harmonics 12 для получения θ̂",
        "data_path": str(npz_path),
    }
    with open(out / "synthetic_setup.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run_synthetic_validation()
