"""
anomaly_validation.py — ROC-AUC валидация детектора аномалий R(t).
Работает с predictions.csv из run_experiment.py.
"""
from __future__ import annotations
import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")


def load_predictions(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def build_ground_truth(D_norm: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    nonzero = D_norm[D_norm > 1e-6]
    if len(nonzero) == 0: return np.zeros(len(D_norm), dtype=int)
    return (D_norm > np.percentile(nonzero, percentile)).astype(int)


def _cusum(D, k=0.5):
    mu, sigma = D.mean(), D.std() + 1e-10
    S = np.zeros(len(D))
    for i in range(1, len(D)):
        S[i] = max(0.0, S[i-1] + (D[i]-mu)/sigma - k)
    return S


def _rolling_z(D, window=20):
    scores = np.zeros(len(D))
    for i in range(window, len(D)):
        chunk = D[i-window:i]
        scores[i] = abs((D[i]-chunk.mean())/(chunk.std()+1e-10))
    return scores


def validate(predictions_csv: str, output_dir: str = ".",
             gt_percentile: float = 20.0) -> dict:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("sklearn не установлен"); return {}

    df = load_predictions(predictions_csv)
    D_norm = df["D_obs"].values
    R      = df["R"].values
    y_true = build_ground_truth(D_norm, gt_percentile)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        y_true = build_ground_truth(D_norm, 30.0)

    detectors = {
        "PINN R(t)": R,
        "CUSUM": _cusum(D_norm),
        "Rolling Z-score": _rolling_z(D_norm),
    }
    try:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=max(0.01, y_true.mean()), random_state=42)
        iso.fit(D_norm.reshape(-1,1))
        detectors["Isolation Forest"] = -iso.score_samples(D_norm.reshape(-1,1))
    except Exception: pass

    rows = []
    for name, scores in detectors.items():
        n = min(len(scores), len(y_true))
        s, y = scores[:n], y_true[:n]
        if y.sum() == 0 or y.sum() == n: continue
        try:
            rows.append({
                "Детектор": name,
                "ROC-AUC": round(float(roc_auc_score(y, s)), 4),
                "PR-AUC":  round(float(average_precision_score(y, s)), 4),
            })
        except Exception: pass

    rows.sort(key=lambda r: r["ROC-AUC"], reverse=True)
    print("\n=== ВАЛИДАЦИЯ ДЕТЕКТОРА АНОМАЛИЙ ===")
    for r in rows: print(f"  {r['Детектор']:<22}: ROC-AUC={r['ROC-AUC']:.4f}  PR-AUC={r['PR-AUC']:.4f}")

    result = {"metrics": rows}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "anomaly_validation.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "final_2017_reconstruction/predictions.csv"
    validate(csv, output_dir=".")
