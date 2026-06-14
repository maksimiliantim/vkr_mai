"""
run_experiment_cse2018_v3.py
============================
v3 эксперимент на CSE-CIC-IDS2018.

Архитектура: оригинальная (PiecewiseForcing + ODENet из pinn_model.py),
улучшение v3: статические лямбды заменяются на AdaptiveLossWeights
(Wang et al. 2021) — 4 обучаемых softplus-параметризованных λ.

Режимы: reconstruction, holdout, forecast
Поддержка checkpoint/time_budget для обучения по чанкам.
"""

from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── импорт оригинальной архитектуры ──────────────────────────────────────────
CSE_DIR = Path(__file__).parent.parent.parent / "vkr_code_cse2018"
sys.path.insert(0, str(CSE_DIR))
from pinn_model import CyberThreatPINN, get_device       # noqa: E402
from residual_analysis import predict_trajectory, compute_residuals, detect_anomalies  # noqa: E402


# ── AdaptiveLossWeights (Wang et al. 2021) ────────────────────────────────────

class AdaptiveLossWeights(nn.Module):
    """4 обучаемых λ: data, ode, ic, reg — softplus-параметризация."""

    def __init__(self, init_data=1.0, init_ode=0.1, init_ic=10.0, init_reg=0.01):
        super().__init__()

        def _raw(v: float) -> nn.Parameter:
            val = math.log(math.expm1(v)) if v > 0.5 else math.log(v + 1e-8)
            return nn.Parameter(torch.tensor(val, dtype=torch.float32))

        self.raw_data = _raw(init_data)
        self.raw_ode  = _raw(init_ode)
        self.raw_ic   = _raw(init_ic)
        self.raw_reg  = _raw(init_reg)

    @property
    def lambda_data(self): return F.softplus(self.raw_data)
    @property
    def lambda_ode(self):  return F.softplus(self.raw_ode)
    @property
    def lambda_ic(self):   return F.softplus(self.raw_ic)
    @property
    def lambda_reg(self):  return F.softplus(self.raw_reg)

    def as_dict(self) -> dict:
        return {
            "lambda_data": float(self.lambda_data),
            "lambda_ode":  float(self.lambda_ode),
            "lambda_ic":   float(self.lambda_ic),
            "lambda_reg":  float(self.lambda_reg),
        }


# ── обёртка модели с AdaptiveLossWeights ─────────────────────────────────────

class AdaptivePINN(nn.Module):
    def __init__(self, n_windows: int, hidden_dim=128, n_layers=6,
                 init_lambda_data=1.0, init_lambda_ode=0.1,
                 init_lambda_ic=10.0, init_lambda_reg=0.01):
        super().__init__()
        self.pinn = CyberThreatPINN(n_windows, hidden_dim, n_layers)
        self.weights = AdaptiveLossWeights(
            init_data=init_lambda_data, init_ode=init_lambda_ode,
            init_ic=init_lambda_ic,   init_reg=init_lambda_reg,
        )

    def forward(self, t):
        return self.pinn(t)

    def compute_ode_residuals(self, t):
        return self.pinn.compute_ode_residuals(t)

    def get_parameters_dict(self):
        return self.pinn.get_parameters_dict()

    @property
    def forcing(self):
        return self.pinn.forcing


def compute_loss(model: AdaptivePINN, t_data, D_obs, t_colloc,
                 I0=0.05, D0=0.0):
    w = model.weights
    _, D_hat = model(t_data)
    L_data = torch.mean((D_hat - D_obs) ** 2)

    r_I, r_D = model.compute_ode_residuals(t_colloc)
    L_ode = torch.mean(r_I ** 2) + torch.mean(r_D ** 2)

    t0 = torch.zeros(1, device=t_data.device)
    I0_hat, D0_hat = model(t0)
    L_ic = (I0_hat[0] - I0) ** 2 + (D0_hat[0] - D0) ** 2

    u = model.forcing.u_values
    L_reg = torch.mean(u ** 2)
    if len(u) > 1:
        L_reg = L_reg + torch.mean((u[1:] - u[:-1]) ** 2)

    total = (w.lambda_data * L_data + w.lambda_ode * L_ode
             + w.lambda_ic * L_ic + w.lambda_reg * L_reg)
    return total, L_data, L_ode, L_ic, L_reg


# ── функции метрик и разделения ───────────────────────────────────────────────

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    res = y_true - y_pred
    mae  = float(np.mean(np.abs(res)))
    rmse = float(np.sqrt(np.mean(res ** 2)))
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


def split_masks(n, mode, train_ratio=0.8, holdout_step=5):
    idx = np.arange(n)
    if mode == "reconstruction":
        train_mask = np.ones(n, dtype=bool)
    elif mode == "forecast":
        split = int(n * train_ratio)
        train_mask = idx < split
    else:  # holdout
        test_mask = (idx % holdout_step) == (holdout_step - 1)
        test_mask[0] = False
        train_mask = ~test_mask
    return train_mask, ~train_mask


# ── обучение ──────────────────────────────────────────────────────────────────

def train(args, t_train: np.ndarray, D_train: np.ndarray,
          n_windows_total: int) -> AdaptivePINN:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    t_data = torch.tensor(t_train, dtype=torch.float32, device=device)
    D_obs  = torch.tensor(D_train, dtype=torch.float32, device=device)

    # коллокационные точки (адаптивные вокруг пиков)
    def resample_colloc():
        tc = np.random.uniform(0, 1, args.n_colloc).astype(np.float32)
        peaks = np.argsort(D_train)[-int(args.n_colloc * 0.3):]
        tp = (t_train[peaks] + np.random.normal(0, 0.01, len(peaks))).clip(0, 1)
        return torch.tensor(np.sort(np.concatenate([tc, tp.astype(np.float32)])),
                            dtype=torch.float32, device=device)

    t_colloc = resample_colloc()

    model = AdaptivePINN(
        n_windows=n_windows_total,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        init_lambda_data=args.lambda_data,
        init_lambda_ode=args.lambda_ode,
        init_lambda_ic=args.lambda_ic,
        init_lambda_reg=args.lambda_reg,
    ).to(device)

    start_epoch = 0
    best_loss   = float("inf")

    # ── resume from checkpoint ────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from checkpoint at epoch {start_epoch - 1} "
              f"(best_loss={best_loss:.4e})")
    else:
        # ── Phase 1: pretrain data only ───────────────────────────────────────
        print("Phase 1: pretrain")
        opt_pre = torch.optim.Adam(model.pinn.ode_net.parameters(), lr=args.lr_pretrain)
        for ep in range(1, args.n_pretrain + 1):
            opt_pre.zero_grad()
            _, D_hat = model(t_data)
            loss = torch.mean((D_hat - D_obs) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pre.step()
            if ep % 500 == 0:
                print(f"  pretrain [{ep}] data={loss.item():.4e}")

    # ── Phase 2: adaptive PINN ────────────────────────────────────────────────
    main_params   = [p for n, p in model.named_parameters() if "weights." not in n]
    weight_params = list(model.weights.parameters())
    optimizer = torch.optim.Adam([
        {"params": main_params,   "lr": args.lr_pinn,       "weight_decay": 1e-7},
        {"params": weight_params, "lr": args.lr_pinn * 0.1, "weight_decay": 0.0},
    ])
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

    patience_counter = 0
    deadline = time.time() + (args.time_budget or 1e9)

    for epoch in range(start_epoch, args.n_pinn + 1):
        if time.time() >= deadline:
            print(f"Time budget reached; checkpoint saved at epoch {epoch - 1}")
            _save_ckpt(ckpt_path, model, optimizer, epoch - 1, best_loss)
            return model

        if epoch % 500 == 0:
            t_colloc = resample_colloc()

        model.train()
        optimizer.zero_grad()
        total, L_data, L_ode, L_ic, L_reg = compute_loss(
            model, t_data, D_obs, t_colloc, args.I0, args.D0)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if float(total) < best_loss:
            best_loss = float(total)
            patience_counter = 0
        else:
            patience_counter += 1

        if args.log_every and epoch % args.log_every == 0:
            w = model.weights
            _, D_hat = model(t_data)
            r2_tr = metrics(D_train, D_hat.detach().cpu().numpy())["R²"]
            print(f"[{epoch:5d}] loss={float(total):.4e} "
                  f"data={float(L_data):.4e} ode={float(L_ode):.4e} "
                  f"ic={float(L_ic):.4e} "
                  f"R2={r2_tr:.4f} "
                  f"λ_data={float(w.lambda_data):.3f} "
                  f"λ_ode={float(w.lambda_ode):.5f}")

        if patience_counter >= args.patience and epoch >= args.min_pinn:
            print(f"Early stop at epoch {epoch}")
            break

    # ── Phase 3: L-BFGS ──────────────────────────────────────────────────────
    print("Phase 3: L-BFGS")
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=20,
        history_size=50, line_search_fn="strong_wolfe")
    for step in range(args.n_lbfgs):
        def closure():
            lbfgs.zero_grad()
            loss, *_ = compute_loss(model, t_data, D_obs, t_colloc, args.I0, args.D0)
            loss.backward()
            return loss
        lbfgs.step(closure)
        if (step + 1) % 20 == 0:
            print(f"  L-BFGS [{step+1}]")

    return model


def _save_ckpt(path, model, optimizer, epoch, best_loss):
    if path is None:
        return
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch":           epoch,
        "best_loss":       best_loss,
    }, path)


# ── вывод результатов ─────────────────────────────────────────────────────────

def predict(model, t: np.ndarray) -> np.ndarray:
    device = get_device()
    model.eval()
    with torch.no_grad():
        t_t = torch.tensor(t, dtype=torch.float32, device=device)
        _, D_pred = model(t_t)
    return D_pred.cpu().numpy()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed", required=True)
    p.add_argument("--dataset",      default="CSE-IDS2018")
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--mode",         choices=["reconstruction", "holdout", "forecast"],
                   default="reconstruction")
    p.add_argument("--train_ratio",  type=float, default=0.8)
    p.add_argument("--holdout_step", type=int,   default=5)

    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--n_layers",     type=int,   default=6)
    p.add_argument("--n_pretrain",   type=int,   default=1000)
    p.add_argument("--n_pinn",       type=int,   default=5000)
    p.add_argument("--min_pinn",     type=int,   default=1000)
    p.add_argument("--n_lbfgs",      type=int,   default=100)
    p.add_argument("--patience",     type=int,   default=2000)
    p.add_argument("--n_colloc",     type=int,   default=1000)

    p.add_argument("--lr_pretrain",  type=float, default=1e-3)
    p.add_argument("--lr_pinn",      type=float, default=5e-4)
    p.add_argument("--lambda_data",  type=float, default=1.0)
    p.add_argument("--lambda_ode",   type=float, default=0.1)
    p.add_argument("--lambda_ic",    type=float, default=10.0)
    p.add_argument("--lambda_reg",   type=float, default=0.01)
    p.add_argument("--I0",           type=float, default=0.05)
    p.add_argument("--D0",           type=float, default=0.0)

    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--time_budget",  type=float, default=None,
                   help="Секунды на фазу 2; при превышении — сохранить checkpoint и выйти")
    p.add_argument("--log_every",    type=int,   default=200)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.preprocessed)
    t    = data["t_norm"].astype(np.float32)
    d    = data["D_norm"].astype(np.float32)
    n    = len(t)

    train_mask, test_mask = split_masks(
        n, args.mode, args.train_ratio, args.holdout_step)
    print(f"mode={args.mode}, train={train_mask.sum()}, test={test_mask.sum()}")

    model = train(args, t[train_mask], d[train_mask], n_windows_total=n)

    # predictions
    D_pred = predict(model, t)

    train_m = metrics(d[train_mask], D_pred[train_mask])
    test_m  = metrics(d[test_mask],  D_pred[test_mask]) if test_mask.any() else {}

    # anomaly detection
    device = get_device()
    t_grid = np.linspace(0, 1, 600).astype(np.float32)
    r_I, r_D, R = compute_residuals(model.pinn, t_grid, device)
    anomaly_info = detect_anomalies(R, t_grid, t_scale=1.0, threshold_sigma=2.0)

    w = model.weights
    summary = {
        "dataset":      args.dataset,
        "mode":         args.mode,
        "model_version": "v3",
        "improvement":  "self-adaptive loss weighting (Wang et al. 2021)",
        "params":       model.get_parameters_dict(),
        "learned_weights": w.as_dict(),
        "train_metrics": train_m,
        "n_anomaly_points": int(anomaly_info.get("n_anomalies", 0)),
        "residual_threshold": float(anomaly_info.get("threshold", 0)),
    }
    if test_m:
        summary["test_metrics"] = test_m
    print(f"  train R²={train_m['R²']:.4f}")
    if test_m:
        print(f"  test  R²={test_m['R²']:.4f}")

    import pandas as pd
    part = np.where(train_mask, "train", "test")
    pd.DataFrame({
        "t": t, "D_obs": d, "D_pred": D_pred, "part": part
    }).to_csv(out / "predictions.csv", index=False)

    torch.save(model.state_dict(), out / "model.pt")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x)
                  if isinstance(x, (np.floating, float)) else
                  int(x) if isinstance(x, (np.integer, int)) else str(x))

    print(f"\nR²: {train_m['R²']:.4f}")
    if test_m:
        print(f"R²: {test_m['R²']:.4f}")


if __name__ == "__main__":
    main()
