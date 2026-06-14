from __future__ import annotations

import argparse
import csv
import json
import time
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    return str(value)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-10))
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


# ---------------------------------------------------------------------------
# IMPROVEMENT 1: SIREN layer & StateNetwork
#   Replaces Tanh + fixed Fourier features with sinusoidal activations.
#   Paper: Sitzmann et al. (2020) "Implicit Neural Representations with
#          Periodic Activation Functions".
# ---------------------------------------------------------------------------

class SirenLayer(nn.Module):
    """Single SIREN linear layer with sin activation."""

    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0,
                 is_first: bool = False) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, in_features: int) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class StateNetwork(nn.Module):
    """
    IMPROVEMENT 1: SIREN-based state network.
    Replaces the old Fourier+Tanh MLP with a pure SIREN network.
    SIREN naturally captures high-frequency detail in I(t) and D(t)
    and provides smooth, analytically tractable derivatives — critical
    for accurate ODE residual computation via autograd.
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 4,
                 omega_0: float = 30.0, **_kwargs) -> None:
        super().__init__()
        layers: list[nn.Module] = [SirenLayer(1, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_normal_(self.head.weight, gain=0.1)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = t.flatten().unsqueeze(-1)  # (N, 1)
        h = self.net(t)
        out = self.head(h)             # (N, 2)
        I = torch.sigmoid(out[:, 0])
        D = torch.nn.functional.softplus(out[:, 1])
        return I, D


# ---------------------------------------------------------------------------
# IMPROVEMENT 2: Time-varying ODE parameters
#   β, γ, κ, δ are now functions of time learned by tiny MLPs,
#   rather than scalar constants. This allows the model to capture
#   non-stationary attack dynamics (e.g., patch deployment → γ rises).
# ---------------------------------------------------------------------------

class TimeVaryingParam(nn.Module):
    """Lightweight MLP that outputs a bounded scalar for each time point."""

    def __init__(self, low: float, high: float, hidden: int = 16) -> None:
        super().__init__()
        self.low = low
        self.high = high
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.flatten().unsqueeze(-1)
        raw = self.net(t).squeeze(-1)
        return self.low + (self.high - self.low) * torch.sigmoid(raw)


class BoundedODEParameters(nn.Module):
    """
    IMPROVEMENT 2: Time-varying ODE parameters.
    Each parameter is a small MLP β(t), γ(t), κ(t), δ(t) instead of
    a single scalar. A regularisation term (TV loss) keeps them smooth.
    """

    _bounds = {
        "beta":  (0.01, 5.0),
        "gamma": (0.01, 2.0),
        "kappa": (0.1,  10.0),
        "delta": (0.01, 5.0),
    }

    def __init__(self) -> None:
        super().__init__()
        self.beta_net  = TimeVaryingParam(*self._bounds["beta"])
        self.gamma_net = TimeVaryingParam(*self._bounds["gamma"])
        self.kappa_net = TimeVaryingParam(*self._bounds["kappa"])
        self.delta_net = TimeVaryingParam(*self._bounds["delta"])

    def beta(self, t: torch.Tensor)  -> torch.Tensor: return self.beta_net(t)
    def gamma(self, t: torch.Tensor) -> torch.Tensor: return self.gamma_net(t)
    def kappa(self, t: torch.Tensor) -> torch.Tensor: return self.kappa_net(t)
    def delta(self, t: torch.Tensor) -> torch.Tensor: return self.delta_net(t)

    def tv_loss(self, t: torch.Tensor) -> torch.Tensor:
        """Total-variation regularisation to keep parameters smooth."""
        loss = torch.tensor(0.0)
        for net in (self.beta_net, self.gamma_net, self.kappa_net, self.delta_net):
            p = net(t)
            loss = loss + torch.mean((p[1:] - p[:-1]) ** 2)
        return loss

    def as_dict(self, t: torch.Tensor) -> dict[str, list[float]]:
        with torch.no_grad():
            return {
                "beta":  self.beta(t).cpu().tolist(),
                "gamma": self.gamma(t).cpu().tolist(),
                "kappa": self.kappa(t).cpu().tolist(),
                "delta": self.delta(t).cpu().tolist(),
            }


# ---------------------------------------------------------------------------
# Smooth forcing (unchanged)
# ---------------------------------------------------------------------------

class SmoothForcing(nn.Module):
    """Smooth low-frequency external forcing u(t)."""

    def __init__(self, n_harmonics: int = 4) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        self.linear = nn.Linear(1 + 2 * n_harmonics, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -3.0)

    def features(self, t: torch.Tensor) -> torch.Tensor:
        columns = [torch.ones_like(t)]
        for k in range(1, self.n_harmonics + 1):
            columns.append(torch.sin(2.0 * math.pi * k * t))
            columns.append(torch.cos(2.0 * math.pi * k * t))
        return torch.stack(columns, dim=1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.linear(self.features(t.flatten())).flatten())


# ---------------------------------------------------------------------------
# IMPROVEMENT 3: Self-adaptive loss weighting
#   Learnable log-scale weights λ_i trained jointly with the model.
#   Inspired by: Wang et al. (2021) "Understanding and mitigating
#   gradient pathologies in PINNs", NeurIPS.
#   Each λ_i is parameterised as exp(log_λ_i) to stay positive.
# ---------------------------------------------------------------------------

class AdaptiveLossWeights(nn.Module):
    """
    IMPROVEMENT 3: Self-adaptive loss balancing.
    Replaces hand-tuned fixed lambdas. Each weight is a learnable
    log-scale scalar: λ_i = softplus(raw_i). The model learns to up-weight
    whichever loss term is hardest to satisfy at each training stage.
    """

    def __init__(self, init_data: float = 1.0, init_ode: float = 0.001,
                 init_ic: float = 0.1, init_forcing: float = 0.001,
                 init_smooth: float = 0.1, init_tv: float = 0.01) -> None:
        super().__init__()
        def _raw(v: float) -> nn.Parameter:
            # inverse softplus
            return nn.Parameter(torch.tensor(math.log(math.expm1(v) if v > 0.001 else v), dtype=torch.float32))
        self.raw_data    = _raw(init_data)
        self.raw_ode     = _raw(init_ode)
        self.raw_ic      = _raw(init_ic)
        self.raw_forcing = _raw(init_forcing)
        self.raw_smooth  = _raw(init_smooth)
        self.raw_tv      = _raw(init_tv)

    @staticmethod
    def _w(raw: nn.Parameter) -> torch.Tensor:
        return torch.nn.functional.softplus(raw)

    @property
    def lambda_data(self)    -> torch.Tensor: return self._w(self.raw_data)
    @property
    def lambda_ode(self)     -> torch.Tensor: return self._w(self.raw_ode)
    @property
    def lambda_ic(self)      -> torch.Tensor: return self._w(self.raw_ic)
    @property
    def lambda_forcing(self) -> torch.Tensor: return self._w(self.raw_forcing)
    @property
    def lambda_smooth(self)  -> torch.Tensor: return self._w(self.raw_smooth)
    @property
    def lambda_tv(self)      -> torch.Tensor: return self._w(self.raw_tv)

    def as_dict(self) -> dict[str, float]:
        return {k: float(v.detach()) for k, v in {
            "lambda_data":    self.lambda_data,
            "lambda_ode":     self.lambda_ode,
            "lambda_ic":      self.lambda_ic,
            "lambda_forcing": self.lambda_forcing,
            "lambda_smooth":  self.lambda_smooth,
            "lambda_tv":      self.lambda_tv,
        }.items()}


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RegularizedPINN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_harmonics: int = 4,
        omega_0: float = 30.0,
        # kept for backward-compat, ignored now
        n_state_harmonics: int = 12,
        # FIX: these were parsed from the CLI but never forwarded here, so
        # AdaptiveLossWeights always silently fell back on its hardcoded
        # defaults (init_ode=0.001 etc.) regardless of --lambda_* flags.
        init_lambda_data: float = 1.0,
        init_lambda_ode: float = 0.001,
        init_lambda_ic: float = 0.1,
        init_lambda_forcing: float = 0.001,
        init_lambda_smooth: float = 0.1,
    ):
        super().__init__()
        self.state   = StateNetwork(hidden_dim=hidden_dim, n_layers=n_layers, omega_0=omega_0)
        self.forcing = SmoothForcing(n_harmonics=n_harmonics)
        self.params  = BoundedODEParameters()
        self.weights = AdaptiveLossWeights(
            init_data=init_lambda_data, init_ode=init_lambda_ode, init_ic=init_lambda_ic,
            init_forcing=init_lambda_forcing, init_smooth=init_lambda_smooth,
        )

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state(t.flatten())

    def ode_residuals(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = t.flatten().requires_grad_(True)
        I, D = self.state(t)
        u = self.forcing(t)
        beta  = self.params.beta(t)
        gamma = self.params.gamma(t)
        kappa = self.params.kappa(t)
        delta = self.params.delta(t)
        dI_dt = torch.autograd.grad(I.sum(), t, create_graph=True, retain_graph=True)[0]
        dD_dt = torch.autograd.grad(D.sum(), t, create_graph=True, retain_graph=True)[0]
        r_I = dI_dt - (beta * I * (1.0 - I) - gamma * I + u)
        r_D = dD_dt - (kappa * I - delta * D)
        return r_I, r_D


# ---------------------------------------------------------------------------
# IMPROVEMENT 4: Adaptive collocation (residual-based refinement)
#   After a warm-up period, replace the bottom-k lowest-residual collocation
#   points with new random samples from high-residual regions.
#   Paper: Daw et al. (2022) "Mitigating Propagation Failures in PINNs
#          using Retain-Resample-Release".
# ---------------------------------------------------------------------------

def adaptive_collocation(
    model: RegularizedPINN,
    t_colloc: torch.Tensor,
    n_resample: int,
) -> torch.Tensor:
    """
    IMPROVEMENT 4: Residual-based adaptive collocation.
    Re-sample the `n_resample` collocation points with lowest ODE residual
    by drawing new candidates proportional to exp(R(t)) — concentrating
    more points in regions the model currently struggles with.
    """
    with torch.enable_grad():
        r_I, r_D = model.ode_residuals(t_colloc.clone())
        R = (r_I.detach() ** 2 + r_D.detach() ** 2).sqrt()

    # keep high-residual points, drop the lowest-R ones
    n_keep = len(t_colloc) - n_resample
    _, keep_idx = torch.topk(R, k=n_keep)
    t_keep = t_colloc[keep_idx]

    # sample new points proportional to residual magnitude
    weights = R / (R.sum() + 1e-10)
    sample_idx = torch.multinomial(weights, num_samples=n_resample, replacement=True)
    t_new = t_colloc[sample_idx] + 0.01 * torch.randn(n_resample)
    t_new = t_new.clamp(0.0, 1.0)

    return torch.sort(torch.cat([t_keep, t_new]))[0].detach()


# ---------------------------------------------------------------------------
# Train / split helpers (unchanged logic, updated for new model)
# ---------------------------------------------------------------------------

def split_masks(
    n: int,
    mode: str,
    train_ratio: float,
    holdout_step: int,
    holdout_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "reconstruction":
        train_mask = np.ones(n, dtype=bool)
    elif mode == "forecast":
        split_idx = int(n * train_ratio)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:split_idx] = True
    else:
        idx = np.arange(n)
        offset = holdout_offset % holdout_step
        test_mask = (idx % holdout_step) == offset
        test_mask[0] = False
        train_mask = ~test_mask
    return train_mask, ~train_mask


def train(args: argparse.Namespace, t_train: np.ndarray, d_train: np.ndarray) -> RegularizedPINN:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = RegularizedPINN(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_harmonics=args.n_harmonics,
        omega_0=args.omega_0,
        init_lambda_data=args.lambda_data,
        init_lambda_ode=args.lambda_ode,
        init_lambda_ic=args.lambda_ic,
        init_lambda_forcing=args.lambda_forcing,
        init_lambda_smooth=args.lambda_smooth,
    )

    # Separate optimiser for adaptive weights (slower LR to avoid instability)
    main_params  = [p for n, p in model.named_parameters() if "weights." not in n]
    weight_params = list(model.weights.parameters())
    optimizer = torch.optim.Adam([
        {"params": main_params,   "lr": args.lr,          "weight_decay": args.weight_decay},
        {"params": weight_params, "lr": args.lr * 0.1,    "weight_decay": 0.0},
    ])

    t_data = torch.tensor(t_train, dtype=torch.float32)
    d_data = torch.tensor(d_train, dtype=torch.float32)
    data_weights = 1.0 + args.peak_weight * d_data
    data_weights = data_weights / torch.mean(data_weights)

    t_colloc = torch.linspace(0.0, 1.0, args.n_colloc)
    t0 = torch.zeros(1)

    best_state = None
    best_loss  = float("inf")
    patience   = args.patience
    start_epoch = 1

    # --- Resume from checkpoint (added for chunked/resumable runs under a wall-clock budget) ---
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        t_colloc   = ckpt["t_colloc"]
        best_state = ckpt["best_state"]
        best_loss  = ckpt["best_loss"]
        patience   = ckpt["patience"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']} (best_loss={best_loss:.4e})")

    _budget_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        if args.time_budget is not None and (time.time() - _budget_start) > args.time_budget:
            if args.checkpoint_path:
                torch.save({
                    "epoch": epoch - 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "t_colloc": t_colloc,
                    "best_state": best_state,
                    "best_loss": best_loss,
                    "patience": patience,
                }, args.checkpoint_path)
                print(f"Time budget reached; checkpoint saved at epoch {epoch - 1}")
            break
        optimizer.zero_grad()

        I_hat, D_hat = model(t_data)
        r_I, r_D     = model.ode_residuals(t_colloc)
        I0_hat, D0_hat = model(t0)
        u = model.forcing(t_colloc)

        data_loss    = torch.mean(data_weights * (D_hat - d_data) ** 2)
        ode_loss     = torch.mean(r_I ** 2) + torch.mean(r_D ** 2)
        ic_loss      = (I0_hat[0] - args.I0) ** 2 + (D0_hat[0] - args.D0) ** 2
        forcing_loss = torch.mean(u ** 2)
        smooth_loss  = torch.mean((u[1:] - u[:-1]) ** 2)
        tv_loss      = model.params.tv_loss(t_colloc)

        w = model.weights
        total = (
            w.lambda_data    * data_loss
            + w.lambda_ode     * ode_loss
            + w.lambda_ic      * ic_loss
            + w.lambda_forcing * forcing_loss
            + w.lambda_smooth  * smooth_loss
            + w.lambda_tv      * tv_loss
        )

        total.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # IMPROVEMENT 4: adaptive collocation refresh every N epochs
        if epoch % args.colloc_refresh == 0 and epoch >= args.min_epochs // 4:
            n_resample = max(1, args.n_colloc // 5)
            t_colloc = adaptive_collocation(model, t_colloc, n_resample)

        total_value = float(total.detach())
        if total_value < best_loss - 1e-7:
            best_loss  = total_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience   = args.patience
        else:
            patience -= 1

        if epoch == 1 or epoch % args.log_every == 0:
            train_r2 = metrics(d_train, D_hat.detach().numpy())["R²"]
            print(
                f"[{epoch:5d}] loss={total_value:.4e} "
                f"data={float(data_loss.detach()):.4e} "
                f"ode={float(ode_loss.detach()):.4e} "
                f"R2_train={train_r2:.4f} "
                f"λ_data={float(w.lambda_data.detach()):.3f} "
                f"λ_ode={float(w.lambda_ode.detach()):.4f}"
            )

        if epoch >= args.min_epochs and patience <= 0:
            print(f"Early stop at epoch {epoch}, best loss={best_loss:.4e}")
            if args.checkpoint_path:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "t_colloc": t_colloc,
                    "best_state": best_state,
                    "best_loss": best_loss,
                    "patience": patience,
                    "finished": True,
                }, args.checkpoint_path)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: RegularizedPINN, t: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        tensor_t = torch.tensor(t, dtype=torch.float32)
        I, D = model(tensor_t)
        u = model.forcing(tensor_t)
    return {
        "I_pred": I.numpy(),
        "D_pred": D.numpy(),
        "u_pred": u.numpy(),
    }


# ---------------------------------------------------------------------------
# IMPROVEMENT 5: Improved anomaly detection
#   Rolling z-score on ODE residual R(t) with configurable window,
#   replacing the global mean+2σ threshold that ignores non-stationarity.
#   Separately: Deep Ensembles runner is in ensemble_experiment.py.
# ---------------------------------------------------------------------------

def residual_indicator(
    model: RegularizedPINN,
    t: np.ndarray,
    window: int = 10,
    z_thresh: float = 2.5,
) -> dict[str, np.ndarray | float]:
    """
    IMPROVEMENT 5: Rolling z-score anomaly threshold.
    For each point i, compute z-score against a local window [i-w, i+w]
    rather than the global distribution. This correctly handles the
    non-stationary nature of network traffic residuals.
    """
    tensor_t = torch.tensor(t, dtype=torch.float32)
    r_I, r_D = model.ode_residuals(tensor_t)
    r_i = r_I.detach().numpy()
    r_d = r_D.detach().numpy()
    R   = np.sqrt(r_i ** 2 + r_d ** 2)
    n   = len(R)

    # rolling mean and std with half-window
    z_scores = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        local = R[lo:hi]
        mu    = local.mean()
        sigma = local.std() + 1e-8
        z_scores[i] = (R[i] - mu) / sigma

    anomaly_mask = z_scores > z_thresh

    # keep global threshold for backward compat in saved outputs
    global_threshold = float(R.mean() + 2.0 * R.std())

    return {
        "r_I":             r_i,
        "r_D":             r_d,
        "R":               R,
        "z_scores":        z_scores,
        "threshold":       global_threshold,
        "rolling_z_thresh": z_thresh,
        "anomaly_mask":    anomaly_mask,
    }


# ---------------------------------------------------------------------------
# Output / saving
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    t: np.ndarray,
    d_obs: np.ndarray,
    train_mask: np.ndarray,
    pred: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray | float],
    summary: dict[str, Any],
    model: RegularizedPINN,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)
    torch.save(model.state_dict(), output_dir / "model.pt")

    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "part", "t", "D_obs", "D_pred", "I_pred", "u_pred",
            "r_I", "r_D", "R", "z_score", "is_anomaly",
        ])
        for i in range(len(t)):
            part = "train" if train_mask[i] else "test"
            writer.writerow([
                part,
                float(t[i]),
                float(d_obs[i]),
                float(pred["D_pred"][i]),
                float(pred["I_pred"][i]),
                float(pred["u_pred"][i]),
                float(residuals["r_I"][i]),
                float(residuals["r_D"][i]),
                float(residuals["R"][i]),
                float(residuals["z_scores"][i]),
                bool(residuals["anomaly_mask"][i]),
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved PINN v2 experiment")
    parser.add_argument("--preprocessed", required=True)
    parser.add_argument("--dataset",      default="CIC-IDS2017")
    parser.add_argument("--output_dir",   default="results_v2")
    parser.add_argument("--mode",         choices=["reconstruction", "holdout", "forecast"],
                        default="reconstruction")
    parser.add_argument("--train_ratio",    type=float, default=0.8)
    parser.add_argument("--holdout_step",   type=int,   default=5)
    parser.add_argument("--holdout_offset", type=int,   default=4)
    # Architecture
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--n_layers",    type=int,   default=4)
    parser.add_argument("--n_harmonics", type=int,   default=4)
    parser.add_argument("--omega_0",     type=float, default=10.0,
                        help="SIREN frequency scaling. NOTE: the original v2 default of 30 (the value used in the Sitzmann et al. SIREN paper for coordinates normalised to roughly [-1,1]) is too aggressive for t normalised to [0,1] here -- empirically it makes training oscillate wildly and collapse to a constant output after ~2-3k epochs. 10 trains stably.")
    parser.add_argument("--grad_clip",   type=float, default=1.0,
                        help="Gradient-norm clipping threshold (0/None disables). Without it, training of this SIREN+adaptive-weights combination diverges (loss -> ~1e10) instead of converging.")
    parser.add_argument("--n_state_harmonics", type=int, default=12,
                        help="(legacy, ignored in v2)")
    # Training
    parser.add_argument("--epochs",           type=int,   default=5000)
    parser.add_argument("--min_epochs",       type=int,   default=1200)
    parser.add_argument("--patience",         type=int,   default=800)
    parser.add_argument("--n_colloc",         type=int,   default=1000)
    parser.add_argument("--colloc_refresh",   type=int,   default=200,
                        help="Refresh adaptive collocation every N epochs")
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--weight_decay",     type=float, default=1e-7)
    # Fixed fallback lambdas (used only to initialise adaptive weights)
    parser.add_argument("--lambda_data",    type=float, default=1.0)
    parser.add_argument("--lambda_ode",     type=float, default=0.001)
    parser.add_argument("--lambda_ic",      type=float, default=0.1)
    parser.add_argument("--lambda_forcing", type=float, default=0.001)
    parser.add_argument("--lambda_smooth",  type=float, default=0.1)
    parser.add_argument("--lambda_d_smooth",type=float, default=0.0,
                        help="(legacy, merged into tv_loss in v2)")
    parser.add_argument("--peak_weight",    type=float, default=0.0)
    # Initial conditions
    parser.add_argument("--I0", type=float, default=0.05)
    parser.add_argument("--D0", type=float, default=0.0)
    # Anomaly detection
    parser.add_argument("--anomaly_window",   type=int,   default=10,
                        help="Rolling window for z-score anomaly detection")
    parser.add_argument("--anomaly_z_thresh", type=float, default=2.5,
                        help="Z-score threshold for anomaly flag")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to load/save a training checkpoint for chunked/resumable runs (added for sandbox time-limit workaround)")
    parser.add_argument("--time_budget", type=float, default=None,
                        help="Wall-clock seconds after which training checkpoints and exits early (added for sandbox time-limit workaround)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--log_every",type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data  = np.load(args.preprocessed)
    t     = data["t_norm"].astype(np.float32)
    d_obs = data["D_norm"].astype(np.float32)

    train_mask, test_mask = split_masks(
        len(t), args.mode, args.train_ratio,
        args.holdout_step, args.holdout_offset,
    )

    print(f"mode={args.mode}, train={int(train_mask.sum())}, test={int(test_mask.sum())}")
    model     = train(args, t[train_mask], d_obs[train_mask])
    pred      = predict(model, t)
    residuals = residual_indicator(
        model, t,
        window=args.anomaly_window,
        z_thresh=args.anomaly_z_thresh,
    )

    t_tensor = torch.tensor(t, dtype=torch.float32)
    summary: dict[str, Any] = {
        "dataset":          args.dataset,
        "mode":             args.mode,
        "model_version":    "v2",
        "improvements":     [
            "SIREN StateNetwork",
            "time-varying ODE parameters",
            "self-adaptive loss weights",
            "residual-based adaptive collocation",
            "rolling z-score anomaly detection",
        ],
        "learned_weights":  model.weights.as_dict(),
        "train_metrics":    metrics(d_obs[train_mask], pred["D_pred"][train_mask]),
        "n_anomaly_points": int(np.sum(residuals["anomaly_mask"])),
        "residual_threshold": float(residuals["threshold"]),
        "config":           vars(args),
    }
    if test_mask.any():
        summary["test_metrics"] = metrics(d_obs[test_mask], pred["D_pred"][test_mask])
    else:
        summary["metrics"] = metrics(d_obs, pred["D_pred"])

    save_outputs(Path(args.output_dir), t, d_obs, train_mask, pred, residuals, summary, model)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
