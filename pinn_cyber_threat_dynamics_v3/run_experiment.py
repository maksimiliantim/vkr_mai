from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


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


class BoundedODEParameters(nn.Module):
    bounds = {
        "beta": (0.01, 5.0),
        "gamma": (0.01, 2.0),
        "kappa": (0.1, 10.0),
        "delta": (0.01, 5.0),
    }

    def __init__(self) -> None:
        super().__init__()
        self.raw_beta = nn.Parameter(self._to_raw(0.5, *self.bounds["beta"]))
        self.raw_gamma = nn.Parameter(self._to_raw(0.1, *self.bounds["gamma"]))
        self.raw_kappa = nn.Parameter(self._to_raw(1.0, *self.bounds["kappa"]))
        self.raw_delta = nn.Parameter(self._to_raw(0.3, *self.bounds["delta"]))

    @staticmethod
    def _to_raw(value: float, low: float, high: float) -> torch.Tensor:
        ratio = np.clip((value - low) / (high - low), 1e-4, 1.0 - 1e-4)
        return torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)

    @staticmethod
    def _bounded(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(raw)

    @property
    def beta(self) -> torch.Tensor:
        return self._bounded(self.raw_beta, *self.bounds["beta"])

    @property
    def gamma(self) -> torch.Tensor:
        return self._bounded(self.raw_gamma, *self.bounds["gamma"])

    @property
    def kappa(self) -> torch.Tensor:
        return self._bounded(self.raw_kappa, *self.bounds["kappa"])

    @property
    def delta(self) -> torch.Tensor:
        return self._bounded(self.raw_delta, *self.bounds["delta"])

    def as_dict(self) -> dict[str, float]:
        return {
            "beta": float(self.beta.detach().cpu()),
            "gamma": float(self.gamma.detach().cpu()),
            "kappa": float(self.kappa.detach().cpu()),
            "delta": float(self.delta.detach().cpu()),
        }


class SmoothForcing(nn.Module):
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


class FourierFeatures(nn.Module):
    def __init__(self, n_harmonics: int = 12) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.flatten()
        columns = [t]
        for k in range(1, self.n_harmonics + 1):
            columns.append(torch.sin(2.0 * math.pi * k * t))
            columns.append(torch.cos(2.0 * math.pi * k * t))
        return torch.stack(columns, dim=1)


class StateNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_layers: int = 4,
                 n_state_harmonics: int = 12) -> None:
        super().__init__()
        in_dim = 1 + 2 * n_state_harmonics
        self.fourier = FourierFeatures(n_harmonics=n_state_harmonics)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_normal_(self.head.weight, gain=0.1)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(self.fourier(t.flatten()))
        out = self.head(h)
        I = torch.sigmoid(out[:, 0])
        D = torch.nn.functional.softplus(out[:, 1])
        return I, D


# ---------------------------------------------------------------------------
# IMPROVEMENT: Self-adaptive loss weighting (Wang et al. 2021)
#   Each lambda_i is a learnable parameter initialised from the CLI value.
#   Parameterised as softplus(raw_i) to stay positive.
#   Trained with a slower LR (lr * 0.1) so the main network leads.
# ---------------------------------------------------------------------------

class AdaptiveLossWeights(nn.Module):
    """
    Replaces fixed hand-tuned lambda scalars with learnable log-scale weights.
    The model learns to up-weight whichever loss term is hardest to satisfy.
    Inspired by: Wang et al. (2021) "Understanding and mitigating gradient
    pathologies in physics-informed neural networks", NeurIPS.
    """

    def __init__(
        self,
        init_data: float = 1.0,
        init_ode: float = 0.001,
        init_ic: float = 0.1,
        init_forcing: float = 0.001,
        init_smooth: float = 0.1,
        init_d_smooth: float = 0.0,
    ) -> None:
        super().__init__()

        def _raw(v: float) -> nn.Parameter:
            # inverse-softplus so that softplus(_raw(v)) == v initially
            val = math.log(math.expm1(v)) if v > 0.001 else math.log(v + 1e-8)
            return nn.Parameter(torch.tensor(val, dtype=torch.float32))

        self.raw_data     = _raw(init_data)
        self.raw_ode      = _raw(init_ode)
        self.raw_ic       = _raw(init_ic)
        self.raw_forcing  = _raw(init_forcing)
        self.raw_smooth   = _raw(init_smooth)
        # d_smooth can start at 0; clamp to a small floor so softplus works
        self.raw_d_smooth = _raw(max(init_d_smooth, 1e-4))

    @staticmethod
    def _w(raw: nn.Parameter) -> torch.Tensor:
        return torch.nn.functional.softplus(raw)

    @property
    def lambda_data(self)     -> torch.Tensor: return self._w(self.raw_data)
    @property
    def lambda_ode(self)      -> torch.Tensor: return self._w(self.raw_ode)
    @property
    def lambda_ic(self)       -> torch.Tensor: return self._w(self.raw_ic)
    @property
    def lambda_forcing(self)  -> torch.Tensor: return self._w(self.raw_forcing)
    @property
    def lambda_smooth(self)   -> torch.Tensor: return self._w(self.raw_smooth)
    @property
    def lambda_d_smooth(self) -> torch.Tensor: return self._w(self.raw_d_smooth)

    def as_dict(self) -> dict[str, float]:
        return {k: float(v.detach()) for k, v in {
            "lambda_data":     self.lambda_data,
            "lambda_ode":      self.lambda_ode,
            "lambda_ic":       self.lambda_ic,
            "lambda_forcing":  self.lambda_forcing,
            "lambda_smooth":   self.lambda_smooth,
            "lambda_d_smooth": self.lambda_d_smooth,
        }.items()}


class RegularizedPINN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_harmonics: int = 4,
        n_state_harmonics: int = 12,
        init_lambda_data: float = 1.0,
        init_lambda_ode: float = 0.001,
        init_lambda_ic: float = 0.1,
        init_lambda_forcing: float = 0.001,
        init_lambda_smooth: float = 0.1,
        init_lambda_d_smooth: float = 0.0,
    ):
        super().__init__()
        self.state = StateNetwork(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_state_harmonics=n_state_harmonics,
        )
        self.forcing = SmoothForcing(n_harmonics=n_harmonics)
        self.params  = BoundedODEParameters()
        self.weights = AdaptiveLossWeights(
            init_data=init_lambda_data,
            init_ode=init_lambda_ode,
            init_ic=init_lambda_ic,
            init_forcing=init_lambda_forcing,
            init_smooth=init_lambda_smooth,
            init_d_smooth=init_lambda_d_smooth,
        )

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state(t.flatten())

    def ode_residuals(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = t.flatten().requires_grad_(True)
        I, D = self.state(t)
        u = self.forcing(t)
        dI_dt = torch.autograd.grad(I.sum(), t, create_graph=True, retain_graph=True)[0]
        dD_dt = torch.autograd.grad(D.sum(), t, create_graph=True, retain_graph=True)[0]
        r_I = dI_dt - (self.params.beta * I * (1.0 - I) - self.params.gamma * I + u)
        r_D = dD_dt - (self.params.kappa * I - self.params.delta * D)
        return r_I, r_D


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
        n_state_harmonics=args.n_state_harmonics,
        init_lambda_data=args.lambda_data,
        init_lambda_ode=args.lambda_ode,
        init_lambda_ic=args.lambda_ic,
        init_lambda_forcing=args.lambda_forcing,
        init_lambda_smooth=args.lambda_smooth,
        init_lambda_d_smooth=args.lambda_d_smooth,
    )

    # Separate optimizer: adaptive weights use a slower LR to stay stable
    main_params   = [p for n, p in model.named_parameters() if "weights." not in n]
    weight_params = list(model.weights.parameters())
    optimizer = torch.optim.Adam([
        {"params": main_params,   "lr": args.lr,       "weight_decay": args.weight_decay},
        {"params": weight_params, "lr": args.lr * 0.1, "weight_decay": 0.0},
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

    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_state  = ckpt["best_state"]
        best_loss   = ckpt["best_loss"]
        patience    = ckpt["patience"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']} (best_loss={best_loss:.4e})")

    _budget_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        if args.time_budget is not None and (time.time() - _budget_start) > args.time_budget:
            if args.checkpoint_path:
                torch.save({"epoch": epoch-1, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "best_state": best_state,
                    "best_loss": best_loss, "patience": patience}, args.checkpoint_path)
                print(f"Time budget reached; checkpoint saved at epoch {epoch-1}")
            break
        optimizer.zero_grad()

        I_hat, D_hat   = model(t_data)
        r_I, r_D       = model.ode_residuals(t_colloc)
        I0_hat, D0_hat = model(t0)
        u              = model.forcing(t_colloc)
        _, D_colloc    = model(t_colloc)

        data_loss     = torch.mean(data_weights * (D_hat - d_data) ** 2)
        ode_loss      = torch.mean(r_I ** 2) + torch.mean(r_D ** 2)
        ic_loss       = (I0_hat[0] - args.I0) ** 2 + (D0_hat[0] - args.D0) ** 2
        forcing_loss  = torch.mean(u ** 2)
        smooth_loss   = torch.mean((u[1:] - u[:-1]) ** 2)
        d_smooth_loss = torch.mean((D_colloc[1:] - D_colloc[:-1]) ** 2)

        w = model.weights
        total = (
            w.lambda_data     * data_loss
            + w.lambda_ode      * ode_loss
            + w.lambda_ic       * ic_loss
            + w.lambda_forcing  * forcing_loss
            + w.lambda_smooth   * smooth_loss
            + w.lambda_d_smooth * d_smooth_loss
        )

        total.backward()
        optimizer.step()

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
                f"λ_ode={float(w.lambda_ode.detach()):.5f}"
            )

        if epoch >= args.min_epochs and patience <= 0:
            print(f"Early stop at epoch {epoch}, best loss={best_loss:.4e}")
            if args.checkpoint_path:
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "best_state": best_state,
                    "best_loss": best_loss, "patience": patience, "finished": True},
                    args.checkpoint_path)
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


def residual_indicator(model: RegularizedPINN, t: np.ndarray) -> dict[str, np.ndarray | float]:
    tensor_t = torch.tensor(t, dtype=torch.float32)
    r_I, r_D = model.ode_residuals(tensor_t)
    r_i = r_I.detach().numpy()
    r_d = r_D.detach().numpy()
    R = np.sqrt(r_i ** 2 + r_d ** 2)
    threshold = float(R.mean() + 2.0 * R.std())
    return {
        "r_I": r_i,
        "r_D": r_d,
        "R": R,
        "threshold": threshold,
        "anomaly_mask": R > threshold,
    }


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
        writer.writerow(["part", "t", "D_obs", "D_pred", "I_pred", "u_pred",
                         "r_I", "r_D", "R", "is_anomaly"])
        for i in range(len(t)):
            part = "train" if train_mask[i] else "test"
            writer.writerow([
                part,
                float(t[i]), float(d_obs[i]),
                float(pred["D_pred"][i]), float(pred["I_pred"][i]),
                float(pred["u_pred"][i]),
                float(residuals["r_I"][i]), float(residuals["r_D"][i]),
                float(residuals["R"][i]), bool(residuals["anomaly_mask"][i]),
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PINN v3 — original architecture + adaptive loss weights")
    parser.add_argument("--preprocessed", required=True)
    parser.add_argument("--dataset",      default="CIC-IDS2017")
    parser.add_argument("--output_dir",   default="results_v3")
    parser.add_argument("--mode", choices=["reconstruction", "holdout", "forecast"],
                        default="reconstruction")
    parser.add_argument("--train_ratio",    type=float, default=0.8)
    parser.add_argument("--holdout_step",   type=int,   default=5)
    parser.add_argument("--holdout_offset", type=int,   default=4)
    # Architecture (identical to original)
    parser.add_argument("--hidden_dim",        type=int, default=128)
    parser.add_argument("--n_layers",          type=int, default=4)
    parser.add_argument("--n_harmonics",       type=int, default=4)
    parser.add_argument("--n_state_harmonics", type=int, default=12)
    # Training
    parser.add_argument("--epochs",      type=int,   default=5000)
    parser.add_argument("--min_epochs",  type=int,   default=1200)
    parser.add_argument("--patience",    type=int,   default=800)
    parser.add_argument("--n_colloc",    type=int,   default=1000)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-7)
    # Initial lambda values — used to initialise AdaptiveLossWeights
    # (unlike v2, these are properly forwarded into the model)
    parser.add_argument("--lambda_data",     type=float, default=1.0)
    parser.add_argument("--lambda_ode",      type=float, default=0.001)
    parser.add_argument("--lambda_ic",       type=float, default=0.1)
    parser.add_argument("--lambda_forcing",  type=float, default=0.001)
    parser.add_argument("--lambda_smooth",   type=float, default=0.1)
    parser.add_argument("--lambda_d_smooth", type=float, default=0.0)
    parser.add_argument("--peak_weight", type=float, default=0.0)
    parser.add_argument("--I0",  type=float, default=0.05)
    parser.add_argument("--D0",  type=float, default=0.0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--time_budget",     type=float, default=None)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--log_every", type=int, default=500)
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
    residuals = residual_indicator(model, t)

    summary: dict[str, Any] = {
        "dataset":          args.dataset,
        "mode":             args.mode,
        "model_version":    "v3",
        "improvement":      "self-adaptive loss weighting (Wang et al. 2021)",
        "params":           model.params.as_dict(),
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
