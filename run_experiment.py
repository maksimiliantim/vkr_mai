from __future__ import annotations

import argparse
import csv
import json
import math
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
        return torch.nn.functional.softplus(self.linear(self.features(t)).flatten())


class FourierFeatures(nn.Module):
    """Deterministic Fourier features for time-only regression."""

    def __init__(self, n_harmonics: int = 12) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics

    @property
    def n_features(self) -> int:
        return 1 + 2 * self.n_harmonics

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.flatten()
        columns = [t]
        for k in range(1, self.n_harmonics + 1):
            columns.append(torch.sin(2.0 * math.pi * k * t))
            columns.append(torch.cos(2.0 * math.pi * k * t))
        return torch.stack(columns, dim=1)


class StateNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_layers: int = 4, n_state_harmonics: int = 12) -> None:
        super().__init__()
        self.features = FourierFeatures(n_harmonics=n_state_harmonics)
        layers: list[nn.Module] = [nn.Linear(self.features.n_features, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.6)
                nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(self.features(t))
        I = torch.sigmoid(out[:, 0])
        D = torch.nn.functional.softplus(out[:, 1])
        return I, D


class RegularizedPINN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_harmonics: int = 4,
        n_state_harmonics: int = 12,
    ):
        super().__init__()
        self.state = StateNetwork(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_state_harmonics=n_state_harmonics,
        )
        self.forcing = SmoothForcing(n_harmonics=n_harmonics)
        self.params = BoundedODEParameters()

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
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t_data = torch.tensor(t_train, dtype=torch.float32)
    d_data = torch.tensor(d_train, dtype=torch.float32)
    data_weights = 1.0 + args.peak_weight * d_data
    data_weights = data_weights / torch.mean(data_weights)
    t_colloc = torch.linspace(0.0, 1.0, args.n_colloc)
    t0 = torch.zeros(1)
    best_state = None
    best_loss = float("inf")
    patience = args.patience

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        I_hat, D_hat = model(t_data)
        r_I, r_D = model.ode_residuals(t_colloc)
        I0_hat, D0_hat = model(t0)
        u = model.forcing(t_colloc)
        _, D_colloc = model(t_colloc)

        data_loss = torch.mean(data_weights * (D_hat - d_data) ** 2)
        ode_loss = torch.mean(r_I**2) + torch.mean(r_D**2)
        ic_loss = (I0_hat[0] - args.I0) ** 2 + (D0_hat[0] - args.D0) ** 2
        forcing_loss = torch.mean(u**2)
        smooth_loss = torch.mean((u[1:] - u[:-1]) ** 2)
        d_smooth_loss = torch.mean((D_colloc[1:] - D_colloc[:-1]) ** 2)
        total = (
            args.lambda_data * data_loss
            + args.lambda_ode * ode_loss
            + args.lambda_ic * ic_loss
            + args.lambda_forcing * forcing_loss
            + args.lambda_smooth * smooth_loss
            + args.lambda_d_smooth * d_smooth_loss
        )

        total.backward()
        optimizer.step()

        total_value = float(total.detach())
        if total_value < best_loss - 1e-7:
            best_loss = total_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1

        if epoch == 1 or epoch % args.log_every == 0:
            train_r2 = metrics(d_train, D_hat.detach().numpy())["R²"]
            print(
                f"[{epoch:5d}] loss={total_value:.4e} "
                f"data={float(data_loss.detach()):.4e} "
                f"ode={float(ode_loss.detach()):.4e} "
                f"R2_train={train_r2:.4f}"
            )

        if epoch >= args.min_epochs and patience <= 0:
            print(f"Early stop at epoch {epoch}, best loss={best_loss:.4e}")
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
    R = np.sqrt(r_i**2 + r_d**2)
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
        writer.writerow(["part", "t", "D_obs", "D_pred", "I_pred", "u_pred", "r_I", "r_D", "R", "is_anomaly"])
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
                bool(residuals["anomaly_mask"][i]),
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean regularized PINN experiment")
    parser.add_argument("--preprocessed", required=True)
    parser.add_argument("--dataset", default="CIC-IDS2017")
    parser.add_argument("--output_dir", default="results_new_logic")
    parser.add_argument("--mode", choices=["reconstruction", "holdout", "forecast"], default="reconstruction")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--holdout_step", type=int, default=5)
    parser.add_argument("--holdout_offset", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_harmonics", type=int, default=4)
    parser.add_argument("--n_state_harmonics", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--min_epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=800)
    parser.add_argument("--n_colloc", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--lambda_data", type=float, default=1.0)
    parser.add_argument("--lambda_ode", type=float, default=0.001)
    parser.add_argument("--lambda_ic", type=float, default=0.1)
    parser.add_argument("--lambda_forcing", type=float, default=0.001)
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_d_smooth", type=float, default=0.0)
    parser.add_argument("--peak_weight", type=float, default=0.0)
    parser.add_argument("--I0", type=float, default=0.05)
    parser.add_argument("--D0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.preprocessed)
    t = data["t_norm"].astype(np.float32)
    d_obs = data["D_norm"].astype(np.float32)
    train_mask, test_mask = split_masks(
        len(t),
        args.mode,
        args.train_ratio,
        args.holdout_step,
        args.holdout_offset,
    )

    print(f"mode={args.mode}, train={int(train_mask.sum())}, test={int(test_mask.sum())}")
    model = train(args, t[train_mask], d_obs[train_mask])
    pred = predict(model, t)
    residuals = residual_indicator(model, t)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "mode": args.mode,
        "params": model.params.as_dict(),
        "train_metrics": metrics(d_obs[train_mask], pred["D_pred"][train_mask]),
        "n_anomaly_points": int(np.sum(residuals["anomaly_mask"])),
        "residual_threshold": float(residuals["threshold"]),
        "config": vars(args),
    }
    if test_mask.any():
        summary["test_metrics"] = metrics(d_obs[test_mask], pred["D_pred"][test_mask])
    else:
        summary["metrics"] = metrics(d_obs, pred["D_pred"])

    save_outputs(Path(args.output_dir), t, d_obs, train_mask, pred, residuals, summary, model)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
