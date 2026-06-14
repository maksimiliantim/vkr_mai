"""
baselines.py — сравнительные базелайны для RegularizedPINN.
Адаптировано под архитектуру run_experiment.py.
"""
from __future__ import annotations
import warnings, numpy as np
import torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings("ignore")


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    res = y_true - y_pred
    mae = float(np.mean(np.abs(res)))
    rmse = float(np.sqrt(np.mean(res**2)))
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-10))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def run_naive_lag1(D_norm):
    pred = np.roll(D_norm, 1); pred[0] = D_norm[0]
    return {"model": "Naive Lag-1", "pred": pred, "metrics": _metrics(D_norm, pred)}


def run_holt_winters(D_norm):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        m = ExponentialSmoothing(D_norm.astype(np.float64), trend="add",
                                  seasonal=None, initialization_method="estimated").fit(optimized=True)
        pred = np.clip(m.fittedvalues, 0, None)
    except Exception:
        pred = np.full_like(D_norm, D_norm.mean())
    return {"model": "Holt-Winters", "pred": pred, "metrics": _metrics(D_norm, pred)}


def run_arima(D_norm, order=(2, 1, 2)):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        pred = np.clip(ARIMA(D_norm.astype(np.float64), order=order).fit().fittedvalues, 0, None)
    except Exception:
        pred = np.full_like(D_norm, D_norm.mean())
    return {"model": f"ARIMA{order}", "pred": pred, "metrics": _metrics(D_norm, pred)}


def _make_seqs(D, seq_len):
    X = np.array([D[i:i+seq_len] for i in range(len(D)-seq_len)], dtype=np.float32)
    y = D[seq_len:].astype(np.float32)
    return X, y


class _LSTM(nn.Module):
    def __init__(self, h=64, n=2):
        super().__init__()
        self.lstm = nn.LSTM(1, h, n, batch_first=True, dropout=0.1 if n>1 else 0)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return torch.nn.functional.softplus(self.fc(o[:,-1,:]).squeeze(-1))


class _GRU(nn.Module):
    def __init__(self, h=64, n=2):
        super().__init__()
        self.gru = nn.GRU(1, h, n, batch_first=True, dropout=0.1 if n>1 else 0)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.gru(x)
        return torch.nn.functional.softplus(self.fc(o[:,-1,:]).squeeze(-1))


def _train_rnn(model, D_norm, n_epochs=2000, seq_len=10, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    X, y = _make_seqs(D_norm, seq_len)
    Xt = torch.tensor(X[:,:,None], device=device)
    yt = torch.tensor(y, device=device)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    best_loss, best_st = float("inf"), None
    for _ in range(n_epochs):
        opt.zero_grad(); loss = torch.mean((model(Xt)-yt)**2)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); best_st = {k: v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        pred_seq = model(Xt).cpu().numpy()
    return np.concatenate([D_norm[:seq_len], pred_seq])


def run_lstm(D_norm, n_epochs=2000, seed=42):
    pred = _train_rnn(_LSTM(), D_norm, n_epochs, seed=seed)
    return {"model": "LSTM", "pred": pred, "metrics": _metrics(D_norm, pred)}


def run_gru(D_norm, n_epochs=2000, seed=42):
    pred = _train_rnn(_GRU(), D_norm, n_epochs, seed=seed)
    return {"model": "GRU", "pred": pred, "metrics": _metrics(D_norm, pred)}


class _NeuralODE(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2,h), nn.Tanh(), nn.Linear(h,h), nn.Tanh(), nn.Linear(h,2))
        for m in self.f:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3); nn.init.zeros_(m.bias)
        self.y0 = nn.Parameter(torch.tensor([0.05, 0.0]))
    def rk4(self, t):
        y = self.y0.unsqueeze(0); out = [y]
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            k1 = self.f(y); k2 = self.f(y+0.5*dt*k1); k3 = self.f(y+0.5*dt*k2); k4 = self.f(y+dt*k3)
            y = y + (dt/6.0)*(k1+2*k2+2*k3+k4); out.append(y)
        return torch.cat(out, 0)
    def forward(self, t):
        return torch.nn.functional.softplus(self.rk4(t)[:,1])


def run_neural_ode(t_norm, D_norm, n_epochs=3000, seed=42):
    torch.manual_seed(seed)
    t_t = torch.tensor(t_norm, dtype=torch.float32)
    D_t = torch.tensor(D_norm, dtype=torch.float32)
    model = _NeuralODE()
    opt = optim.Adam(model.parameters(), lr=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    best_loss, best_st = float("inf"), None
    for ep in range(1, n_epochs+1):
        opt.zero_grad(); loss = torch.mean((model(t_t)-D_t)**2)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); best_st = {k: v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        pred = model(t_t).numpy()
    return {"model": "Neural ODE (RK4)", "pred": pred, "metrics": _metrics(D_norm, pred)}


def run_all_baselines(t_norm, D_norm, seed=42):
    print("Запуск базелайнов...")
    results = {}
    for name, fn in [
        ("Naive Lag-1",  lambda: run_naive_lag1(D_norm)),
        ("Holt-Winters", lambda: run_holt_winters(D_norm)),
        ("ARIMA(2,1,2)", lambda: run_arima(D_norm)),
        ("LSTM",         lambda: run_lstm(D_norm, seed=seed)),
        ("GRU",          lambda: run_gru(D_norm, seed=seed)),
        ("Neural ODE",   lambda: run_neural_ode(t_norm, D_norm, seed=seed)),
    ]:
        print(f"  {name}...")
        try:
            res = fn()
            results[name] = res
            print(f"    R²={res['metrics']['R2']:.4f} MAE={res['metrics']['MAE']:.4f}")
        except Exception as e:
            print(f"    Ошибка: {e}")
    return results
