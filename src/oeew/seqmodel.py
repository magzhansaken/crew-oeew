"""GRU over daily activity sequences plus static features (CPU, PyTorch)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)


class GRUNet(nn.Module):
    def __init__(self, n_ch: int, n_static: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x, s):
        _, h = self.gru(x)
        return self.head(torch.cat([h[-1], s], 1)).squeeze(1)


def standardize(train: np.ndarray, *others: np.ndarray):
    axes = tuple(range(train.ndim - 1))
    mu = train.mean(axis=axes, keepdims=True)
    sd = train.std(axis=axes, keepdims=True) + 1e-6
    return [(a - mu) / sd for a in (train, *others)]


def fit_fixed(Xtr, Str, ytr, Xte_list, seed: int = 0, epochs: int = 5, bs: int = 512, lr: float = 1e-3):
    """Train for a fixed number of epochs (no validation data); return scores for each array pair in Xte_list."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu_x = Xtr.mean(axis=(0, 1), keepdims=True); sd_x = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    mu_s = Str.mean(axis=0, keepdims=True); sd_s = Str.std(axis=0, keepdims=True) + 1e-6
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    xtr, str_, ytr_t = T((Xtr - mu_x) / sd_x), T((Str - mu_s) / sd_s), T(ytr)
    net = GRUNet(Xtr.shape[2], Str.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(max(1, epochs)):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lossf(net(xtr[b], str_[b]), ytr_t[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return [torch.sigmoid(net(T((X - mu_x) / sd_x), T((S - mu_s) / sd_s))).numpy() for X, S in Xte_list]


def fit_predict(Xtr, Str, ytr, Xva, Sva, yva, Xte, Ste, seed: int = 0, epochs: int = 15, bs: int = 512, lr: float = 1e-3,
                return_epoch: bool = False):
    """Train with early stopping on the validation part (by log-loss); return validation and test scores."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    Str, Sva, Ste = standardize(Str, Sva, Ste)
    net = GRUNet(Xtr.shape[2], Str.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    xtr, str_, ytr_t = T(Xtr), T(Str), T(ytr)
    xva, sva = T(Xva), T(Sva)
    best, best_state, bad, best_ep = np.inf, None, 0, 1
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(xtr[b], str_[b]), ytr_t[b])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(xva, sva)).numpy()
        vl = -np.mean(yva * np.log(np.clip(pv, 1e-6, 1)) + (1 - yva) * np.log(np.clip(1 - pv, 1e-6, 1)))
        if vl < best - 1e-4:
            best, bad, best_ep = vl, 0, ep + 1
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pv = torch.sigmoid(net(xva, sva)).numpy()
        pt = torch.sigmoid(net(T(Xte), T(Ste))).numpy()
    return (pv, pt, best_ep) if return_epoch else (pv, pt)
