"""Metrics: discrimination, probability quality, capacity-constrained recall, subgroup gaps, bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with 10 equal-frequency (quantile) bins, as in Lim (2026); ties share a bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))[1:-1]
    b = np.searchsorted(edges, p, side="right")  # tied scores always share a bin
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any()))


def ece_width(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any()))


def recall_at(y, p, frac: float = 0.2, groups=None) -> float:
    """Share of positives inside the top `frac` of scores (pooled, or within each group when groups is given)."""
    y, p = np.asarray(y), np.asarray(p)
    flag = np.zeros(len(p), bool)
    if groups is None:
        k = int(np.ceil(frac * len(p)))
        flag[np.argsort(-p, kind="mergesort")[:k]] = True
    else:
        g = np.asarray(groups)
        for v in np.unique(g):
            ix = np.where(g == v)[0]
            k = int(np.ceil(frac * len(ix)))
            flag[ix[np.argsort(-p[ix], kind="mergesort")[:k]]] = True
    return float(flag[y == 1].mean())


def all_metrics(y, p, groups=None) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                logloss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), ece=ece(y, p), ece_w=ece_width(y, p),
                rec20=recall_at(y, p, 0.2), rec20_within=recall_at(y, p, 0.2, groups) if groups is not None else np.nan,
                prev=y.mean(), mean_p=p.mean(), n=len(y))


def per_group(df: pd.DataFrame, col: str, pcol: str = "p") -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(col):
        if d["y"].nunique() < 2:
            continue
        m = all_metrics(d["y"], d[pcol])
        m[col] = g
        rows.append(m)
    return pd.DataFrame(rows)


def cluster_bootstrap(df: pd.DataFrame, fn, cluster: str = "id_student", B: int = 1000, seed: int = 0):
    """Bootstrap over learners (clusters): resample learners with replacement and recompute fn(df_sample)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(df[cluster], sort=True)
    order = np.argsort(codes, kind="mergesort")
    lens = np.bincount(codes, minlength=len(uniq))
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        ln = lens[pick]
        offs = np.repeat(starts[pick] - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln)
        ix = order[np.arange(ln.sum()) + offs]
        out.append(fn(df.iloc[ix]))
    return np.array(out)


def auroc_fast(y, p) -> float:
    """AUROC via the Mann-Whitney rank statistic (ties get average ranks); equals sklearn's roc_auc_score."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj
