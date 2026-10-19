"""Probability calibration fitted on training data only.

- platt: logistic recalibration of out-of-fold training scores (Lim, 2026).
- isotonic: isotonic regression on the same out-of-fold scores.
- em_prior: label-shift EM of Saerens et al. (2002) on the unlabeled target presentation.
- cohort_prior (ours): a ridge model, trained on the training presentations, maps label-free cohort statistics
  observed on day d (early-withdrawal rate, inactivity share, median activity ratio, missed-deadline share,
  pre-start activity) to the outcome prevalence of the risk set; target probabilities are then shifted on the logit
  scale so that their mean matches the predicted prevalence. Target labels are never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Platt:
    def fit(self, s, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s).reshape(-1, 1), y)
        return self

    def __call__(self, s):
        return self.m.predict_proba(logit(s).reshape(-1, 1))[:, 1]


class Isotonic:
    def fit(self, s, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(s, y)
        return self

    def __call__(self, s):
        return self.m.predict(s)


def shift_to_mean(p: np.ndarray, target: float, iters: int = 60) -> np.ndarray:
    """Add a constant to the logits so that the mean probability equals target (bisection)."""
    z = logit(p)
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return sigmoid(z + (lo + hi) / 2)


def em_prior(p: np.ndarray, pi_s: float, iters: int = 200) -> np.ndarray:
    pi_t = pi_s
    q = p
    for _ in range(iters):
        a, b = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
        q = a * p / (a * p + b * (1 - p))
        new = q.mean()
        if abs(new - pi_t) < 1e-7:
            break
        pi_t = new
    return q


COHORT_STATS = ["withdrawn_before_d", "share_inactive", "median_active_ratio", "share_missed", "median_pre_clicks"]


def cohort_stats(feat: pd.DataFrame, base: pd.DataFrame, d: int) -> pd.DataFrame:
    """Label-free statistics per presentation on day d.
    withdrawn_before_d uses all registrations of the presentation (registry data, known on day d)."""
    allreg = base.groupby("presentation").size()
    wd = base[base["date_unregistration"] < d].groupby("presentation").size().reindex(allreg.index).fillna(0)
    g = feat.groupby("presentation")
    out = pd.DataFrame({
        "withdrawn_before_d": wd / allreg,
        "share_inactive": g["active_days"].apply(lambda s: (s == 0).mean()),
        "median_active_ratio": g["active_ratio"].median(),
        "share_missed": g["n_missed"].apply(lambda s: (s > 0).mean()),
        "median_pre_clicks": g["pre_clicks"].median(),
    }).loc[g.size().index]
    return out


class CohortPrior:
    """Ridge regression from cohort statistics to risk-set prevalence, fitted on training presentations."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, stats: pd.DataFrame, prevalence: pd.Series):
        self.mu, self.sd = stats.mean(), stats.std().replace(0, 1)
        self.m = Ridge(alpha=self.alpha).fit((stats - self.mu) / self.sd, logit(prevalence.loc[stats.index].to_numpy()))
        return self

    def predict(self, stats: pd.DataFrame) -> pd.Series:
        return pd.Series(sigmoid(self.m.predict((stats - self.mu) / self.sd)), index=stats.index)
