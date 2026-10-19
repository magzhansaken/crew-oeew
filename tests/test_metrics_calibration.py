"""Metric formulas and calibration utilities on cases with known answers."""
import numpy as np
import pandas as pd
import pytest

from oeew import calibration as C, metrics as M


def test_ece_zero_when_calibrated():
    p = np.repeat(np.linspace(0.05, 0.95, 10), 100)
    y = np.concatenate([np.r_[np.ones(int(round(q * 100))), np.zeros(100 - int(round(q * 100)))]
                        for q in np.linspace(0.05, 0.95, 10)])
    assert M.ece(y, p) == pytest.approx(0.0, abs=1e-9)
    assert M.ece_width(y, p) == pytest.approx(0.0, abs=1e-9)


def test_ece_constant_offset():
    y = np.r_[np.ones(300), np.zeros(700)]
    assert M.ece(y, np.full(1000, 0.5)) == pytest.approx(0.2)


def test_recall_at_top_fraction():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([.9, .8, .1, .2, .3, .1, .2, .3, .1, .05])
    assert M.recall_at(y, p, 0.2) == pytest.approx(2 / 3)
    g = np.array([0] * 5 + [1] * 5)  # top-1 inside each half
    assert M.recall_at(y, p, 0.2, g) == pytest.approx(1 / 3)


def test_holm_matches_definition():
    adj = M.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_shift_to_mean_hits_target():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 500)
    for q in (0.1, 0.45, 0.8):
        s = C.shift_to_mean(p, q)
        assert s.mean() == pytest.approx(q, abs=1e-4)
        assert np.all(np.diff(s[np.argsort(p)]) >= -1e-12)  # ranking (AUROC) is unchanged


def test_em_prior_keeps_scores_when_prior_matches():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.3
    p = np.clip(np.where(y, 0.6, 0.2) + rng.normal(0, 0.05, 5000), 0.01, 0.99)
    q = C.em_prior(p, p.mean())
    assert np.allclose(q, p, atol=1e-6)


def test_platt_is_monotone():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.01, 0.99, 2000); y = rng.random(2000) < s
    pl = C.Platt().fit(s, y)
    grid = np.linspace(0.02, 0.98, 50)
    assert np.all(np.diff(pl(grid)) > 0)


def test_cohort_prior_recovers_linear_logit():
    rng = np.random.default_rng(0)
    st = pd.DataFrame(rng.normal(size=(40, 5)), columns=C.COHORT_STATS, index=[f"p{i}" for i in range(40)])
    prev = pd.Series(C.sigmoid(0.8 * st.iloc[:, 0] - 0.3), index=st.index)
    cp = C.CohortPrior(alpha=1e-3).fit(st, prev)
    assert np.allclose(cp.predict(st), prev, atol=1e-3)


def test_auroc_fast_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    y = rng.random(3000) < 0.4
    p = np.round(rng.random(3000) * 0.5 + 0.3 * y, 2)  # with ties
    assert M.auroc_fast(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_cluster_bootstrap_keeps_clusters_whole():
    df = pd.DataFrame({"id_student": [5, 5, 5, 9, 2, 2], "v": [1, 1, 1, 2, 3, 3]})
    for seed in range(5):
        s = M.cluster_bootstrap(df, lambda d: d.groupby("id_student").size().to_dict(), B=1, seed=seed)[0]
        for k, v in s.items():
            assert v % {5: 3, 9: 1, 2: 2}[k] == 0  # a learner's rows are always drawn together


def test_em_recovers_prior_under_pure_label_shift():
    """Saerens EM recovers the target prior when p(x|y) is unchanged and posteriors are calibrated at the source."""
    rng = np.random.default_rng(0)
    from scipy.stats import norm

    def posterior(x, pi):  # x | y=1 ~ N(1,1), x | y=0 ~ N(-1,1)
        a, b = pi * norm.pdf(x, 1, 1), (1 - pi) * norm.pdf(x, -1, 1)
        return a / (a + b)
    y_t = rng.random(20000) < 0.2
    x_t = np.where(y_t, rng.normal(1, 1, 20000), rng.normal(-1, 1, 20000))
    q = C.em_prior(posterior(x_t, 0.5), 0.5)
    assert q.mean() == pytest.approx(0.2, abs=0.01)
