"""Model factory. Lim-2026 configurations are reproduced exactly where the paper states them."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Lim (IJACSA 17(9), 2026), Section on models: L2 LR; RF with 160 trees and min leaf 10; HGB with 180 iterations,
# learning rate 0.08, 31 leaf nodes and L2 regularisation (value not stated; 1.0 used here).
LIM = {
    "lr": dict(C=1.0),
    "rf": dict(n_estimators=160, min_samples_leaf=10),
    "hgb": dict(max_iter=180, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0),
}

LGBM_GRID = [dict(num_leaves=nl, learning_rate=0.05, min_child_samples=mc, n_estimators=300)
             for nl in (15, 31) for mc in (50, 200)]
CAT_GRID = [dict(depth=dp, learning_rate=0.05, iterations=400, l2_leaf_reg=3) for dp in (4, 6)]


def make(name: str, params: dict | None = None, seed: int = 0):
    p = dict(params or {})
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, **{**LIM["lr"], **p}))
    if name == "rf":
        return RandomForestClassifier(n_jobs=2, random_state=seed, **{**LIM["rf"], **p})
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, **{**LIM["hgb"], **p})
    if name == "lgbm":
        base = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=2,
                    random_state=seed)
        return lgb.LGBMClassifier(**{**base, **p})
    if name == "cat":
        base = dict(verbose=0, random_seed=seed, thread_count=2, allow_writing_files=False)
        return CatBoostClassifier(**{**base, **p})
    raise ValueError(name)


def grid(name: str) -> list[dict]:
    return {"lgbm": LGBM_GRID, "cat": CAT_GRID}.get(name, [{}])


def predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
