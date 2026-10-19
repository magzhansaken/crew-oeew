"""Evaluation protocols. Every split removes from training all learners who appear in the test part.

P1 leave-one-module-out (course-disjoint and student-disjoint), as in Lim (IJACSA 17(9), 2026).
P2 forward in time: train on 2013 presentations, test on 2014 presentations (student-disjoint).
P3 random student-disjoint 5-fold cross-validation that mixes modules (optimistic reference).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _disjoint(df: pd.DataFrame, test_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    te = np.where(test_mask)[0]
    test_students = set(df.loc[test_mask, "id_student"])
    tr = np.where(~test_mask & ~df["id_student"].isin(test_students).to_numpy())[0]
    return tr, te


def p1_lomo(df: pd.DataFrame):
    for m in sorted(df["code_module"].unique()):
        tr, te = _disjoint(df, (df["code_module"] == m).to_numpy())
        yield f"LOMO:{m}", tr, te


def p2_forward(df: pd.DataFrame):
    test = df["code_presentation"].str.startswith("2014").to_numpy()
    tr, te = _disjoint(df, test)
    yield "FWD:2013->2014", tr, te


def p3_random(df: pd.DataFrame, k: int = 5, seed: int = 0):
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(sgk.split(df, df["y"], df["id_student"])):
        mask = np.zeros(len(df), bool); mask[te] = True
        tr, te2 = _disjoint(df, mask)
        yield f"RND:{i}", tr, te2


PROTOCOLS = {"P1": p1_lomo, "P2": p2_forward, "P3": p3_random}


def inner_module_folds(df: pd.DataFrame, n_groups: int = 3, seed: int = 0):
    """Inner course-disjoint folds inside a training set: modules are partitioned into n_groups groups."""
    mods = sorted(df["code_module"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(mods)
    groups = [mods[i::n_groups] for i in range(n_groups)]
    for g in groups:
        tr, te = _disjoint(df, df["code_module"].isin(g).to_numpy())
        yield tr, te


def inner_student_folds(df: pd.DataFrame, k: int = 4, seed: int = 0):
    """Inner student-grouped folds used for out-of-fold calibration (as in Lim, 2026)."""
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in sgk.split(df, df["y"], df["id_student"]):
        yield tr, te
