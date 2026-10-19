"""Leakage-safe features at a prediction day d (records dated before d only).

Families
- REG  : registration attributes known at enrolment (Lim, 2026).
- VLE  : click aggregates over days 0..d-1 (Lim, 2026 feature set).
- SUB  : submission behaviour for submissions dated before d (Lim, 2026).
- CAL  : assessment-calendar features (deadlines are published in advance; course-level design + student response).
- TRD  : temporal trajectory features (trend, gaps, regularity).
- PRE  : activity before the official start (dates < 0, available before day 0).
- REL  : cohort-relative transforms of student-level behavioural features: percentile rank and z-score within
         the active risk set of the same module presentation on day d. No outcome labels are used.
The risk set on day d excludes enrolments whose unregistration date precedes d (Lim, 2026).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY

GROUPS = {  # activity types -> sequence channels
    "content": ["oucontent", "resource", "page", "url", "subpage", "htmlactivity", "dualpane", "folder",
                "sharedsubpage", "glossary"],
    "social": ["forumng", "oucollaborate", "ouelluminate", "ouwiki"],
    "quiz": ["quiz", "externalquiz", "questionnaire"],
    "home": ["homepage"],
    "other": ["dataplus", "repeatactivity"],
}


def top_types(svle: pd.DataFrame, k: int = 14) -> list[str]:
    """The k most frequent activity types (by number of log rows) over the whole table, sorted by name for stability."""
    vc = svle["activity_type"].value_counts()
    return sorted(vc.index[:k].tolist())


def risk_set(base: pd.DataFrame, d: int) -> pd.DataFrame:
    return base[~(base["date_unregistration"] < d)].copy()


def _slope(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[1]
    if n < 2:
        return np.zeros(mat.shape[0])
    x = np.arange(n) - (n - 1) / 2
    return (mat * x).sum(1) / (x ** 2).sum()


def _longest_gap(active: np.ndarray) -> np.ndarray:
    out = np.zeros(active.shape[0])
    cur = np.zeros(active.shape[0])
    for j in range(active.shape[1]):
        cur = np.where(active[:, j], 0, cur + 1)
        out = np.maximum(out, cur)
    return out


def build(t: dict, base: pd.DataFrame, d: int, types: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Feature table for prediction day d plus the column names of each family."""
    act = risk_set(base, d).reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(act[KEY])
    sv = t["svle"]
    w = sv[(sv["date"] >= 0) & (sv["date"] <= d - 1)]
    keys = w[KEY].astype({"code_module": str, "code_presentation": str})
    w = w.assign(code_module=keys["code_module"], code_presentation=keys["code_presentation"])
    fam: dict[str, list[str]] = {}

    # ---- daily click matrix (enrolment x day) for VLE/TRD
    daily = w.groupby(KEY + ["date"], observed=True)["sum_click"].sum().reset_index()
    pos = pd.Series(np.arange(len(act)), index=idx)
    di = daily.set_index(KEY).index
    rows = pos.reindex(di).to_numpy()
    ok = ~np.isnan(rows)
    mat = np.zeros((len(act), d))
    mat[rows[ok].astype(int), daily["date"].to_numpy()[ok].astype(int)] = daily["sum_click"].to_numpy()[ok]
    active = mat > 0

    f = pd.DataFrame(index=act.index)
    f["clicks_total"] = mat.sum(1)
    f["clicks_recent"] = mat[:, max(0, d - 7):].sum(1)
    f["clicks_first"] = mat[:, :min(7, d)].sum(1)
    f["active_days"] = active.sum(1)
    f["active_ratio"] = f["active_days"] / d
    f["daily_mean"] = mat.mean(1)
    f["daily_std"] = mat.std(1)
    f["daily_max"] = mat.max(1)
    ures = w.groupby(KEY, observed=True)["id_site"].nunique()
    f["unique_res"] = ures.reindex(idx).fillna(0).to_numpy()
    f["recent_first_ratio"] = (f["clicks_recent"] + 1) / (f["clicks_first"] + 1)
    bytype = w[w["activity_type"].isin(types)].groupby(KEY + ["activity_type"], observed=True)["sum_click"].sum().unstack(fill_value=0)
    for ty in types:
        col = bytype[ty] if ty in bytype.columns else pd.Series(dtype=float)
        f["c_" + ty] = col.reindex(idx).fillna(0).to_numpy()
    vle_cols = ["clicks_total", "clicks_recent", "clicks_first", "active_days", "active_ratio", "daily_mean",
                "daily_std", "daily_max", "unique_res", "recent_first_ratio"] + ["c_" + ty for ty in types]
    for c in vle_cols:
        if c not in ("active_ratio", "recent_first_ratio"):
            f[c] = np.log1p(f[c])
    fam["VLE"] = vle_cols

    # ---- registration
    f["date_registration"] = act["date_registration"].fillna(act["date_registration"].median()).to_numpy()
    f["num_of_prev_attempts"] = act["num_of_prev_attempts"].to_numpy()
    f["studied_credits"] = act["studied_credits"].to_numpy()
    fam["REG"] = ["date_registration", "num_of_prev_attempts", "studied_credits"]

    # ---- submissions before d (Lim) and assessment calendar (ours)
    a = t["assess"][t["assess"]["assessment_type"] != "Exam"][["code_module", "code_presentation", "id_assessment", "date"]]
    sa = t["sass"].merge(a, on="id_assessment", how="inner")
    sa = sa[sa["date_submitted"] <= d - 1]
    sa = sa.assign(late=(sa["date_submitted"] > sa["date"]).astype(int), early=(sa["date"] - sa["date_submitted"]))
    g = sa.groupby(KEY)
    f["n_submitted"] = g.size().reindex(idx).fillna(0).to_numpy()
    f["n_late"] = g["late"].sum().reindex(idx).fillna(0).to_numpy()
    f["any_submitted"] = (f["n_submitted"] > 0).astype(int)
    f["mean_days_early"] = g["early"].mean().reindex(idx).fillna(0).to_numpy()
    fam["SUB"] = ["n_submitted", "n_late", "any_submitted", "mean_days_early"]

    due = a[a["date"] <= d - 1]
    n_due = due.groupby(["code_module", "code_presentation"]).size()
    pres = pd.MultiIndex.from_frame(act[["code_module", "code_presentation"]])
    f["n_due"] = n_due.reindex(pres).fillna(0).to_numpy()
    sub_due = sa.merge(due[["id_assessment"]], on="id_assessment").groupby(KEY).size()
    f["n_due_submitted"] = sub_due.reindex(idx).fillna(0).to_numpy()
    f["n_missed"] = f["n_due"] - f["n_due_submitted"]
    f["frac_due_submitted"] = np.where(f["n_due"] > 0, f["n_due_submitted"] / f["n_due"].clip(lower=1), 1.0)
    nxt = a[a["date"] >= d].groupby(["code_module", "code_presentation"])["date"].min()
    f["days_to_next_deadline"] = (nxt.reindex(pres).to_numpy() - d)
    f["days_to_next_deadline"] = f["days_to_next_deadline"].fillna(f["days_to_next_deadline"].max())
    fam["CAL"] = ["n_due", "n_missed", "frac_due_submitted", "days_to_next_deadline"]

    # ---- trajectory
    lw = min(14, d)
    f["slope14"] = _slope(np.log1p(mat[:, d - lw:]))
    prev = mat[:, max(0, d - 14):max(0, d - 7)].sum(1)
    f["last_vs_prev_week"] = np.log1p(mat[:, max(0, d - 7):].sum(1)) - np.log1p(prev)
    f["longest_gap"] = _longest_gap(active)
    last = np.where(active.any(1), d - 1 - (active[:, ::-1].argmax(1)), -1)
    f["days_since_active"] = np.where(last >= 0, d - 1 - last, d)
    weeks = [active[:, i:i + 7].any(1) for i in range(0, d, 7)]
    f["active_weeks_ratio"] = np.mean(weeks, axis=0)
    tshare = bytype.div(bytype.sum(axis=1).replace(0, 1), axis=0)
    ent = -(tshare * np.log(tshare.where(tshare > 0, 1))).sum(axis=1)
    f["type_entropy"] = ent.reindex(idx).fillna(0).to_numpy()
    fam["TRD"] = ["slope14", "last_vs_prev_week", "longest_gap", "days_since_active", "active_weeks_ratio", "type_entropy"]

    # ---- pre-start activity
    pre = sv[sv["date"] < 0].groupby(KEY, observed=True)
    pre_c = pre["sum_click"].sum()
    pre_c.index = pre_c.index.set_levels([lv.astype(str) for lv in pre_c.index.levels[:2]] + [pre_c.index.levels[2]])
    pre_d = pre["date"].nunique()
    pre_d.index = pre_c.index
    f["pre_clicks"] = np.log1p(pre_c.reindex(idx).fillna(0).to_numpy())
    f["pre_active_days"] = pre_d.reindex(idx).fillna(0).to_numpy()
    fam["PRE"] = ["pre_clicks", "pre_active_days"]

    # ---- cohort-relative transforms (no labels)
    behav = fam["VLE"] + fam["SUB"] + ["n_missed", "frac_due_submitted"] + fam["TRD"] + fam["PRE"] + fam["REG"]
    f["presentation"] = act["presentation"].to_numpy()
    f = f.copy()
    grp = f.groupby("presentation")
    rel, newcols = [], {}
    for c in behav:
        newcols["r_" + c] = grp[c].rank(pct=True)
        mu = grp[c].transform("mean")
        sd = grp[c].transform("std").replace(0, np.nan)
        newcols["z_" + c] = ((f[c] - mu) / sd).fillna(0).clip(-5, 5)
        rel += ["r_" + c, "z_" + c]
    fam["REL"] = rel
    f = pd.concat([f.drop(columns="presentation"), pd.DataFrame(newcols, index=f.index)], axis=1)
    out = pd.concat([act[KEY + ["presentation", "y", "final_result", "gender", "disability", "imd_band", "age_band",
                                "date_unregistration"]].reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    return out, fam


def sequences(t: dict, base: pd.DataFrame, d: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Daily sequences (enrolment x day x channel) for the risk set on day d.
    Channels: log1p clicks in 5 activity groups, active flag, submissions that day, course deadline that day."""
    act = risk_set(base, d).reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(act[KEY])
    pos = pd.Series(np.arange(len(act)), index=idx)
    sv = t["svle"]
    w = sv[(sv["date"] >= 0) & (sv["date"] <= d - 1)]
    keys = w[KEY].astype({"code_module": str, "code_presentation": str})
    w = w.assign(code_module=keys["code_module"], code_presentation=keys["code_presentation"])
    g2c = {ty: i for i, (gname, tys) in enumerate(GROUPS.items()) for ty in tys}
    w = w.assign(ch=w["activity_type"].map(g2c).fillna(4).astype(int))
    agg = w.groupby(KEY + ["date", "ch"], observed=True)["sum_click"].sum().reset_index()
    X = np.zeros((len(act), d, 8), dtype=np.float32)
    r = pos.reindex(agg.set_index(KEY).index).to_numpy()
    ok = ~np.isnan(r)
    X[r[ok].astype(int), agg["date"].to_numpy()[ok].astype(int), agg["ch"].to_numpy()[ok]] = np.log1p(agg["sum_click"].to_numpy()[ok])
    X[:, :, 5] = (X[:, :, :5].sum(2) > 0)
    a = t["assess"][t["assess"]["assessment_type"] != "Exam"][["code_module", "code_presentation", "id_assessment", "date"]]
    sa = t["sass"].merge(a, on="id_assessment")
    sa = sa[(sa["date_submitted"] >= 0) & (sa["date_submitted"] <= d - 1)]
    sc = sa.groupby(KEY + ["date_submitted"]).size().reset_index(name="n")
    r = pos.reindex(sc.set_index(KEY).index).to_numpy()
    ok = ~np.isnan(r)
    X[r[ok].astype(int), sc["date_submitted"].to_numpy()[ok].astype(int), 6] = sc["n"].to_numpy()[ok]
    dl = a[(a["date"] >= 0) & (a["date"] <= d - 1)]
    pres = act[["code_module", "code_presentation"]].astype(str)
    for (m, p), grp in dl.groupby(["code_module", "code_presentation"]):
        rows = np.where((pres["code_module"] == m) & (pres["code_presentation"] == p))[0]
        for day in grp["date"].astype(int):
            X[rows, day, 7] = 1.0
    return act, X
