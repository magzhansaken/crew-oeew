"""Statistics and tables for the paper, computed from results/eval (no model fitting here).

Outputs (results/analysis/):
- main_<proto>.csv           pooled and module-level metrics per day/method/calibration
- per_presentation.parquet   metrics per test presentation (unit of the paired tests)
- tests.csv                  paired Wilcoxon tests over presentations (Holm-adjusted within each family) and
                             learner-cluster bootstrap 95% CIs of pooled differences
- prevalence.csv             presentation-level prevalence estimates (platt_cd mean, EM, ACC, cohort prior, truth)
- fairness.csv, cost.csv
- numbers.json               every number quoted in the paper
Usage: python experiments/analysis.py [--boot 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oeew import metrics as Mx  # noqa: E402

EV, OUT = ROOT / "results" / "eval", ROOT / "results" / "analysis"
CACHE = OUT / "cache"  # bootstrap results keyed by comparison and by the input files' timestamps (resumable runs)


def cached(key: str, inputs: list[Path], fn):
    """Return fn() from the cache when the same key was computed from unchanged input files."""
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = [int(fp.stat().st_mtime) for fp in inputs]
    fp = CACHE / (key.replace("|", "_").replace("+", "p") + ".json")
    if fp.exists():
        c = json.load(open(fp))
        if c.get("stamp") == stamp:
            return c["value"]
    v = fn()
    json.dump(dict(stamp=stamp, value=v), open(fp, "w"))
    return v
OURS = "crew"
BASELINES = ["lim_lr", "lim_rf", "lim_hgb", "lgbm_lim", "lgbm_abs_bal", "cat_abs", "gru_abs", "gru_rel", "lgbm_abs"]
CAL_OURS = "s_platt"     # CREW is calibrated exactly like the baselines (Platt on learner-grouped training OOF)
CAL_LIM = "s_platt"      # Lim (2026): Platt on student-grouped out-of-fold scores


def calib_of(df_cols, pref):
    for c in pref:
        if c in df_cols:
            return c
    return None


def per_presentation(rows: pd.DataFrame, col: str) -> pd.DataFrame:
    out = []
    for pr, g in rows.groupby("presentation"):
        y, p = g["y"].to_numpy(), g[col].to_numpy()
        out.append(dict(presentation=pr, code_module=g["code_module"].iloc[0], n=len(g), prev=y.mean(), mean_p=p.mean(),
                        prev_err=abs(p.mean() - y.mean()), auroc=roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
                        ece=Mx.ece(y, p), brier=float(np.mean((p - y) ** 2)),
                        rec20=Mx.recall_at(y, p, 0.2)))
    return pd.DataFrame(out)


def load_rows(proto, d, m):
    fp = EV / "rows" / f"{proto}_d{d}_{m}.parquet"
    return pd.read_parquet(fp) if fp.exists() else None


def _cluster_index(groups):
    codes, uniq = pd.factorize(groups, sort=True)
    order = np.argsort(codes, kind="mergesort")
    lens = np.bincount(codes, minlength=len(uniq))
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return order, lens, starts


def boot_arrays(y, pa, pb, groups, B: int, seed: int = 0):
    """Learner-cluster bootstrap of (a - b) differences in AUROC, ECE and Brier score on identical test rows."""
    y, pa, pb = np.asarray(y, float), np.asarray(pa, float), np.asarray(pb, float)

    def fn(yy, a, c):
        return np.array([Mx.auroc_fast(yy, a) - Mx.auroc_fast(yy, c), Mx.ece(yy, a) - Mx.ece(yy, c),
                         np.mean((a - yy) ** 2) - np.mean((c - yy) ** 2)])
    est = fn(y, pa, pb)
    order, lens, starts = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    bs = np.empty((B, 3))
    for k in range(B):
        pick = rng.integers(0, len(lens), len(lens))
        ln = lens[pick]
        offs = np.repeat(starts[pick] - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln)
        ix = order[np.arange(ln.sum()) + offs]
        bs[k] = fn(y[ix], pa[ix], pb[ix])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    p_two = [float(min(1.0, 2 * min((bs[:, i] <= 0).mean(), (bs[:, i] >= 0).mean()))) for i in range(3)]
    names = ("auroc", "ece", "brier")
    return ({f"d_{n}": float(e) for n, e in zip(names, est)} | {f"lo_{n}": float(v) for n, v in zip(names, lo)} |
            {f"hi_{n}": float(v) for n, v in zip(names, hi)} | {f"pboot_{n}": v for n, v in zip(names, p_two)})


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, ca: str, cb: str, B: int, seed: int = 0):
    k = ["code_module", "code_presentation", "id_student"]
    j = a[k + ["y", ca]].rename(columns={ca: "pa"}).merge(b[k + [cb]].rename(columns={cb: "pb"}), on=k)
    if len(j) != len(a):
        raise ValueError("test rows differ between methods")
    return boot_arrays(j["y"], j["pa"], j["pb"], j["id_student"], B, seed)


def main(B: int):
    OUT.mkdir(parents=True, exist_ok=True)
    M = pd.read_parquet(EV / "metrics.parquet")
    M.to_csv(OUT / "metrics_all.csv", index=False)
    nums: dict = {}

    # ---------- per-presentation table for every (proto, day, method, calibration of interest)
    PP = []
    for (proto, d, m), _ in M.groupby(["proto", "day", "method"]):
        rows = load_rows(proto, d, m)
        if rows is None:
            continue
        for c in [c for c in rows.columns if c == "raw" or c.startswith("s_")]:
            pp = per_presentation(rows, c).assign(proto=proto, day=d, method=m, calib=c)
            PP.append(pp)
    PP = pd.concat(PP, ignore_index=True)
    PP.to_parquet(OUT / "per_presentation.parquet")

    # ---------- paired tests: each of our models vs every other model, per protocol and day
    ABLATIONS = {"crew_lim", "crew_rel", "crew_noCAL", "lgbm_abs_noCAL", "lgbm_abs_noTRD", "lgbm_abs_noPRE", "lim_lr_vle"}
    tests = []
    for (proto, d), g in M.groupby(["proto", "day"]):
        present = sorted(set(g["method"]))
        for ours in [m for m in ("crew", "gru_rel") if m in present]:
            a = load_rows(proto, d, ours)
            ca = "s_platt"
            fam_p, block = {}, []
            for b_m in [m for m in present if m != ours and m not in ABLATIONS]:
                b = load_rows(proto, d, b_m)
                for cb_name, cb in (("platt", "s_platt"), ("prior", "s_platt+cprior"), ("gate", "s_gate")):
                    if b is None or cb not in b.columns:
                        continue
                    r = dict(proto=proto, day=d, ours=ours, ours_cal=ca, baseline=b_m, base_cal=cb)
                    pa = PP.query("proto==@proto and day==@d and method==@ours and calib==@ca").set_index("presentation")
                    pb = PP.query("proto==@proto and day==@d and method==@b_m and calib==@cb").set_index("presentation")
                    common = pa.index.intersection(pb.index)
                    for met in ("auroc", "ece", "brier"):
                        da, db = pa.loc[common, met], pb.loc[common, met]
                        ok = da.notna() & db.notna()
                        diff = (da - db)[ok]
                        r[f"pres_{met}_wins"] = int((diff > 0).sum()) if met == "auroc" else int((diff < 0).sum())
                        r[f"pres_{met}_n"] = int(ok.sum())
                        r[f"pres_{met}_median_diff"] = float(diff.median()) if len(diff) else np.nan
                        if ok.sum() >= 5 and (diff != 0).any():
                            r[f"p_{met}"] = float(wilcoxon(da[ok], db[ok]).pvalue)
                            fam_p[(b_m, cb_name, met)] = r[f"p_{met}"]
                    if B > 0 and cb == "s_platt" and len(a) == len(b):
                        r |= cached(f"test|{proto}|{d}|{ours}|{b_m}|B{B}",
                                    [EV / "rows" / f"{proto}_d{d}_{ours}.parquet", EV / "rows" / f"{proto}_d{d}_{b_m}.parquet"],
                                    lambda: boot_diff(a, b, ca, cb, B))
                    block.append(r)
            adj = Mx.holm(fam_p) if fam_p else {}
            for r in block:
                cb_name = {"s_platt": "platt", "s_platt+cprior": "prior", "s_gate": "gate"}[r["base_cal"]]
                for met in ("auroc", "ece", "brier"):
                    if (r["baseline"], cb_name, met) in adj:
                        r[f"p_{met}_holm"] = adj[(r["baseline"], cb_name, met)]
            tests += block
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "tests.csv", index=False)

    # ---------- effect of the cohort prior on each model (same rows): prior or gate minus Platt, P1
    pe = []
    for (proto, d, m), _ in M[M["proto"] == "P1"].groupby(["proto", "day", "method"]):
        rows_m = load_rows(proto, d, m)
        if rows_m is None or "s_platt+cprior" not in rows_m.columns:
            continue
        for c in ("s_platt+cprior", "s_gate"):
            r = dict(proto=proto, day=d, method=m, calib=c)
            if B > 0:
                r |= cached(f"prior|{proto}|{d}|{m}|{c}|B{B}", [EV / "rows" / f"{proto}_d{d}_{m}.parquet"],
                            lambda: boot_arrays(rows_m["y"], rows_m[c], rows_m["s_platt"], rows_m["id_student"], B))
            pe.append(r)
    pd.DataFrame(pe).to_csv(OUT / "prior_effect.csv", index=False)

    # ---------- prevalence estimation quality (presentation level)
    prev = PP[PP["calib"].str.startswith("s_") & ~PP["calib"].str.contains("oracle")]
    pv = prev.groupby(["proto", "day", "method", "calib"]).agg(mae=("prev_err", "mean"), maxerr=("prev_err", "max"),
                                                             n=("prev_err", "size")).reset_index()
    pv.to_csv(OUT / "prevalence.csv", index=False)

    # ---------- fairness: AUROC and within-course recall@20% by subgroup, ours vs Lim LR (P1, each day)
    fr = []
    for d in sorted(M.query("proto=='P1'")["day"].unique()):
        for m, c in ((OURS, None), ("lim_lr", CAL_LIM)):
            rows = load_rows("P1", d, m)
            if rows is None:
                continue
            c = c or calib_of(rows.columns, [CAL_OURS, "s_platt"])
            for attr in ("gender", "disability", "imd_band", "age_band"):
                for v, g in rows.groupby(attr):
                    if g["y"].nunique() < 2 or len(g) < 200:
                        continue
                    fr.append(dict(day=d, method=m, calib=c, attr=attr, group=v, n=len(g), prev=g["y"].mean(),
                                   auroc=roc_auc_score(g["y"], g[c]), ece=Mx.ece(g["y"], g[c]),
                                   rec20=Mx.recall_at(g["y"].to_numpy(), g[c].to_numpy(), 0.2),
                                   mean_p=g[c].mean()))
    F = pd.DataFrame(fr)
    F.to_csv(OUT / "fairness.csv", index=False)

    # ---------- cost (fit/inference time, features) from run logs
    cost = []
    for fp in sorted((ROOT / "results" / "preds").glob("*.json")):
        proto, dtag, m = fp.stem.split("_", 2)
        lg = json.load(open(fp))
        cost.append(dict(proto=proto, day=int(dtag[1:]), method=m,
                         t_select=float(np.mean([x.get("t_select", 0) for x in lg])),
                         t_fit=float(np.mean([x.get("t_fit_mean", np.nan) for x in lg])),
                         t_infer=float(np.mean([x.get("t_infer_mean", np.nan) for x in lg])),
                         n_features=float(np.mean([x.get("n_features", np.nan) for x in lg])),
                         n_test=int(np.sum([x.get("n_test", 0) for x in lg]))))
    pd.DataFrame(cost).to_csv(OUT / "cost.csv", index=False)

    # ---------- numbers.json: pooled metrics of every (proto, day, method, calib)
    keep = ["auroc", "auprc", "brier", "logloss", "ece", "ece_w", "rec20", "rec20_within", "mod_auroc", "mod_ece",
            "mod_ece_max", "mod_auprc", "mod_brier", "n", "prev", "mean_p"] + [c for c in M.columns if c.startswith("seed_")]
    for _, r in M.iterrows():
        nums[f"{r['proto']}|d{r['day']}|{r['method']}|{r['calib']}"] = {k: (float(r[k]) if isinstance(r[k], (int, float, np.floating, np.integer)) and pd.notna(r[k]) else (r[k] if pd.notna(r[k]) else None)) for k in keep if k in r}
    nums["tests"] = T.replace({np.nan: None}).to_dict(orient="records")
    json.dump(nums, open(OUT / "numbers.json", "w"), indent=1, default=float)
    print("analysis written to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    main(ap.parse_args().boot)
