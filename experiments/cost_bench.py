"""Computational cost of every model on one held-out module (P1, day 28), measured on an otherwise idle machine.

For a new course the platform trains once on the other courses: nested selection, out-of-fold scores for Platt
scaling (four learner-grouped and three course-disjoint folds), five seeds of the final model, and the label-free
cohort prior. This script times each step for one model and records the peak resident memory of its process.

Usage: python experiments/cost_bench.py --method crew [--fold LOMO:CCC] [--day 28]
       python experiments/cost_bench.py --all          (runs every model in its own process, then the feature steps)
Output: results/analysis/cost_bench.csv
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from oeew import calibration as C, data, features as F, models as M, protocols as P  # noqa: E402

OUT = ROOT / "results" / "analysis" / "cost_bench.csv"
TAB = ["lim_lr", "lim_rf", "lim_hgb", "lgbm_lim", "lgbm_iw", "lgbm_abs_bal", "cat_abs", "lgbm_abs", "crew"]
SEQ = ["gru_abs", "gru_rel"]


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def one(method: str, fold_name: str, d: int, seeds: int = 5) -> dict:
    import run_tabular as R
    feat = pd.read_parquet(ROOT / "results" / "features" / f"feat_d{d}.parquet")
    fam = json.load(open(ROOT / "results" / "features" / f"fam_d{d}.json"))
    fold = next((f, a, b) for f, a, b in P.PROTOCOLS["P1"](feat) if f == fold_name)
    _, tri, tei = fold
    tr, te = feat.iloc[tri].reset_index(drop=True), feat.iloc[tei].reset_index(drop=True)
    r = dict(method=method, day=d, fold=fold_name, n_train=len(tr), n_test=len(te))
    if method in TAB:
        spec = R.method_specs(fam)[method]
        t0 = time.perf_counter()
        if spec.get("iw"):
            tr = tr.assign(_w=R.domain_weights(tr[spec["sets"][0]], te[spec["sets"][0]]))
        (si, cfg), _ = R.select(spec, tr)
        cols = spec["sets"][si]
        r["t_select"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        oof = np.zeros(len(tr))
        for a, b in P.inner_student_folds(tr, 4):
            oof[b] = M.predict(R.fit(spec, cfg, cols, tr.iloc[a], 0), tr.iloc[b][cols])
        for a, b in P.inner_module_folds(tr, 3):
            M.predict(R.fit(spec, cfg, cols, tr.iloc[a], 0), tr.iloc[b][cols])
        platt = C.Platt().fit(oof, tr["y"])
        r["t_calib"] = time.perf_counter() - t0
        n_seed = 1 if spec["model"] == "lr" else seeds
        fits, infs, raw = [], [], []
        for s in range(n_seed):
            t0 = time.perf_counter(); mdl = R.fit(spec, cfg, cols, tr, s); fits.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); raw.append(M.predict(mdl, te[cols])); infs.append(time.perf_counter() - t0)
        r["t_fit_one"], r["t_fit_all"] = float(np.mean(fits)), float(np.sum(fits))
        r["t_infer_per_1000"] = float(np.sum(infs)) / len(te) * 1000
        r["n_features"] = len(cols)
        p = platt(np.mean(raw, 0))
    else:
        from oeew import seqmodel as S
        from run_gru import cohort_normalise
        t = data.load(data.download(ROOT / "data"))
        base = data.base_table(t)
        act, Xs = F.sequences(t, base, d)
        pres = feat["presentation"].to_numpy()
        y = feat["y"].to_numpy().astype(np.float32)
        if method == "gru_abs":
            X = Xs
            St = feat[["date_registration", "num_of_prev_attempts", "studied_credits", "pre_clicks",
                       "pre_active_days"]].to_numpy(np.float32)
        else:
            X = cohort_normalise(Xs, pres)
            St = feat[["r_date_registration", "r_num_of_prev_attempts", "r_studied_credits", "r_pre_clicks",
                       "r_pre_active_days"]].to_numpy(np.float32)
        itr, iva = next(P.inner_module_folds(tr, 3))
        a, b = tri[itr], tri[iva]
        r["t_select"] = 0.0  # early stopping is part of each fit
        fits, eps, pts = [], [], []
        for s in range(seeds):
            t0 = time.perf_counter()
            _, pt, ep = S.fit_predict(X[a], St[a], y[a], X[b], St[b], y[b], X[tei], St[tei], seed=s, return_epoch=True)
            fits.append(time.perf_counter() - t0); eps.append(ep); pts.append(pt)
        E = int(np.median(eps))
        t0 = time.perf_counter()
        oof = np.zeros(len(tri))
        for ia, ib in P.inner_student_folds(tr, 4):
            oof[ib] = S.fit_fixed(X[tri[ia]], St[tri[ia]], y[tri[ia]], [(X[tri[ib]], St[tri[ib]])], 0, E)[0]
        for ia, ib in P.inner_module_folds(tr, 3):
            S.fit_fixed(X[tri[ia]], St[tri[ia]], y[tri[ia]], [(X[tri[ib]], St[tri[ib]])], 0, E)
        platt = C.Platt().fit(oof, tr["y"])
        r["t_calib"] = time.perf_counter() - t0
        r["t_fit_one"], r["t_fit_all"] = float(np.mean(fits)), float(np.sum(fits))
        import torch
        net = S.GRUNet(X.shape[2], St.shape[1]).eval()  # forward pass costs the same for trained weights
        xt, stt = torch.tensor(X[tei], dtype=torch.float32), torch.tensor(St[tei], dtype=torch.float32)
        t0 = time.perf_counter()
        for _ in range(seeds):
            with torch.no_grad():
                torch.sigmoid(net(xt, stt))
        r["t_infer_per_1000"] = (time.perf_counter() - t0) / len(tei) * 1000
        r["n_features"] = int(X.shape[1] * X.shape[2] + St.shape[1])
        p = platt(np.mean(pts, 0))
    r["t_total"] = r["t_select"] + r["t_calib"] + r["t_fit_all"]
    r["peak_mb"] = peak_mb()
    r["mean_p"] = float(np.mean(p))
    return r


def prior_cost(d: int) -> dict:
    t = data.load(data.download(ROOT / "data"))
    base = data.base_table(t)
    feat = pd.read_parquet(ROOT / "results" / "features" / f"feat_d{d}.parquet")
    t0 = time.perf_counter()
    stats = C.cohort_stats(feat, base, d)
    t_stats = time.perf_counter() - t0
    trp = feat.groupby("presentation")["y"].mean()
    t0 = time.perf_counter()
    cp = C.CohortPrior().fit(stats.loc[trp.index], trp)
    pi = cp.predict(stats.loc[trp.index])
    p = np.full(len(feat), 0.44)
    for pres, ix in feat.groupby("presentation").indices.items():
        p[ix] = C.shift_to_mean(p[ix], pi[pres])
    t_prior = time.perf_counter() - t0
    types = F.top_types(t["svle"])
    t0 = time.perf_counter()
    F.build(t, base, d, types)
    t_feat = time.perf_counter() - t0
    t0 = time.perf_counter()
    F.sequences(t, base, d)
    t_seq = time.perf_counter() - t0
    return dict(method="_features_and_prior", day=d, n_test=len(feat), t_features=t_feat, t_sequences=t_seq,
                t_cohort_stats=t_stats, t_prior_all_presentations=t_prior, peak_mb=peak_mb())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method")
    ap.add_argument("--fold", default="LOMO:CCC")
    ap.add_argument("--day", type=int, default=28)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prior", action="store_true")
    a = ap.parse_args()
    if a.all:
        rows = []
        for m in TAB + SEQ:
            res = subprocess.run([sys.executable, "-I", __file__, "--method", m, "--fold", a.fold, "--day", str(a.day)],
                                 capture_output=True, text=True, check=True)
            rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
            print(rows[-1], flush=True)
        res = subprocess.run([sys.executable, "-I", __file__, "--prior", "--day", str(a.day)], capture_output=True,
                             text=True, check=True)
        rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print("written", OUT)
    elif a.prior:
        print(json.dumps(prior_cost(a.day)))
    else:
        print(json.dumps(one(a.method, a.fold, a.day)))
