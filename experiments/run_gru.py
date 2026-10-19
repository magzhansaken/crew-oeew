"""GRU baselines on daily activity sequences under P1/P2 (absolute vs cohort-relative sequences).

Usage: python experiments/run_gru.py --days 14 28 --protocols P1 --seeds 3
Early stopping and Platt calibration use one course-disjoint group of training modules (no target labels).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oeew import calibration as C, data, features as F, protocols as P, seqmodel as S  # noqa: E402

PRED_DIR = ROOT / "results" / "preds"


def cohort_normalise(X: np.ndarray, pres: np.ndarray) -> np.ndarray:
    Z = np.zeros_like(X)
    for p in np.unique(pres):
        ix = pres == p
        mu = X[ix].mean(0, keepdims=True); sd = X[ix].std(0, keepdims=True) + 1e-6
        Z[ix] = (X[ix] - mu) / sd
    return np.clip(Z, -5, 5)


def main(days, protocols, seeds):
    t = data.load(data.download(ROOT / "data"))
    base = data.base_table(t)
    for d in days:
        feat = pd.read_parquet(ROOT / "results" / "features" / f"feat_d{d}.parquet")
        act, Xs = F.sequences(t, base, d)
        assert (act["id_student"].to_numpy() == feat["id_student"].to_numpy()).all()
        pres = feat["presentation"].to_numpy()
        static_abs = feat[["date_registration", "num_of_prev_attempts", "studied_credits", "pre_clicks", "pre_active_days"]].to_numpy(np.float32)
        static_rel = feat[["r_date_registration", "r_num_of_prev_attempts", "r_studied_credits", "r_pre_clicks", "r_pre_active_days"]].to_numpy(np.float32)
        variants = {"gru_abs": (Xs, static_abs), "gru_rel": (cohort_normalise(Xs, pres), static_rel)}
        for proto in protocols:
            for name, (X, St) in variants.items():
                out = PRED_DIR / f"{proto}_d{d}_{name}.parquet"
                if out.exists():
                    continue
                rows, log = [], []
                for fold, tri, tei in P.PROTOCOLS[proto](feat):
                    tr = feat.iloc[tri].reset_index(drop=True)
                    itr, iva = next(P.inner_module_folds(tr, 3))
                    a, b = tri[itr], tri[iva]
                    pv_all, pt_all, tfit = [], [], []
                    for s in range(seeds):
                        t0 = time.time()
                        pv, pt = S.fit_predict(X[a], St[a], feat["y"].to_numpy()[a].astype(np.float32),
                                               X[b], St[b], feat["y"].to_numpy()[b].astype(np.float32),
                                               X[tei], St[tei], seed=s)
                        tfit.append(time.time() - t0); pv_all.append(pv); pt_all.append(pt)
                    pv, pt = np.mean(pv_all, 0), np.mean(pt_all, 0)
                    platt = C.Platt().fit(pv, feat["y"].to_numpy()[b])
                    te = feat.iloc[tei]
                    df = te[["code_module", "code_presentation", "id_student", "presentation", "y", "final_result",
                             "gender", "disability", "imd_band", "age_band"]].copy()
                    df["fold"] = fold
                    for s in range(seeds):
                        df[f"raw_s{s}"] = pt_all[s]
                    df["raw"], df["platt"], df["platt_cd"] = pt, platt(pt), platt(pt)
                    rows.append(df)
                    log.append(dict(fold=fold, t_fit_mean=float(np.mean(tfit)), n_train=len(a), n_val=len(b), n_test=len(tei)))
                    from sklearn.metrics import roc_auc_score
                    print(f"d{d} {proto} {name} {fold}: AUROC={roc_auc_score(te['y'], pt):.4f} fit={np.mean(tfit):.0f}s", flush=True)
                pd.concat(rows).to_parquet(out)
                json.dump(log, open(out.with_suffix(".json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", default=[14, 28])
    ap.add_argument("--protocols", nargs="+", default=["P1"])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(a.days, a.protocols, a.seeds)
