"""GRU baselines with the same calibration inputs as the tabular models (version used in the paper).

Per outer fold and variant (absolute or cohort-normalised daily sequences):
- main model: trained on two thirds of the training modules with early stopping on the remaining course-disjoint
  group (log-loss), averaged over seeds;
- out-of-fold scores for Platt scaling from four learner-grouped folds of the training data, and course-disjoint
  out-of-fold scores from three module groups, both trained for the early-stopped number of epochs (seed 0).
Outputs follow run_tabular.py, so evaluate.py applies Platt, the cohort prior, the gate and the other schemes.
Usage: python experiments/run_gru2.py --days 7 14 28 42 --protocols P1 --seeds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from oeew import calibration as C, data, features as F, protocols as P, seqmodel as S  # noqa: E402
from run_gru import cohort_normalise  # noqa: E402

PRED_DIR, OOF_DIR = ROOT / "results" / "preds", ROOT / "results" / "oof"


def main(days, protocols, seeds):
    t = data.load(data.download(ROOT / "data"))
    base = data.base_table(t)
    for d in days:
        feat = pd.read_parquet(ROOT / "results" / "features" / f"feat_d{d}.parquet")
        act, Xs = F.sequences(t, base, d)
        assert (act["id_student"].to_numpy() == feat["id_student"].to_numpy()).all()
        pres = feat["presentation"].to_numpy()
        y = feat["y"].to_numpy().astype(np.float32)
        st_abs = feat[["date_registration", "num_of_prev_attempts", "studied_credits", "pre_clicks", "pre_active_days"]].to_numpy(np.float32)
        st_rel = feat[["r_date_registration", "r_num_of_prev_attempts", "r_studied_credits", "r_pre_clicks", "r_pre_active_days"]].to_numpy(np.float32)
        variants = {"gru_abs": (Xs, st_abs), "gru_rel": (cohort_normalise(Xs, pres), st_rel)}
        for proto in protocols:
            for name, (X, St) in variants.items():
                out = PRED_DIR / f"{proto}_d{d}_{name}.parquet"
                if out.exists() and (OOF_DIR / out.name).exists():
                    continue
                rows, log, oofs = [], [], []
                for fold, tri, tei in P.PROTOCOLS[proto](feat):
                    tr = feat.iloc[tri].reset_index(drop=True)
                    itr, iva = next(P.inner_module_folds(tr, 3))
                    a, b = tri[itr], tri[iva]
                    pt_all, eps, tfit = [], [], []
                    for s in range(seeds):
                        t0 = time.time()
                        _, pt, ep = S.fit_predict(X[a], St[a], y[a], X[b], St[b], y[b], X[tei], St[tei], seed=s,
                                                  return_epoch=True)
                        tfit.append(time.time() - t0); pt_all.append(pt); eps.append(ep)
                    E = int(np.median(eps))
                    oof = np.zeros(len(tri)); cd = np.zeros(len(tri))
                    for ia, ib in P.inner_student_folds(tr, 4):
                        oof[ib] = S.fit_fixed(X[tri[ia]], St[tri[ia]], y[tri[ia]], [(X[tri[ib]], St[tri[ib]])], 0, E)[0]
                    for ia, ib in P.inner_module_folds(tr, 3):
                        cd[ib] = S.fit_fixed(X[tri[ia]], St[tri[ia]], y[tri[ia]], [(X[tri[ib]], St[tri[ib]])], 0, E)[0]
                    pm = np.mean(pt_all, 0)
                    platt, platt_cd = C.Platt().fit(oof, tr["y"]), C.Platt().fit(cd, tr["y"])
                    te = feat.iloc[tei]
                    df = te[["code_module", "code_presentation", "id_student", "presentation", "y", "final_result",
                             "gender", "disability", "imd_band", "age_band"]].copy()
                    df["fold"] = fold
                    for s in range(seeds):
                        df[f"raw_s{s}"] = pt_all[s]
                    df["raw"], df["platt"], df["platt_cd"] = pm, platt(pm), platt_cd(pm)
                    rows.append(df)
                    o = tr[["presentation", "code_module", "id_student", "y"]].copy()
                    o["fold"] = fold; o["oof"] = oof; o["oof_cd"] = cd
                    oofs.append(o)
                    log.append(dict(fold=fold, epochs=eps, t_fit_mean=float(np.mean(tfit)), n_train=len(a), n_val=len(b),
                                    n_test=len(tei), n_features=int(X.shape[1] * X.shape[2] + St.shape[1])))
                    print(f"d{d} {proto} {name} {fold}: AUROC={roc_auc_score(te['y'], pm):.4f} E={E} fit={np.mean(tfit):.0f}s",
                          flush=True)
                pd.concat(rows).to_parquet(out)
                OOF_DIR.mkdir(parents=True, exist_ok=True)
                pd.concat(oofs).to_parquet(OOF_DIR / out.name)
                json.dump(log, open(out.with_suffix(".json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", default=[7, 14, 28, 42])
    ap.add_argument("--protocols", nargs="+", default=["P1"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    main(a.days, a.protocols, a.seeds)
