"""Run tabular models under protocols P1-P3 and prediction days; save row-level predictions.

Usage: python experiments/run_tabular.py --days 7 14 28 42 --protocols P1 P2 P3 --methods all --seeds 5
All tuning, calibration and prevalence models are fitted inside the training part of each outer split.
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
from oeew import calibration as C  # noqa: E402
from oeew import data, features as F, models as M, protocols as P  # noqa: E402

FEAT_DIR = ROOT / "results" / "features"
PRED_DIR = ROOT / "results" / "preds"
OOF_DIR = ROOT / "results" / "oof"


def feature_table(t, base, d, types):
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    fp, jp = FEAT_DIR / f"feat_d{d}.parquet", FEAT_DIR / f"fam_d{d}.json"
    if fp.exists() and jp.exists():
        return pd.read_parquet(fp), json.load(open(jp))
    X, fam = F.build(t, base, d, types)
    X.to_parquet(fp); json.dump(fam, open(jp, "w"))
    return X, fam


def method_specs(fam):
    ABS = fam["REG"] + fam["VLE"] + fam["SUB"] + fam["CAL"] + fam["TRD"] + fam["PRE"]
    LIM = fam["REG"] + fam["VLE"] + fam["SUB"]
    REL_LIM = [c for c in fam["REL"] if c[2:] in set(LIM)]
    drop = lambda f: [c for c in ABS if c not in set(fam[f])]  # noqa: E731
    drop_rel = lambda f: [c for c in fam["REL"] if c[2:] not in set(fam[f])]  # noqa: E731
    return {
        # ablations (pre-specified): model only, one family removed, relative encoding of Lim's features only
        "lgbm_abs_noCAL": dict(model="lgbm", sets=[drop("CAL")], tune=True),
        "lgbm_abs_noTRD": dict(model="lgbm", sets=[drop("TRD")], tune=True),
        "lgbm_abs_noPRE": dict(model="lgbm", sets=[drop("PRE")], tune=True),
        "crew_lim": dict(model="lgbm", sets=[REL_LIM, REL_LIM + LIM], tune=True),
        "crew_noCAL": dict(model="lgbm", sets=[drop_rel("CAL"), drop_rel("CAL") + drop("CAL")], tune=True),
        "lim_lr": dict(model="lr", sets=[LIM]),
        "lim_lr_vle": dict(model="lr", sets=[fam["VLE"]]),
        "lim_rf": dict(model="rf", sets=[LIM]),
        "lim_hgb": dict(model="hgb", sets=[LIM]),
        "lgbm_lim": dict(model="lgbm", sets=[LIM], tune=True),
        "lgbm_abs": dict(model="lgbm", sets=[ABS], tune=True),
        "lgbm_abs_bal": dict(model="lgbm", sets=[ABS], tune=True, balanced=True),
        # covariate-shift baseline: importance weights from a domain classifier (training vs unlabeled target cohort)
        "lgbm_iw": dict(model="lgbm", sets=[ABS], tune=True, iw=True),
        "cat_abs": dict(model="cat", sets=[ABS], tune=True),
        "crew": dict(model="lgbm", sets=[fam["REL"], fam["REL"] + ABS], tune=True),
        "crew_rel": dict(model="lgbm", sets=[fam["REL"]], tune=True),
    }


def domain_weights(src: pd.DataFrame, tgt: pd.DataFrame, clip: float = 20.0) -> np.ndarray:
    """Importance weights w(x) = p(target|x)/p(source|x) * n_s/n_t from an L2 logistic domain classifier
    (Sugiyama et al., 2007); only unlabeled target features are used. Clipped and normalised to mean 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Z = pd.concat([src, tgt]).to_numpy(float)
    dom = np.r_[np.zeros(len(src)), np.ones(len(tgt))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(Z, dom)
    p = np.clip(clf.predict_proba(src.to_numpy(float))[:, 1], 1e-6, 1 - 1e-6)
    w = np.clip(p / (1 - p) * len(src) / len(tgt), 1 / clip, clip)
    return w / w.mean()


def weights(df, balanced):
    if "_w" in df.columns:
        return df["_w"].to_numpy()
    if not balanced:
        return None
    n = df["code_module"].map(df["code_module"].value_counts())
    return (1.0 / n * len(df) / df["code_module"].nunique()).to_numpy()


def fit(spec, cfg, cols, df, seed):
    m = M.make(spec["model"], cfg, seed)
    w = weights(df, spec.get("balanced"))
    if w is not None:
        m.fit(df[cols], df["y"], **({"sample_weight": w}))
    else:
        m.fit(df[cols], df["y"])
    return m


def select(spec, tr):
    """Nested selection of (feature set, hyperparameters) by pooled AUROC over inner course-disjoint folds."""
    cands = [(si, cfg) for si in range(len(spec["sets"])) for cfg in M.grid(spec["model"])]
    if len(cands) == 1:
        return cands[0], None
    folds = list(P.inner_module_folds(tr, 3))
    scores = {}
    for si, cfg in cands:
        ys, ps = [], []
        for itr, iva in folds:
            m = fit(spec, cfg, spec["sets"][si], tr.iloc[itr], 0)
            ps.append(M.predict(m, tr.iloc[iva][spec["sets"][si]])); ys.append(tr.iloc[iva]["y"].to_numpy())
        scores[(si, json.dumps(cfg, sort_keys=True))] = roc_auc_score(np.concatenate(ys), np.concatenate(ps))
    best = max(scores, key=scores.get)
    return (best[0], json.loads(best[1])), scores[best]


def run(days, protocols, methods, seeds):
    t = data.load(data.download(ROOT / "data"))
    base = data.base_table(t)
    types = F.top_types(t["svle"])
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for d in days:
        X, fam = feature_table(t, base, d, types)
        stats = C.cohort_stats(X, base, d)
        specs = method_specs(fam)
        for proto in protocols:
            for meth in (list(specs) if methods == ["all"] else methods):
                out = PRED_DIR / f"{proto}_d{d}_{meth}.parquet"
                if out.exists():
                    continue
                spec = specs[meth]
                rows, log, oof_rows = [], [], []
                for fold, tri, tei in P.PROTOCOLS[proto](X):
                    tr, te = X.iloc[tri].reset_index(drop=True), X.iloc[tei].reset_index(drop=True)
                    t0 = time.time()
                    if spec.get("iw"):
                        tr = tr.assign(_w=domain_weights(tr[spec["sets"][0]], te[spec["sets"][0]]))
                    (si, cfg), inner = select(spec, tr)
                    cols = spec["sets"][si]
                    t_sel = time.time() - t0
                    # out-of-fold scores for calibration (student-grouped, inside training)
                    oof = np.zeros(len(tr))
                    for a, b in P.inner_student_folds(tr, 4):
                        oof[b] = M.predict(fit(spec, cfg, cols, tr.iloc[a], 0), tr.iloc[b][cols])
                    platt, iso = C.Platt().fit(oof, tr["y"]), C.Isotonic().fit(oof, tr["y"])
                    # course-disjoint out-of-fold scores inside training (mimic an unseen course)
                    cd = np.full(len(tr), np.nan)
                    for a, b in P.inner_module_folds(tr, 3):
                        cd[b] = M.predict(fit(spec, cfg, cols, tr.iloc[a], 0), tr.iloc[b][cols])
                    platt_cd = C.Platt().fit(cd, tr["y"])
                    oofdf = tr[["presentation", "code_module", "id_student", "y"]].copy()
                    oofdf["fold"] = fold; oofdf["oof"] = oof; oofdf["oof_cd"] = cd
                    oof_rows.append(oofdf)
                    raw, t_fit, t_inf = [], [], []
                    for s in range(seeds if spec["model"] not in ("lr",) else 1):
                        t1 = time.time(); mdl = fit(spec, cfg, cols, tr, s); t_fit.append(time.time() - t1)
                        t1 = time.time(); raw.append(M.predict(mdl, te[cols])); t_inf.append(time.time() - t1)
                    raw = np.array(raw)
                    pm = raw.mean(0)
                    p_platt = platt(pm)
                    # cohort prior (ours) and EM, per target presentation
                    trp = tr.groupby("presentation")["y"].mean()
                    cp = C.CohortPrior().fit(stats.loc[trp.index], trp)
                    pi_hat = cp.predict(stats.loc[te["presentation"].unique()])
                    p_cohort = p_platt.copy(); p_em = p_platt.copy()
                    for pres, ix in te.groupby("presentation").indices.items():
                        p_cohort[ix] = C.shift_to_mean(p_platt[ix], pi_hat[pres])
                        p_em[ix] = C.em_prior(p_platt[ix], tr["y"].mean())
                    keep = ["code_module", "code_presentation", "id_student", "presentation", "y", "final_result",
                            "gender", "disability", "imd_band", "age_band"]
                    df = te[keep].copy()
                    df["fold"] = fold
                    for s in range(len(raw)):
                        df[f"raw_s{s}"] = raw[s]
                    df["raw"], df["platt"], df["iso"], df["cohort"], df["em"] = pm, p_platt, iso(pm), p_cohort, p_em
                    df["platt_cd"] = platt_cd(pm)
                    df["pi_hat"] = te["presentation"].map(pi_hat).to_numpy()
                    rows.append(df)
                    log.append(dict(fold=fold, n_train=len(tr), n_test=len(te), set_index=si, cfg=cfg, inner_auroc=inner,
                                    t_select=t_sel, t_fit_mean=float(np.mean(t_fit)), t_infer_mean=float(np.mean(t_inf)),
                                    n_features=len(cols)))
                    print(f"d{d} {proto} {meth} {fold}: AUROC raw={roc_auc_score(te['y'], pm):.4f} set={si} cfg={cfg} "
                          f"fit={np.mean(t_fit):.1f}s", flush=True)
                pd.concat(rows).to_parquet(out)
                OOF_DIR.mkdir(parents=True, exist_ok=True)
                pd.concat(oof_rows).to_parquet(OOF_DIR / out.name)
                json.dump(log, open(out.with_suffix(".json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", default=[7, 14, 28, 42])
    ap.add_argument("--protocols", nargs="+", default=["P1", "P2", "P3"])
    ap.add_argument("--methods", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    run(a.days, a.protocols, a.methods, a.seeds)
