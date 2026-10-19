"""SHAP attributions of the proposed model under P1 and their stability across held-out modules.

For each outer fold the model is refitted with the configuration chosen by the nested selection (read from the run
log), and TreeSHAP values are computed on up to 2,000 test enrolments of the held-out module.
Outputs: results/analysis/shap_fold.csv (mean |SHAP| per feature and fold), shap_stability.json
Usage: python experiments/explain.py --day 28 --method crew
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from oeew import models as M, protocols as P  # noqa: E402
from run_tabular import fit, method_specs  # noqa: E402

OUT = ROOT / "results" / "analysis"


def family_of(col: str, fam: dict) -> str:
    for k, v in fam.items():
        if col in v:
            return k if k != "REL" else "REL"
    return "?"


def main(day: int, method: str, n_rows: int = 2000):
    OUT.mkdir(parents=True, exist_ok=True)
    X = pd.read_parquet(ROOT / "results" / "features" / f"feat_d{day}.parquet")
    fam = json.load(open(ROOT / "results" / "features" / f"fam_d{day}.json"))
    spec = method_specs(fam)[method]
    log = {r["fold"]: r for r in json.load(open(ROOT / "results" / "preds" / f"P1_d{day}_{method}.json"))}
    rows = []
    for fold, tri, tei in P.p1_lomo(X):
        tr, te = X.iloc[tri].reset_index(drop=True), X.iloc[tei].reset_index(drop=True)
        si, cfg = log[fold]["set_index"], log[fold]["cfg"]
        cols = spec["sets"][si]
        mdl = fit(spec, cfg, cols, tr, 0)
        sample = te.sample(min(n_rows, len(te)), random_state=0)
        sv = shap.TreeExplainer(mdl).shap_values(sample[cols])
        sv = sv[1] if isinstance(sv, list) else sv
        imp = np.abs(sv).mean(0)
        for c, v in zip(cols, imp):
            rows.append(dict(fold=fold, module=fold.split(":")[1], feature=c, mean_abs_shap=float(v), set_index=si))
        print(fold, "set", si, "top:", [cols[i] for i in np.argsort(-imp)[:5]], flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / f"shap_fold_d{day}_{method}.csv", index=False)
    # stability: rank agreement of feature importances between every pair of held-out modules (common features)
    W = S.pivot_table(index="feature", columns="module", values="mean_abs_shap")
    common = W.dropna()
    taus = [kendalltau(common[a], common[b]).statistic for a, b in itertools.combinations(common.columns, 2)]
    rhos = [spearmanr(common[a], common[b]).statistic for a, b in itertools.combinations(common.columns, 2)]
    top10 = {m: set(common[m].nlargest(10).index) for m in common.columns}
    jac = [len(top10[a] & top10[b]) / len(top10[a] | top10[b]) for a, b in itertools.combinations(common.columns, 2)]
    share = S.assign(family=S["feature"].map(lambda c: "REL" if c in fam["REL"] else family_of(c, {k: v for k, v in fam.items() if k != "REL"})))
    fam_share = (share.groupby(["module", "family"])["mean_abs_shap"].sum() /
                 share.groupby("module")["mean_abs_shap"].sum()).unstack().fillna(0)
    res = dict(day=day, method=method, n_common_features=int(len(common)),
               kendall_tau_mean=float(np.mean(taus)), kendall_tau_min=float(np.min(taus)),
               spearman_mean=float(np.mean(rhos)), top10_jaccard_mean=float(np.mean(jac)),
               top_features=common.mean(1).nlargest(12).round(5).to_dict(),
               family_share_mean=fam_share.mean().round(4).to_dict())
    json.dump(res, open(OUT / f"shap_stability_d{day}_{method}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def base_name(col: str) -> str:
    """Base feature of a cohort-relative column (r_x: percentile rank of x, z_x: standard score of x)."""
    return col[2:] if col[:2] in ("r_", "z_") else col


def compare(day: int, methods: list[str]):
    """Stability of attributions at the level of base features, so that models with and without the relative
    encoding are compared on the same units: |SHAP| of r_x and z_x is summed into x, then ranks are compared
    between every pair of held-out modules on the base features that all compared models share."""
    W = {}
    for m in methods:
        S = pd.read_csv(OUT / f"shap_fold_d{day}_{m}.csv").assign(base=lambda d: d["feature"].map(base_name))
        W[m] = S.pivot_table(index="base", columns="module", values="mean_abs_shap", aggfunc="sum").dropna()
    common = sorted(set.intersection(*[set(w.index) for w in W.values()]))
    res = dict(day=day, n_common_base=len(common))
    for m, w in W.items():
        w = w.loc[common]
        pairs = list(itertools.combinations(w.columns, 2))
        top = {c: set(w[c].nlargest(10).index) for c in w.columns}
        res[m] = dict(kendall_tau_mean=float(np.mean([kendalltau(w[a], w[b]).statistic for a, b in pairs])),
                      kendall_tau_min=float(np.min([kendalltau(w[a], w[b]).statistic for a, b in pairs])),
                      top10_jaccard_mean=float(np.mean([len(top[a] & top[b]) / len(top[a] | top[b]) for a, b in pairs])),
                      top1=w.idxmax().value_counts().to_dict(),
                      top_base=w.div(w.sum(0), axis=1).mean(1).nlargest(8).round(4).to_dict())
    json.dump(res, open(OUT / f"shap_compare_d{day}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=28)
    ap.add_argument("--method", default="crew")
    ap.add_argument("--compare", nargs="+", default=None, help="compare stability of these methods' attributions")
    a = ap.parse_args()
    if a.compare:
        compare(a.day, a.compare)
    else:
        main(a.day, a.method)
