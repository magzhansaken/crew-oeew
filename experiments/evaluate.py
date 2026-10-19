"""Post-hoc evaluation: prevalence estimators, nested choice of calibration, metrics per protocol/day/method.

For every outer fold only training information is used to fit calibrators and prevalence models:
Base calibrators (fitted on training out-of-fold scores):
- platt : Platt on student-grouped OOF scores (Lim, 2026)
- cd    : Platt on course-disjoint OOF scores (training modules held out in groups; ours)
Prevalence adjustments per target presentation, on top of either base:
- em    : Saerens et al. (2002) EM;  acc: adjusted classify-and-count / BBSE (Forman, 2008; Lipton et al., 2018)
- cprior: cohort prior from label-free cohort statistics (ours);  cpcc: cohort prior that also uses the mean score
- oracle: true target prevalence (upper bound only; uses target labels; never a method)
auto: the scheme with the lowest inner per-presentation ECE on course-disjoint validation groups (nested)
gate: Platt with or without the cohort prior, whichever has the lower inner per-presentation ECE (nested)
Usage: python experiments/evaluate.py  -> results/eval/metrics.parquet, results/eval/per_module.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oeew import calibration as C, data, metrics as Mx, protocols as P  # noqa: E402

PRED, OOF, OUT = ROOT / "results" / "preds", ROOT / "results" / "oof", ROOT / "results" / "eval"
BASES = {"platt": "oof", "cd": "oof_cd"}       # calibrator fitted on student-grouped or course-disjoint OOF scores
ADJ = ["none", "em", "acc", "cprior", "cpcc"]   # prevalence adjustment applied per target presentation
SCHEMES = [b if a == "none" else f"{b}+{a}" for b in BASES for a in ADJ]


def acc_prev(p_src, y_src, p_tgt, thr=0.5):
    yh_s = p_src > thr
    tpr = yh_s[y_src == 1].mean(); fpr = yh_s[y_src == 0].mean()
    q = (np.mean(p_tgt > thr) - fpr) / max(tpr - fpr, 1e-3)
    return float(np.clip(q, 0.01, 0.99))


def fit_base(o: pd.DataFrame, base: str, stats: pd.DataFrame) -> dict:
    """Calibrator and prevalence models fitted on training rows o only (labels of o are training labels)."""
    col = BASES[base]
    pl = C.Platt().fit(o[col].to_numpy(), o["y"].to_numpy())
    p_src = pl(o[col].to_numpy())
    prev = o.groupby("presentation")["y"].mean()
    pcc = pd.Series(p_src, index=o.index).groupby(o["presentation"]).mean()
    cp = C.CohortPrior().fit(stats.loc[prev.index], prev)
    st2 = stats.loc[prev.index].assign(pcc=C.logit(pcc.loc[prev.index].to_numpy()))
    cp2 = C.CohortPrior().fit(st2, prev)
    return dict(pl=pl, p_src=p_src, y_src=o["y"].to_numpy(), cp=cp, cp2=cp2, stats=stats)


def adjust(adj: str, p_cal: np.ndarray, pres: np.ndarray, ctx: dict) -> np.ndarray:
    """Shift calibrated probabilities of each target presentation to an estimated prevalence (logit shift)."""
    out = p_cal.copy()
    if adj == "none":
        return out
    for pr in np.unique(pres):
        ix = pres == pr
        if adj == "em":
            out[ix] = C.em_prior(p_cal[ix], ctx["y_src"].mean()); continue
        if adj == "acc":
            q = acc_prev(ctx["p_src"], ctx["y_src"], p_cal[ix])
        elif adj == "cprior":
            q = float(ctx["cp"].predict(ctx["stats"].loc[[pr]]).iloc[0])
        elif adj == "cpcc":
            st = ctx["stats"].loc[[pr]].assign(pcc=C.logit(np.array([p_cal[ix].mean()])))
            q = float(ctx["cp2"].predict(st).iloc[0])
        elif adj == "oracle":
            q = ctx["true_prev"][pr]
        else:
            raise ValueError(adj)
        out[ix] = C.shift_to_mean(p_cal[ix], q)
    return out


def split_scheme(s):
    b, _, a = s.partition("+")
    return b, (a or "none")


def inner_choice(oof, stats):
    """Choose a scheme by mean per-presentation ECE on inner course-disjoint validation groups of the training data.
    Calibrators are fitted on the other inner groups; validation scores are course-disjoint OOF scores."""
    mods = sorted(oof["code_module"].unique())
    rng = np.random.default_rng(0); rng.shuffle(mods)
    groups = [mods[i::3] for i in range(3)]
    eces = {s: [] for s in SCHEMES}
    for g in groups:
        va = oof["code_module"].isin(g).to_numpy()
        o_tr, o_va = oof[~va], oof[va]
        for base in BASES:
            ctx = fit_base(o_tr, base, stats)
            pv = ctx["pl"](o_va["oof_cd"].to_numpy())
            for adj in ADJ:
                s = base if adj == "none" else f"{base}+{adj}"
                q = adjust(adj, pv, o_va["presentation"].to_numpy(), ctx)
                for pr, ix in o_va.groupby("presentation").indices.items():
                    if o_va["y"].iloc[ix].nunique() > 1:
                        eces[s].append(Mx.ece(o_va["y"].to_numpy()[ix], q[ix]))
    mean = {s: float(np.mean(v)) for s, v in eces.items()}
    return min(mean, key=mean.get), mean


def evaluate_file(fp: Path, stats_by_day: dict):
    proto, dtag, meth = fp.stem.split("_", 2)
    d = int(dtag[1:])
    df = pd.read_parquet(fp)
    oof_fp = OOF / fp.name
    stats = stats_by_day[d]
    choices = []
    if oof_fp.exists():
        oof = pd.read_parquet(oof_fp)
        parts = []
        for fold, te in df.groupby("fold", sort=False):
            o = oof[oof["fold"] == fold]
            te = te.copy()
            true_prev = te.groupby("presentation")["y"].mean().to_dict()
            for base in BASES:
                ctx = fit_base(o, base, stats) | dict(true_prev=true_prev)
                pc = ctx["pl"](te["raw"].to_numpy())
                for adj in ADJ + ["oracle"]:
                    name = base if adj == "none" else f"{base}+{adj}"
                    te["s_" + name] = adjust(adj, pc, te["presentation"].to_numpy(), ctx)
                for k in [c for c in te.columns if c.startswith("raw_s")]:  # same calibrator applied to each seed
                    te[f"seed_{base}_{k[4:]}"] = ctx["pl"](te[k].to_numpy())
            ch, inner = inner_choice(o, stats)
            te["s_auto"] = te["s_" + ch]
            # gated prior: add the cohort prior only if it lowers inner per-presentation ECE (nested, training only)
            gate = "platt+cprior" if inner["platt+cprior"] < inner["platt"] else "platt"
            te["s_gate"] = te["s_" + gate]
            for k in [c for c in te.columns if c.startswith("raw_s")]:
                b = split_scheme(ch)[0]
                te[f"seed_auto_{k[4:]}"] = te[f"seed_{b}_{k[4:]}"] if split_scheme(ch)[1] == "none" else np.nan
            choices.append(dict(fold=fold, choice=ch, gate=gate, **{f"inner_ece_{k}": v for k, v in inner.items()}))
            parts.append(te)
        df = pd.concat(parts)
    else:  # sequence models: calibrator fitted on a course-disjoint validation group (see run_gru.py)
        cd = df["platt_cd"] if "platt_cd" in df.columns else df["platt"]
        df = df.assign(s_cd=cd, s_platt=df["platt"], s_auto=cd)
    cols = ["raw"] + [c for c in df.columns if c.startswith("s_")]
    rows, pm = [], []
    for c in cols:
        m = Mx.all_metrics(df["y"], df[c], df["presentation"])
        mod = df.groupby("code_module").apply(lambda x: pd.Series(Mx.all_metrics(x["y"], x[c])), include_groups=False)
        m.update(dict(proto=proto, day=d, method=meth, calib=c, mod_auroc=mod["auroc"].mean(), mod_ece=mod["ece"].mean(),
                      mod_ece_max=mod["ece"].max(), mod_auprc=mod["auprc"].mean(), mod_brier=mod["brier"].mean()))
        rows.append(m)
        pm.append(mod.reset_index().assign(proto=proto, day=d, method=meth, calib=c))
    for c in ("platt", "cd", "auto"):  # spread over seeds (each seed calibrated by the same fold calibrator)
        seeds = [x for x in df.columns if x.startswith(f"seed_{c}_") and df[x].notna().all()]
        if len(seeds) > 1:
            sm = [Mx.all_metrics(df["y"], df[s]) for s in seeds]
            modau = [df.groupby("code_module").apply(lambda x, s=s: Mx.all_metrics(x["y"], x[s])["auroc"],
                                                      include_groups=False).mean() for s in seeds]
            spread = dict(n_seeds=len(seeds), seed_mod_auroc_mean=float(np.mean(modau)),
                          seed_mod_auroc_sd=float(np.std(modau, ddof=1)),
                          **{f"seed_{k}_{f}": fn([x[k] for x in sm]) for k in ("auroc", "auprc", "brier", "ece")
                             for f, fn in (("mean", np.mean), ("sd", lambda v: float(np.std(v, ddof=1))))})
            for r in rows:
                if r["calib"] == "s_" + c:
                    r.update(spread)
    return pd.DataFrame(rows), pd.concat(pm), pd.DataFrame(choices).assign(proto=proto, day=d, method=meth), df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = data.load(data.download(ROOT / "data"))
    base = data.base_table(t)
    stats_by_day = {}
    for fp in sorted((ROOT / "results" / "features").glob("feat_d*.parquet")):
        d = int(fp.stem.split("_d")[1])
        stats_by_day[d] = C.cohort_stats(pd.read_parquet(fp), base, d)
    R, PM, CH = [], [], []
    (OUT / "rows").mkdir(exist_ok=True)
    cache = OUT / "files"; cache.mkdir(exist_ok=True)
    live = {fp.stem for fp in PRED.glob("*.parquet")}
    for stale in list((OUT / "rows").glob("*.parquet")) + list(cache.glob("*.parquet")):
        if stale.name.split(".")[0] not in live:   # predictions that were removed or re-run elsewhere
            stale.unlink()
    for fp in sorted(PRED.glob("*.parquet")):
        c = [cache / f"{fp.stem}.{k}.parquet" for k in ("m", "pm", "ch")]
        oof_fp = OOF / fp.name
        newest = max(fp.stat().st_mtime, oof_fp.stat().st_mtime if oof_fp.exists() else 0, Path(__file__).stat().st_mtime)
        if all(x.exists() for x in c) and min(x.stat().st_mtime for x in c) > newest:
            r, pm, ch = (pd.read_parquet(x) for x in c)
        else:
            r, pm, ch, df = evaluate_file(fp, stats_by_day)
            df.to_parquet(OUT / "rows" / fp.name)
            for x, v in zip(c, (r, pm, ch)):
                v.to_parquet(x)
            print(fp.stem, "done", flush=True)
        R.append(r); PM.append(pm); CH.append(ch)
    pd.concat(R).to_parquet(OUT / "metrics.parquet")
    pd.concat(PM).to_parquet(OUT / "per_module.parquet")
    pd.concat(CH).to_parquet(OUT / "choices.parquet")


if __name__ == "__main__":
    main()
