"""Derived numbers quoted in the paper (differences and ranges over days) -> extra_numbers.json (merged).
Usage: python experiments/derived_numbers.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "extra_numbers.json"
nums = json.load(open(OUT)) if OUT.exists() else {}


def and_join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else "none"

M = pd.read_parquet(ROOT / "results" / "eval" / "metrics.parquet")
P = M[(M.proto == "P1") & (M.calib == "s_platt")].pivot_table(index="method", columns="day", values="auroc")
E = M[(M.proto == "P1") & (M.calib == "s_platt")].pivot_table(index="method", columns="day", values="ece")
days = [7, 14, 28, 42]
for fam, (a, r) in {"lgbm": ("lgbm_abs", "crew"), "gru": ("gru_abs", "gru_rel")}.items():
    d = (P.loc[r, days] - P.loc[a, days])
    nums[f"rel.{fam}.dmin"], nums[f"rel.{fam}.dmax"] = f"{d.min():.3f}", f"{d.max():.3f}"
    nums[f"rel.{fam}.allpos"] = str(bool((d > 0).all()))
# best model per day and whether it is cohort-relative
rel = {"crew", "gru_rel", "crew_lim", "crew_rel", "crew_noCAL"}
best = {d: P[d].dropna().idxmax() for d in days}
nums["rel.bestauroc.allrel"] = str(all(b in rel for b in best.values()))
bestece = {d: E[d].dropna().idxmin() for d in days}
nums["rel.bestece.allrel"] = str(all(b in rel for b in bestece.values()))
for d in days:
    nums[f"best.d{d}.auroc.method"], nums[f"best.d{d}.ece.method"] = best[d], bestece[d]
# lead of the best cohort-relative model over the best absolute (non-ablation) model, pooled and module-mean AUROC
ABS = ["lim_lr", "lim_rf", "lim_hgb", "lgbm_lim", "lgbm_iw", "lgbm_abs_bal", "cat_abs", "gru_abs", "lgbm_abs"]
REL = ["crew", "gru_rel"]
MA = M[(M.proto == "P1") & (M.calib == "s_platt")].pivot_table(index="method", columns="day", values="mod_auroc")
SD = M[(M.proto == "P1") & (M.calib == "s_platt")].pivot_table(index="method", columns="day", values="seed_auroc_sd")
NAMES = {"cat_abs": "CatBoost", "lgbm_abs": "absolute LightGBM", "gru_abs": "absolute GRU", "lgbm_abs_bal":
         "module-balanced LightGBM", "lim_lr": "LR", "lim_hgb": "HGB", "lim_rf": "RF", "lgbm_lim": "LightGBM (R, V, S)",
         "lgbm_iw": "importance-weighted LightGBM"}
leads, mleads = {}, {}
for d in days:
    for tag, T in (("pooled", P), ("module", MA)):
        a = T.loc[[m for m in ABS if m in T.index], d].dropna()
        r = T.loc[[m for m in REL if m in T.index], d].dropna()
        lead = r.max() - a.max()
        (leads if tag == "pooled" else mleads)[d] = lead
        nums[f"lead.{tag}.d{d}"] = f"{lead:.3f}"
        nums[f"lead.{tag}.d{d}.absbest"] = NAMES[a.idxmax()]
        nums[f"lead.{tag}.d{d}.absbestauroc"] = f"{a.max():.3f}"
        nums[f"lead.{tag}.d{d}.relbest"] = r.idxmax()
nums["lead.pooled.min"], nums["lead.pooled.max"] = f"{min(leads.values()):.3f}", f"{max(leads.values()):.3f}"
nums["lead.module.min"], nums["lead.module.max"] = f"{min(mleads.values()):.3f}", f"{max(mleads.values()):.3f}"
tab = [m for m in SD.index if not m.startswith("gru") and m in ABS + REL and SD.loc[m].notna().any()]
nums["seedsd.tab.max"] = f"{SD.loc[tab, days].max().max():.3f}"
nums["seedsd.gru.max"] = f"{SD.loc[['gru_abs', 'gru_rel'], days].max().max():.3f}"
# pooled AUROC differences that no bootstrap resample reversed (bootstrap p = 0 at B resamples), per day and model
T = pd.read_csv(ROOT / "results" / "analysis" / "tests.csv").query("proto == 'P1' and base_cal == 's_platt'")
for (d, ours), g in T.groupby(["day", "ours"]):
    nums[f"boot.d{d}.{ours}.n"] = str(len(g))
    nums[f"boot.d{d}.{ours}.nzero"] = str(int((g["pboot_auroc"] == 0).sum()))
    nums[f"boot.d{d}.{ours}.allpos"] = str(bool((g["lo_auroc"] > 0).all()))
# ---- RQ2: cohort prior, gate and label-shift estimators (P1)
P1 = M[M.proto == "P1"]
def piv(met, calib):
    return P1[P1.calib == calib].pivot_table(index="method", columns="day", values=met)
absm = [m for m in ABS if m in set(P1.method)]
for met, key in (("ece", "ece"), ("mod_ece", "modece"), ("auroc", "auroc")):
    dp = (piv(met, "s_platt+cprior") - piv(met, "s_platt")).loc[absm]
    dg = (piv(met, "s_gate") - piv(met, "s_platt")).loc[absm]
    for tag, dd in (("prior", dp), ("gate", dg)):
        on = dd[[7, 28, 42]].stack()
        nums[f"{tag}.abs.d7_28_42.{key}.min"], nums[f"{tag}.abs.d7_28_42.{key}.max"] = f"{on.min():.3f}", f"{on.max():.3f}"
        nums[f"{tag}.abs.d14.{key}.min"], nums[f"{tag}.abs.d14.{key}.max"] = f"{dd[14].min():.3f}", f"{dd[14].max():.3f}"
        nums[f"{tag}.abs.all.{key}.min"], nums[f"{tag}.abs.all.{key}.max"] = (f"{dd.stack().min():.3f}",
                                                                              f"{dd.stack().max():.3f}")
    allm = (piv(met, "s_platt+cprior") - piv(met, "s_platt")).loc[absm + REL, 14]
    nums[f"prior.all.d14.{key}.min"], nums[f"prior.all.d14.{key}.max"] = f"{allm.min():.3f}", f"{allm.max():.3f}"
    nums[f"prior.all.d14.{key}.n"] = str(len(allm))
    if key == "auroc":
        nums["prior.all.d14.auroc.dropmin"], nums["prior.all.d14.auroc.dropmax"] = f"{-allm.max():.3f}", f"{-allm.min():.3f}"
        nums["prior.all.d14.auroc.alllower"] = str(bool((allm < 0).all()))
nums["prior.abs.n"] = str(len(absm))
C = pd.read_parquet(ROOT / "results" / "eval" / "choices.parquet").query("proto == 'P1'")
C = C.assign(on=(C["gate"] == "platt+cprior").astype(int))
nums["gate.on.d14.total"] = str(int(C.query("day == 14")["on"].sum()))
nums["gate.folds.d14.total"] = str(int(len(C[(C["day"] == 14) & C["method"].isin(absm + REL)])))
for m in REL + ["lim_lr"]:
    cm = C.query("method == @m")
    nums[f"gate.on.{m}"], nums[f"gate.folds.{m}"] = str(int(cm["on"].sum())), str(len(cm))
# models for which the gate applied the prior only in the AAA fold on day 28, and what that cost in AUROC
on28 = C.query("day == 28 and on == 1").groupby("method")["fold"].apply(list)
aaa_only = [m for m, f in on28.items() if f == ["LOMO:AAA"] and m in absm]
dg28 = (piv("auroc", "s_platt") - piv("auroc", "s_gate")).loc[aaa_only, 28]
nums["gate.d28.aaaonly.n"] = str(len(aaa_only))
nums["gate.d28.aaaonly.dropmin"], nums["gate.d28.aaaonly.dropmax"] = f"{dg28.min():.3f}", f"{dg28.max():.3f}"
cr = C.query("method in @REL")
nums["gate.on.rel"], nums["gate.folds.rel"] = str(int(cr["on"].sum())), str(len(cr))
# base-rate error per presentation (mean predicted probability against the true rate), day 28
PP = pd.read_parquet(ROOT / "results" / "analysis" / "per_presentation.parquet").query("proto == 'P1'")
CAL = {"s_platt": "platt", "s_platt+cprior": "prior", "s_gate": "gate", "s_platt+em": "em", "s_platt+acc": "acc"}
for d in days:
    for m in ("lim_lr", "crew", "cat_abs", "gru_rel"):
        for c, cn in CAL.items():
            g = PP.query("day == @d and method == @m and calib == @c")
            if len(g):
                nums[f"prev.d{d}.{m}.{cn}.mae"] = f"{g['prev_err'].mean():.3f}"
                nums[f"prev.d{d}.{m}.{cn}.max"] = f"{g['prev_err'].max():.3f}"
                nums[f"prev.d{d}.{m}.{cn}.corr"] = f"{np.corrcoef(g['prev'], g['mean_p'])[0, 1]:.2f}"
                nums[f"prev.d{d}.{m}.{cn}.estmin"] = f"{g['mean_p'].min():.2f}"
                nums[f"prev.d{d}.{m}.{cn}.estmax"] = f"{g['mean_p'].max():.2f}"
    a = PP.query("day == @d and method == 'lim_lr' and calib == 's_platt+cprior' and code_module == 'AAA'")
    nums[f"prior.aaa.d{d}.est"], nums[f"prior.aaa.d{d}.true"] = f"{a['mean_p'].mean():.2f}", f"{a['prev'].mean():.2f}"
    em = PP.query("day == @d and method == 'lim_lr' and calib in ['s_platt+em', 's_platt+acc']")
    nums[f"prev.d{d}.lim_lr.emacc.extreme"] = str(int(((em["mean_p"] < 0.05) | (em["mean_p"] > 0.95)).sum()))
    nums[f"prev.d{d}.lim_lr.emacc.n"] = str(len(em))
# per-module ECE of CREW against the benchmark (same Platt calibration), day 28
PMod = pd.read_parquet(ROOT / "results" / "eval" / "per_module.parquet").query("proto == 'P1' and calib == 's_platt'")
for d in days:
    e = PMod.query("day == @d").pivot_table(index="code_module", columns="method", values="ece")
    nums[f"mod.d{d}.crew_vs_lr.lower"] = str(int((e["crew"] < e["lim_lr"]).sum()))
    nums[f"mod.d{d}.crew_vs_lr.higher"] = and_join(e.index[e["crew"] > e["lim_lr"]])
    nums[f"mod.d{d}.crew.worst"] = e["crew"].idxmax()
    for m in [x for x in ("crew", "lim_lr", "cat_abs", "gru_rel", "lgbm_abs_bal") if x in e.columns]:
        for mo in e.index:
            nums[f"mod.d{d}.{m}.{mo}.ece"] = f"{e.loc[mo, m]:.3f}"
    best_max = M.query("proto == 'P1' and day == @d and calib == 's_platt' and method in @absm").reset_index(drop=True)
    nums[f"mod.d{d}.absbest.maxece"] = f"{best_max['mod_ece_max'].min():.3f}"
    nums[f"mod.d{d}.absbest.maxece.method"] = NAMES[best_max.loc[best_max['mod_ece_max'].idxmin(), 'method']]
# ---- RQ3 boundaries: forward protocol P2 (2013 -> 2014); CCC has no 2013 presentation, so it is the unseen course
PM2 = pd.read_parquet(ROOT / "results" / "eval" / "per_module.parquet").query("proto == 'P2' and calib == 's_platt'")
ccc_best = []
for d in days:
    e = PM2.query("day == @d and code_module == 'CCC'").set_index("method")["auroc"]
    if len(e):
        nums[f"p2.d{d}.ccc.crew"] = f"{e['crew']:.3f}"
        other = e.drop("crew")
        nums[f"p2.d{d}.ccc.bestother"] = f"{other.max():.3f}"
        ccc_best.append(bool(e.idxmax() == "crew"))
nums["p2.ccc.crewbest.alldays"] = str(all(ccc_best) and len(ccc_best) == 4)
P2 = M[(M.proto == "P2") & (M.calib == "s_platt")].pivot_table(index="method", columns="day", values="auroc")
nums["p2.n.methods"] = str(len(P2.index))
for d in days:
    nums[f"p2.d{d}.best"] = P2[d].idxmax()
# modules in which CREW ranked better than the benchmark and than CatBoost (P1, Platt)
PM1 = pd.read_parquet(ROOT / "results" / "eval" / "per_module.parquet").query("proto == 'P1' and calib == 's_platt'")
for d in days:
    e = PM1.query("day == @d").pivot_table(index="code_module", columns="method", values="auroc")
    nums[f"modauroc.d{d}.crew_gt_lr"] = str(int((e["crew"] > e["lim_lr"]).sum()))
    nums[f"modauroc.d{d}.crew_gt_cat"] = str(int((e["crew"] > e["cat_abs"]).sum()))
    nums[f"modauroc.d{d}.crew_le_cat"] = and_join(e.index[e["crew"] <= e["cat_abs"]])
# ---- learner subgroups (P1, Platt): results/analysis/subgroups.csv from experiments/subgroups.py
SG = ROOT / "results" / "analysis" / "subgroups.csv"
if SG.exists():
    S = pd.read_csv(SG)
    best = (S.auroc_crew > S.auroc_cat_abs) & (S.auroc_crew > S.auroc_lim_lr)
    nums["sub.all.crewbest"], nums["sub.all.n"] = str(int(best.sum())), str(len(S))
    s28 = S[S.day == 28]
    nums["sub.d28.n"] = str(len(s28))
    nums["sub.d28.crewbest"] = str(int(((s28.auroc_crew > s28.auroc_cat_abs) & (s28.auroc_crew > s28.auroc_lim_lr)).sum()))
    nums["sub.d28.cicat.pos"] = str(int((s28.lo_auroc_vs_cat_abs > 0).sum()))
    nums["sub.d28.ece.lowerlr"] = str(int((s28.ece_crew < s28.ece_lim_lr).sum()))
    for (attr, g), r in s28.set_index(["attr", "group"]).iterrows():
        k = f"sub.d28.{attr}.{g.replace(' %', '').replace(' ', '')}"
        nums[f"{k}.n"] = f"{int(r['n']):,}"
        nums[f"{k}.prev"] = f"{r['prev']:.2f}"
        for m in ("lim_lr", "cat_abs", "crew"):
            nums[f"{k}.{m}.auroc"], nums[f"{k}.{m}.ece"] = f"{r[f'auroc_{m}']:.3f}", f"{r[f'ece_{m}']:.3f}"
            nums[f"{k}.{m}.bias"] = f"{r[f'bias_{m}']:+.3f}"
        nums[f"{k}.cat.lo"], nums[f"{k}.cat.hi"] = f"{r['lo_auroc_vs_cat_abs']:.3f}", f"{r['hi_auroc_vs_cat_abs']:.3f}"
        nums[f"{k}.lrece.lo"], nums[f"{k}.lrece.hi"] = f"{r['lo_ece_vs_lim_lr']:.3f}", f"{r['hi_ece_vs_lim_lr']:.3f}"
    for attr, g in s28.groupby("attr"):
        for m in ("lim_lr", "cat_abs", "crew"):
            nums[f"sub.d28.gap.{attr}.{m}"] = f"{g[f'auroc_{m}'].max() - g[f'auroc_{m}'].min():.3f}"
# ---- feature set chosen by the nested selection of CREW (0: REL only, 1: REL with the absolute families)
tot = 0
for d in days:
    lg = json.load(open(ROOT / "results" / "preds" / f"P1_d{d}_crew.json"))
    k = sum(int(x.get("set_index", 0) == 1) for x in lg)
    nums[f"crew.sets.d{d}.relabs"], nums[f"crew.sets.d{d}.n"] = str(k), str(len(lg))
    tot += k
nums["crew.sets.all.relabs"], nums["crew.sets.all.n"] = str(tot), str(7 * len(days))
AB = ROOT / "results" / "analysis" / "ablation_d28.csv"
if AB.exists():
    A = pd.read_csv(AB).set_index("method")
    for m, r in A.iterrows():
        for k in ("auroc", "ece", "mod_auroc", "mod_ece"):
            nums[f"abl.d28.{m}.{k.replace('_', '')}"] = f"{r[k]:.3f}"
        if pd.notna(r.get("d_auroc", float("nan"))):
            for k in ("d_auroc", "lo_auroc", "hi_auroc", "d_ece", "lo_ece", "hi_ece"):
                nums[f"abl.d28.{m}.{k.replace('_', '')}"] = f"{r[k]:.3f}"
            nums[f"abl.d28.{m}.negdauroc"] = f"{-r['d_auroc']:.3f}"
    AX = ROOT / "results" / "analysis" / "ablation_extra_d28.csv"
    if AX.exists():
        for _, r in pd.read_csv(AX).iterrows():
            for k in ("d_auroc", "lo_auroc", "hi_auroc", "d_ece", "lo_ece", "hi_ece"):
                nums[f"ablx.d28.{r['method']}.{r['ref']}.{k.replace('_', '')}"] = f"{r[k]:.3f}"
# ---- cost benchmark (experiments/cost_bench.py): one median-sized held-out module, idle machine
CB = ROOT / "results" / "analysis" / "cost_bench.csv"
if CB.exists():
    Cb = pd.read_csv(CB).set_index("method")
    for m in [x for x in Cb.index if not x.startswith("_")]:
        r = Cb.loc[m]
        nums[f"cost.{m}.total"] = f"{r['t_total']:.0f}"
        nums[f"cost.{m}.select"] = f"{r['t_select']:.0f}"
        nums[f"cost.{m}.fitone"] = f"{r['t_fit_one']:.1f}"
        nums[f"cost.{m}.inferms"] = f"{1000 * r['t_infer_per_1000']:.0f}"
        nums[f"cost.{m}.mb"] = f"{r['peak_mb']:.0f}"
        nums[f"cost.{m}.nfeat"] = f"{int(r['n_features'])}"
    f = Cb.loc["_features_and_prior"]
    nums["cost.features.s"], nums["cost.sequences.s"] = f"{f['t_features']:.1f}", f"{f['t_sequences']:.1f}"
    nums["cost.prior.s"] = f"{f['t_cohort_stats'] + f['t_prior_all_presentations']:.2f}"
    nums["cost.features.n"] = f"{int(f['n_test']):,}"
    tab = [x for x in Cb.index if not x.startswith(("_", "gru"))]
    nums["cost.maxmb.tab"] = f"{Cb.loc[tab, 'peak_mb'].max():.0f}"
    nums["cost.maxmb.all"] = f"{Cb['peak_mb'].max() / 1024:.1f}"
    nums["cost.ccc.ntrain"] = f"{int(Cb.loc['crew', 'n_train']):,}"
    nums["cost.ccc.ntest"] = f"{int(Cb.loc['crew', 'n_test']):,}"
    nums["cost.crew.ratio.lgbm"] = f"{Cb.loc['crew', 't_total'] / Cb.loc['lgbm_abs', 't_total']:.1f}"
# ---- attribution stability (experiments/explain.py): rank agreement and spread of shares across held-out modules
SC = ROOT / "results" / "analysis" / "shap_compare_d28.json"
if SC.exists():
    sc = json.load(open(SC))
    nums["shap.d28.nbase"] = str(sc["n_common_base"])
    W = {}
    for m in ("crew", "lgbm_abs"):
        nums[f"shap.d28.{m}.tau"] = f"{sc[m]['kendall_tau_mean']:.2f}"
        nums[f"shap.d28.{m}.taumin"] = f"{sc[m]['kendall_tau_min']:.2f}"
        nums[f"shap.d28.{m}.jac"] = f"{sc[m]['top10_jaccard_mean']:.2f}"
        nums[f"shap.d28.{m}.top1missed"] = str(sc[m]["top1"].get("n_missed", 0))
        sh = pd.read_csv(ROOT / "results" / "analysis" / f"shap_fold_d28_{m}.csv")
        sh["base"] = sh["feature"].map(lambda c: c[2:] if c[:2] in ("r_", "z_") else c)
        w = sh.pivot_table(index="base", columns="module", values="mean_abs_shap", aggfunc="sum").fillna(0)
        W[m] = w.div(w.sum(axis=0), axis=1)
    top10 = W["crew"].mean(axis=1).nlargest(10).index
    for m, w in W.items():
        rng = (w.max(axis=1) - w.min(axis=1)).reindex(top10).fillna(0)
        nums[f"shap.d28.{m}.rangetop10"] = f"{rng.mean():.3f}"
        nums[f"shap.d28.{m}.range.missed"] = f"{rng['n_missed']:.3f}"
        nums[f"shap.d28.{m}.range.early"] = f"{rng['mean_days_early']:.3f}"
        nums[f"shap.d28.{m}.share.missed"] = f"{w.loc['n_missed'].mean():.3f}"
    st = json.load(open(ROOT / "results" / "analysis" / "shap_stability_d28_crew.json"))
    nums["shap.d28.crew.relshare"] = f"{100 * st['family_share_mean'].get('REL', 0):.0f}"
    f28 = pd.read_csv(ROOT / "results" / "analysis" / "shap_fold_d28_crew.csv")
    top = f28.loc[f28.groupby("module")["mean_abs_shap"].idxmax(), "feature"]
    nums["shap.d28.crew.top1.rmissed"] = str(int((top == "r_n_missed").sum()))
# ---- practical reading of pooled recall at 20 % on day 28: additional at-risk learners on the same contact list
R28 = M.query("proto == 'P1' and day == 28 and calib == 's_platt'").set_index("method")
npos = int(R28.loc["crew", "n"] * R28.loc["crew", "prev"] + 0.5) if "prev" in R28.columns else None
if npos:
    nums["practical.d28.npos"] = f"{npos:,}"
    for m in ("lim_lr", "cat_abs"):
        nums[f"practical.d28.extra.{m}"] = f"{(R28.loc['crew', 'rec20'] - R28.loc[m, 'rec20']) * npos:,.0f}"
    nums["practical.d28.ncontact"] = f"{int(round(0.2 * R28.loc['crew', 'n'])):,}"
# ---- presentation sizes of the day-28 risk sets (limitations)
F28 = pd.read_parquet(ROOT / "results" / "features" / "feat_d28.parquet").groupby("presentation").size()
nums["data.d28.presmin"], nums["data.d28.presmax"] = f"{F28.min():,}", f"{F28.max():,}"
nums["data.d28.presminname"] = F28.idxmin().replace("_", "\\_")
# number of training presentations available to the cohort prior per held-out module (P1)
npres = F28.groupby(F28.index.str[:3]).size()
nums["prior.ntrainpres.max"], nums["prior.ntrainpres.min"] = str(int(len(F28) - npres.min())), str(int(len(F28) - npres.max()))
# gated prior never lowers the benchmark's AUROC
G = M[(M.proto == "P1") & (M.method == "lim_lr")].pivot_table(index="calib", columns="day", values="auroc")
nums["prior.gate.limlr.neverlower"] = str(bool((G.loc["s_gate"] >= G.loc["s_platt"] - 1e-9).all()))
json.dump(nums, open(OUT, "w"), indent=1, sort_keys=True)
print({k: v for k, v in nums.items() if k.startswith(("rel.", "best.", "prior."))})
