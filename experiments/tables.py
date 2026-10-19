"""LaTeX tables of the paper from results/eval and results/analysis (booktabs; SAI template captions).

Usage: python experiments/tables.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EV, AN = ROOT / "results" / "eval", ROOT / "results" / "analysis"

NAME = {"lim_lr": "LR \\cite{lim2026ijacsa}", "lim_rf": "RF \\cite{lim2026ijacsa}", "lim_hgb": "HGB \\cite{lim2026ijacsa}",
        "lgbm_lim": "LightGBM", "lgbm_iw": "LightGBM, import.\\ weights", "lgbm_abs_bal": "LightGBM, module-bal.",
        "cat_abs": "CatBoost", "gru_abs": "GRU", "gru_rel": "\\textbf{CREW-GRU}", "lgbm_abs": "LightGBM (ablation)",
        "crew_lim": "LightGBM", "crew_rel": "LightGBM", "crew": "\\textbf{CREW}"}
INPUTS = {"lim_lr": "R, V, S", "lim_rf": "R, V, S", "lim_hgb": "R, V, S", "lgbm_lim": "R, V, S", "lgbm_iw": "abs.",
          "lgbm_abs_bal": "abs.", "cat_abs": "abs.", "gru_abs": "seq.", "gru_rel": "seq., rel.",
          "lgbm_abs": "abs.", "crew_lim": "R, V, S, rel.", "crew_rel": "rel.", "crew": "abs., rel."}


def metrics():
    return pd.read_parquet(EV / "metrics.parquet")


def f3(v):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def bold_best(vals, higher=True):
    arr = np.array([np.nan if v is None else v for v in vals], float)
    best = np.nanmax(arr) if higher else np.nanmin(arr)
    return [("\\textbf{" + f3(v) + "}") if (not np.isnan(v) and abs(v - best) < 5e-4) else f3(v) for v in arr]


def table_main(M, out, day=28, alpha=0.05):
    """Main results under P1 on one day: every model with Platt calibration, seed SD of AUROC, significance marks
    (CREW better after Holm, Wilcoxon over held-out presentations) and fit time of one model."""
    order = ["lim_lr", "lim_rf", "lim_hgb", "lgbm_lim", "lgbm_iw", "lgbm_abs_bal", "cat_abs", "gru_abs",
             "lgbm_abs", "gru_rel", "crew"]
    g = M[(M.proto == "P1") & (M.day == day) & (M.calib == "s_platt")].set_index("method")
    order = [m for m in order if m in g.index]
    cols = [("auroc", True), ("auprc", True), ("brier", False), ("ece", False), ("mod_auroc", True), ("mod_ece", False),
            ("mod_ece_max", False), ("rec20", True), ("rec20_within", True)]
    cells = {c: bold_best([g.loc[m, c] for m in order], hi) for c, hi in cols}
    T = pd.read_csv(AN / "tests.csv")
    T = (T.query("proto=='P1' and day==@day and ours=='crew' and base_cal=='s_platt'") if "ours" in T else T.iloc[:0])
    T = T.set_index("baseline")
    cost = pd.read_csv(AN / "cost_bench.csv").set_index("method")  # idle-machine benchmark, as in Table VIII
    star = r"\rlap{\textsuperscript{*}}"
    lines = [r"\begin{table*}[!t]", r"\caption{Results on Held-out Modules, Day %d}\label{tab:main}" % day, r"\centering",
             r"\footnotesize", r"\begin{threeparttable}", r"\setlength{\tabcolsep}{4.2pt}",
             r"\begin{tabular}{@{}llrrrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{4}{c}{Pooled over modules} & \multicolumn{3}{c}{Per module} & \multicolumn{2}{c}{Recall at 20\,\%} & \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}",
             r"Model & Inputs & AUROC & AUPRC & Brier & ECE & AUROC & ECE & max ECE & pooled & in course & Train, s \\",
             r"\midrule",
             r"LR \cite{lim2026ijacsa}, published & R, V, S & 0.661 & 0.594 & 0.237 & 0.071 & -- & -- & 0.357 & -- & -- & -- \\"]
    for i, m in enumerate(order):
        if m == "lgbm_abs":
            lines.append(r"\midrule")
        row = [NAME[m], INPUTS[m]]
        for c, _ in cols:
            v = cells[c][i]
            if c == "auroc" and pd.notna(g.loc[m].get("seed_auroc_sd", np.nan)):
                v += r"{\scriptsize$\pm$%.3f}" % g.loc[m, "seed_auroc_sd"]
            key = {"auroc": "p_auroc_holm", "ece": "p_ece_holm", "brier": "p_brier_holm"}.get(c)
            if key and m != "crew" and m in T.index and pd.notna(T.loc[m, key]) and T.loc[m, key] < alpha:
                v += star
            row.append(v)
        fit = cost.loc[m, "t_total"] if m in cost.index else np.nan
        row.append("--" if np.isnan(fit) else f"{fit:.0f}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{tablenotes}\footnotesize",
              r"\item Every model is calibrated by Platt scaling on learner-grouped out-of-fold training scores. Pooled values"
              r" use all %s held-out enrolments; per-module values are means over the seven modules and the largest module"
              r" ECE. AUROC: mean of five seeds $\pm$ standard deviation over seeds (LR is deterministic). R, V, S:"
              r" registration, VLE and submission families of \cite{lim2026ijacsa}; abs.: all absolute families; seq.:"
              r" daily sequences; rel.: cohort-relative encoding. Bold: best value in a column. *: CREW better"
              r" when the metric is computed within each of the 22 presentations (Wilcoxon, Holm-adjusted $p<0.05$). Train: total training time"
              r" for one new course on two CPU cores (Table~\ref{tab:cost})." % f"{int(g['n'].iloc[0]):,}",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table*}"]
    (out / "tab_main.tex").write_text("\n".join(lines) + "\n")


def table_prior(M, out, day=28):
    """Calibration schemes for every model under P1 on one day: Platt, always-on cohort prior, gated prior, EM,
    ACC/BBSE and the oracle shift. Marks: change against Platt whose 95 % learner-bootstrap interval excludes zero."""
    order = ["lim_lr", "lim_rf", "lim_hgb", "lgbm_lim", "lgbm_iw", "lgbm_abs_bal", "cat_abs", "gru_abs",
             "lgbm_abs", "gru_rel", "crew"]
    g = M[(M.proto == "P1") & (M.day == day)].set_index(["method", "calib"])
    pe = (pd.read_csv(AN / "prior_effect.csv").query("proto=='P1' and day==@day") if (AN / "prior_effect.csv").exists()
          else pd.DataFrame(columns=["method", "calib"])).set_index(["method", "calib"])
    def cell(m, c, met):
        try:
            v = g.loc[(m, c), met]
        except KeyError:
            return "--"
        txt = f3(v)
        if c in ("s_platt+cprior", "s_gate") and (m, c) in pe.index and f"lo_{met}" in pe.columns:
            lo, hi = pe.loc[(m, c), f"lo_{met}"], pe.loc[(m, c), f"hi_{met}"]
            if pd.notna(lo) and (lo > 0 or hi < 0):
                txt += r"\rlap{\textsuperscript{\dag}}"
        return txt
    head = [r"\begin{table*}[!t]", r"\caption{Calibration Schemes and Base-Rate Correction, Day %d}\label{tab:prior}" % day,
            r"\centering", r"\footnotesize", r"\begin{threeparttable}", r"\setlength{\tabcolsep}{4.6pt}",
            r"\begin{tabular}{@{}l" + "r" * 12 + r"@{}}", r"\toprule",
            r" & \multicolumn{3}{c}{Pooled AUROC} & \multicolumn{6}{c}{Pooled ECE} & \multicolumn{3}{c}{Module-mean ECE} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-10}\cmidrule(lr){11-13}",
            r"Model & Platt & prior & gated & Platt & prior & gated & EM & ACC & oracle & Platt & prior & gated \\",
            r"\midrule"]
    lines = list(head)
    for m in [m for m in order if (m, "s_platt") in g.index]:
        if m == "lgbm_abs":
            lines.append(r"\midrule")
        name = NAME[m] + (" (R, V, S)" if m == "lgbm_lim" else "")
        cells = [cell(m, c, "auroc") for c in ("s_platt", "s_platt+cprior", "s_gate")]
        cells += [cell(m, c, "ece") for c in ("s_platt", "s_platt+cprior", "s_gate", "s_platt+em", "s_platt+acc",
                                              "s_platt+oracle")]
        cells += [cell(m, c, "mod_ece") for c in ("s_platt", "s_platt+cprior", "s_gate")]
        lines.append(" & ".join([name] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
              r"\item Held-out-module protocol P1. All schemes start from the same Platt calibrator. Prior: cohort prior"
              r" always applied; gated: applied only when the nested check on training courses favours it; EM and"
              r" ACC: label-shift estimators \cite{saerens2002neco,forman2008dmkd}; oracle: shift to the true rate of the"
              r" presentation (uses target labels, upper bound only). \dag: change against Platt with a 95\,\%"
              r" learner-bootstrap interval that excludes zero.",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table*}"]
    (out / "tab_prior.tex").write_text("\n".join(lines) + "\n")


def table_subgroups(out, day=28):
    """Learner subgroups under P1 on one day: AUROC and ECE of the benchmark, CatBoost and CREW (Platt), with the
    CREW-minus-CatBoost AUROC difference and its 95 % learner-bootstrap interval."""
    S = pd.read_csv(AN / "subgroups.csv").query("day == @day")
    label = {"gender": "Gender", "disability": "Disability", "age_band": "Age band", "imd_group": "IMD band"}
    names = {("gender", "F"): "female", ("gender", "M"): "male", ("disability", "N"): "no", ("disability", "Y"): "yes",
             ("age_band", "0-35"): "0--35", ("age_band", "35-55"): "35--55"}
    order = {"0-30 %": 0, "30-70 %": 1, "70-100 %": 2, "not recorded": 3}
    lines = [r"\begin{table*}[!t]", r"\caption{Results by Learner Subgroup, Day %d}\label{tab:groups}" % day,
             r"\centering", r"\footnotesize", r"\begin{threeparttable}", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
             r" & & & & \multicolumn{4}{c}{AUROC} & \multicolumn{3}{c}{ECE} \\",
             r"\cmidrule(lr){5-8}\cmidrule(lr){9-11}",
             r"Attribute & Group & $n$ & Rate & LR & CatBoost & CREW & CREW $-$ CatBoost (95\,\% CI) & LR & CatBoost & CREW \\",
             r"\midrule"]
    first = True
    for attr in ("gender", "disability", "age_band", "imd_group"):
        g = S[S.attr == attr].copy()
        if attr == "imd_group":
            g = g.assign(o=g["group"].map(order)).sort_values("o")
        if not first:
            lines.append(r"\addlinespace")
        first = False
        for i, (_, r) in enumerate(g.iterrows()):
            grp = names.get((attr, r["group"]), r["group"].replace(" %", r"\,\%").replace("-", "--"))
            au = bold_best([r["auroc_lim_lr"], r["auroc_cat_abs"], r["auroc_crew"]], True)
            ec = bold_best([r["ece_lim_lr"], r["ece_cat_abs"], r["ece_crew"]], False)
            ci = (f"${r['d_auroc_vs_cat_abs']:+.3f}$ (${r['lo_auroc_vs_cat_abs']:+.3f}$, ${r['hi_auroc_vs_cat_abs']:+.3f}$)"
                  if "d_auroc_vs_cat_abs" in r and pd.notna(r["d_auroc_vs_cat_abs"]) else "--")
            lines.append(" & ".join([label[attr] if i == 0 else "", grp, f"{int(r['n']):,}", f"{r['prev']:.2f}"] + au +
                                    [ci] + ec) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
              r"\item Held-out-module protocol P1, Platt calibration for every model. Rate: share of non-success. IMD:"
              r" deprivation index of the learner's area, lower bands more deprived. Learners aged 55 or over"
              r" (fewer than 200 enrolments) are not shown separately. Bold: best of the three models in a group.",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table*}"]
    (out / "tab_groups.tex").write_text("\n".join(lines) + "\n")


def table_ablation(out, day=28):
    """Ablation under P1: each variant with its pooled AUROC difference against the variant it modifies."""
    A = pd.read_csv(AN / f"ablation_d{day}.csv").set_index("method")
    rows = [("A", r"Absolute features, LightGBM", None),
            ("lgbm_lim", r"R, V, S of \cite{lim2026ijacsa}", None), ("lgbm_abs", r"+ CAL, TRD, PRE", "lgbm_lim"),
            ("lgbm_abs_noCAL", r"absolute $-$ CAL", "lgbm_abs"), ("lgbm_abs_noTRD", r"absolute $-$ TRD", "lgbm_abs"),
            ("lgbm_abs_noPRE", r"absolute $-$ PRE", "lgbm_abs"),
            ("B", r"Cohort-relative encoding, LightGBM", None),
            ("crew_lim", r"relative R, V, S", "lgbm_lim"), ("crew_rel", r"relative, all", "lgbm_abs"),
            ("crew_noCAL", r"CREW $-$ CAL", "lgbm_abs_noCAL"), ("crew", r"\textbf{CREW}", "lgbm_abs"),
            ("C", r"Daily sequences, GRU", None),
            ("gru_abs", r"absolute", None), ("gru_rel", r"\textbf{CREW-GRU}", "gru_abs")]
    num = {m: i + 1 for i, m in enumerate([m for m, _, _ in rows if m not in ("A", "B", "C")])}
    lines = [r"\begin{table}[!t]", r"\caption{Ablation of CREW Components, Day %d}\label{tab:ablation}" % day, r"\centering",
             r"\footnotesize", r"\begin{threeparttable}", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}rlrrcr@{}}", r"\toprule",
             r"& Variant & AUROC & $\Delta$AUROC (95\,\% CI) & vs & ECE \\", r"\midrule"]
    for m, lab, ref in rows:
        if m in ("A", "B", "C"):
            if m != "A":
                lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{6}{@{}l}{\textit{%s}} \\" % lab)
            continue
        if m not in A.index:
            continue
        r = A.loc[m]
        d = (f"${r['d_auroc']:+.3f}$ (${r['lo_auroc']:+.3f}$, ${r['hi_auroc']:+.3f}$)" if ref and pd.notna(r.get("d_auroc"))
             else "--")
        lines.append(f"({num[m]}) & {lab} & {r['auroc']:.3f} & {d} & {'(%d)' % num[ref] if ref else '--'} & {r['ece']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
              r"\item Held-out-module protocol P1, Platt calibration. $\Delta$AUROC: pooled AUROC of the variant minus that of"
              r" the variant in column vs, with a 95\,\% learner-bootstrap interval. CREW selects between rows (7) and the union of"
              r" rows (7) and (2) inside the training modules.",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    (out / "tab_ablation.tex").write_text("\n".join(lines) + "\n")


def table_cost(out):
    """Cost of preparing every model for one new course (median-sized held-out module, two CPU cores, no GPU)."""
    Cb = pd.read_csv(AN / "cost_bench.csv").set_index("method")
    order = ["lim_lr", "lim_rf", "lim_hgb", "lgbm_lim", "lgbm_iw", "lgbm_abs_bal", "cat_abs", "gru_abs",
             "lgbm_abs", "gru_rel", "crew"]
    r0 = Cb.loc[order[0]]
    lines = [r"\begin{table}[!t]", r"\caption{Training and Inference Cost per New Course}\label{tab:cost}", r"\centering",
             r"\footnotesize", r"\begin{threeparttable}", r"\setlength{\tabcolsep}{2.6pt}",
             r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{4}{c}{Training, s} & & \\", r"\cmidrule(lr){3-6}",
             r"Model & Feat. & select & calibr. & 5 fits & total & ms/1k & MB \\", r"\midrule"]
    for m in [m for m in order if m in Cb.index]:
        if m == "lgbm_abs":
            lines.append(r"\midrule")
        r = Cb.loc[m]
        name = NAME[m] + (" (R, V, S)" if m == "lgbm_lim" else "")
        name = name.replace("LightGBM, import.\\ weights", "LightGBM, IW").replace("LightGBM, module-bal.", "LightGBM, bal.")
        sel = "--" if r["t_select"] < 0.01 else f"{r['t_select']:.1f}"
        lines.append(" & ".join([name, f"{int(r['n_features'])}", sel, f"{r['t_calib']:.1f}", f"{r['t_fit_all']:.1f}",
                                 f"{r['t_total']:.1f}", f"{1000 * r['t_infer_per_1000']:.0f}", f"{r['peak_mb']:.0f}"])
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
              r"\item Held-out module CCC on day 28 (%s training and %s target enrolments), two CPU cores, no GPU."
              % (f"{int(r0['n_train']):,}", f"{int(r0['n_test']):,}") +
              r" Feat.: inputs of the final model; select: nested selection; calibr.: out-of-fold fits for Platt"
              r" scaling; 5 fits: final model with five seeds (LR: one); ms/1k: scoring 1,000 learners with all seeds;"
              r" MB: peak memory of the process, data loading included.",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    (out / "tab_cost.tex").write_text("\n".join(lines) + "\n")


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    M = metrics()
    table_main(M, out)
    table_prior(M, out)
    if (AN / "subgroups.csv").exists():
        table_subgroups(out)
    if (AN / "ablation_d28.csv").exists():
        table_ablation(out)
    if (AN / "cost_bench.csv").exists():
        table_cost(out)
    print("tables written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures" / "tables"))  # the paper copies tables/*.tex
    main(Path(ap.parse_args().out))
