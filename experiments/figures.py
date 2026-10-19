"""Figures of the paper (vector PDF) from results/eval and results/analysis. No model fitting here.

Usage: python experiments/figures.py [--out figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import style as S  # noqa: E402
from oeew import metrics as Mx  # noqa: E402

EV, AN = ROOT / "results" / "eval", ROOT / "results" / "analysis"
OURS_CAL = "s_platt"
LIM_CAL = "s_platt"


def rows(proto, d, m):
    fp = EV / "rows" / f"{proto}_d{d}_{m}.parquet"
    return pd.read_parquet(fp) if fp.exists() else None


def cal_for(m, df):
    if m == "crew":
        return OURS_CAL
    return LIM_CAL if LIM_CAL in df.columns else "s_cd"


def boot_ci(df, col, B=1000, seed=1):
    """95 % interval of pooled AUROC from a learner-cluster bootstrap (all enrolments of a learner stay together)."""
    y, p = df["y"].to_numpy(), df[col].to_numpy()
    codes, uniq = pd.factorize(df["id_student"], sort=True)
    order = np.argsort(codes, kind="mergesort")
    lens = np.bincount(codes, minlength=len(uniq))
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for k in range(B):
        pick = rng.integers(0, len(lens), len(lens))
        ln = lens[pick]
        ix = order[np.arange(ln.sum()) + np.repeat(starts[pick] - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln)]
        bs[k] = Mx.auroc_fast(y[ix], p[ix])
    return np.percentile(bs, [2.5, 97.5])


FIG3 = [  # (method, label, colour, marker, filled, line style): hue = model family, solid = cohort-relative
    ("crew", "CREW (LightGBM, relative)", "#D55E00", "o", True, "-"),
    ("lgbm_abs", "LightGBM, absolute", "#D55E00", "o", False, (0, (4, 2))),
    ("gru_rel", "CREW-GRU (relative)", "#0072B2", "s", True, "-"),
    ("gru_abs", "GRU, absolute", "#0072B2", "s", False, (0, (4, 2))),
    ("cat_abs", "CatBoost, absolute", "#009E73", "^", False, (0, (4, 2))),
    ("lim_lr", "LR, benchmark", "#555555", "D", False, (0, (1, 1.5))),
]


def fig_days(out, days=(7, 14, 28, 42), B=1000):
    """Held-out-module protocol P1, Platt calibration for every model: (a) pooled AUROC with a 95 % learner-bootstrap
    interval, (b) AUROC averaged over the seven held-out modules, (c) pooled ECE."""
    fig, ax = plt.subplots(1, 3, figsize=(S.FULL_W, 1.85))
    table = []
    for m, lab, col, mk, filled, ls in FIG3:
        xs, au, lo, hi, ma, ec = [], [], [], [], [], []
        for d in days:
            df = rows("P1", d, m)
            if df is None:
                continue
            c = "s_platt"
            y, p = df["y"].to_numpy(), df[c].to_numpy()
            xs.append(d); au.append(Mx.auroc_fast(y, p))
            l, h = boot_ci(df, c, B=B); lo.append(l); hi.append(h)
            ma.append(np.mean([Mx.auroc_fast(g["y"].to_numpy(), g[c].to_numpy()) for _, g in df.groupby("code_module")]))
            ec.append(Mx.ece(y, p))
            table.append(dict(method=m, day=d, auroc=au[-1], lo=l, hi=h, mod_auroc=ma[-1], ece=ec[-1]))
        rel = ls == "-"
        kw = dict(color=col, marker=mk, ls=ls, label=lab, lw=1.25 if rel else 0.9, ms=4.2 if rel else 3.8,
                  mfc=col if filled else "white", mec=col, mew=0.9, zorder=4 if rel else 3)
        ax[0].errorbar(xs, au, yerr=[np.array(au) - lo, np.array(hi) - au], capsize=1.6, elinewidth=0.6, **kw)
        ax[1].plot(xs, ma, **kw)
        ax[2].plot(xs, ec, **kw)
    for a, t, yl in ((ax[0], "(a)", "Pooled AUROC"), (ax[1], "(b)", "Module-mean AUROC"), (ax[2], "(c)", "Pooled ECE")):
        a.set_xticks(days); a.set_xlabel("Prediction day"); a.set_ylabel(yl)
        a.text(0.03, 0.97, t, transform=a.transAxes, va="top", ha="left", fontsize=8)
    ax[2].set_ylim(0, None)
    h, l = ax[0].get_legend_handles_labels()
    h = [x[0] if isinstance(x, tuple) or hasattr(x, "lines") else x for x in h]
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.10), handlelength=2.6, columnspacing=1.2,
               handletextpad=0.4)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "fig3_days.pdf")
    plt.close(fig)
    pd.DataFrame(table).to_csv(out / "fig3_days.csv", index=False)  # table view of the figure


PRIOR_C = "#8E44AD"  # cohort prior (purple) in every figure; validated with the CREW, CatBoost and LR colours
FIG4 = [  # (method, calibration, label, colour, marker, filled)
    ("lim_lr", "s_platt", "LR, benchmark", "#555555", "D", False),
    ("lim_lr", "s_gate", "LR + gated cohort prior", PRIOR_C, "D", True),
    ("cat_abs", "s_platt", "CatBoost, absolute", "#009E73", "^", False),
    ("crew", "s_platt", "CREW", "#D55E00", "o", True),
]


def fig_modules(out, d=28):
    """Per held-out module under P1 on day d: (a) AUROC and (b) ECE (ten equal-frequency bins)."""
    fig, ax = plt.subplots(1, 2, figsize=(S.COL_W, 2.15), sharey=True)
    table, mods = [], None
    for k, (m, c, lab, col, mk, filled) in enumerate(FIG4):
        df = rows("P1", d, m)
        if df is None or c not in df.columns:
            continue
        g = df.groupby("code_module").apply(lambda x: pd.Series(dict(
            auroc=Mx.auroc_fast(x["y"].to_numpy(), x[c].to_numpy()), ece=Mx.ece(x["y"], x[c]))), include_groups=False)
        mods = list(g.index)
        y = np.arange(len(mods)) + (k - (len(FIG4) - 1) / 2) * 0.19
        kw = dict(color=col, marker=mk, ls="none", label=lab, ms=4.0, mfc=col if filled else "white", mec=col, mew=0.9,
                  zorder=4 if m == "crew" else 3)
        ax[0].plot(g["auroc"], y, **kw); ax[1].plot(g["ece"], y, **kw)
        table += [dict(method=m, calib=c, module=mo, auroc=r.auroc, ece=r.ece) for mo, r in g.iterrows()]
    ax[0].set_yticks(range(len(mods))); ax[0].set_yticklabels(mods); ax[0].set_ylim(len(mods) - 0.5, -0.5)
    ax[0].set_xlabel("AUROC within module"); ax[1].set_xlabel("ECE within module")
    ax[1].set_xlim(0, 1.08 * max(r["ece"] for r in table))
    for a, t in ((ax[0], "(a)"), (ax[1], "(b)")):
        a.grid(axis="y", visible=False)
        a.set_title(t, fontsize=8, pad=2)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13), columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out / "fig4_modules.pdf")
    plt.close(fig)
    pd.DataFrame(table).to_csv(out / "fig4_modules.csv", index=False)


FIG5 = [  # (method, calibration, panel title, colour, marker, filled)
    ("lim_lr", "s_platt", "LR, Platt", "#555555", "D", False),
    ("lim_lr", "s_platt+em", "LR + EM", "#0072B2", "s", False),
    ("lim_lr", "s_platt+acc", "LR + ACC/BBSE", "#0072B2", "v", False),
    ("lim_lr", "s_platt+cprior", "Cohort prior", PRIOR_C, "D", True),
    ("crew", "s_platt", "CREW, Platt", "#D55E00", "o", True),
]


def fig_prevalence(out, d=28):
    """Estimated rate of non-success (mean predicted probability) against the true rate of each held-out
    presentation under P1 on day d; one panel per estimator, MAE over the 22 presentations in each panel."""
    fig, ax = plt.subplots(1, len(FIG5), figsize=(S.FULL_W, 1.85), sharex=True, sharey=True)
    table = []
    for a, (m, c, title, col, mk, filled), t in zip(ax, FIG5, "abcde"):
        df = rows("P1", d, m)
        g = df.groupby("presentation").agg(true=("y", "mean"), est=(c, "mean"))
        mae = float((g["est"] - g["true"]).abs().mean())
        a.plot([0, 1], [0, 1], color="#888888", lw=0.6, ls="-", zorder=1)
        a.plot(g["true"], g["est"], mk, color=col, ms=3.6, mfc=col if filled else "white", mec=col, mew=0.8, zorder=3,
               clip_on=False)
        a.set_xlim(0.2, 0.6); a.set_ylim(0, 1)
        a.set_xticks([0.2, 0.4, 0.6]); a.set_yticks([0, 0.5, 1])
        a.set_title(f"({t}) {title}\nMAE {mae:.3f}", fontsize=7.5, pad=3, linespacing=1.15)
        a.set_xlabel("True rate")
        table += [dict(method=m, calib=c, presentation=pr, true=r["true"], est=r["est"]) for pr, r in g.iterrows()]
    ax[0].set_ylabel("Estimated rate")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(out / "fig5_prevalence.pdf")
    plt.close(fig)
    pd.DataFrame(table).to_csv(out / "fig5_prevalence.csv", index=False)


LABEL = {"n_missed": "Missed deadlines", "mean_days_early": "Days early per submission", "unique_res": "Distinct resources",
         "active_days": "Active days", "days_since_active": "Days since last activity", "active_ratio": "Share of active days",
         "clicks_total": "Total clicks", "pre_clicks": "Clicks before start", "longest_gap": "Longest inactive spell",
         "c_forumng": "Forum clicks", "num_of_prev_attempts": "Previous attempts", "c_questionnaire": "Questionnaire clicks",
         "c_oucontent": "Content clicks", "c_page": "Page clicks", "days_to_next_deadline": "Days to next deadline",
         "frac_due_submitted": "Share of due work submitted", "n_late": "Late submissions",
         "clicks_recent": "Clicks in the last week", "daily_mean": "Mean daily clicks", "studied_credits": "Studied credits",
         "date_registration": "Registration day", "c_homepage": "Homepage clicks", "c_resource": "Resource clicks",
         "c_subpage": "Subpage clicks", "c_url": "Link clicks", "c_quiz": "Quiz clicks", "n_submitted": "Submissions",
         "type_entropy": "Activity-type entropy", "slope14": "Two-week click trend", "daily_std": "Daily click spread"}


def fig_shap(out, d=28, methods=(("crew", "CREW (relative)", True), ("lgbm_abs", "LightGBM, absolute", False)), top=10):
    """Share of mean |SHAP| per base feature (rank and standard score of a feature summed) in each held-out module:
    bar = mean over the seven modules, whiskers = minimum and maximum."""
    W = {}
    for m, _, _ in methods:
        sh = pd.read_csv(AN / f"shap_fold_d{d}_{m}.csv")
        sh["base"] = sh["feature"].map(lambda c: c[2:] if c[:2] in ("r_", "z_") else c)
        w = sh.pivot_table(index="base", columns="module", values="mean_abs_shap", aggfunc="sum").fillna(0)
        W[m] = w.div(w.sum(axis=0), axis=1)
    order = W[methods[0][0]].mean(axis=1).nlargest(top).index[::-1]
    fig, ax = plt.subplots(figsize=(S.COL_W, 2.75))
    y = np.arange(len(order))
    h = 0.36
    table = []
    for k, (m, lab, filled) in enumerate(methods):
        w = W[m].reindex(order).fillna(0)
        mean, lo, hi = w.mean(axis=1), w.min(axis=1), w.max(axis=1)
        yy = y + (0.5 - k) * h * 1.08
        ax.barh(yy, mean, height=h, color="#D55E00" if filled else "white", edgecolor="#D55E00", linewidth=0.8,
                label=lab, zorder=3)
        ax.errorbar(mean, yy, xerr=[mean - lo, hi - mean], fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.4,
                    zorder=4)
        table += [dict(method=m, base=b, mean=mean[b], min=lo[b], max=hi[b]) for b in order]
    ax.set_yticks(y); ax.set_yticklabels([LABEL.get(b, b) for b in order], fontsize=7)
    ax.set_xlabel("Share of mean |SHAP| in a held-out module")
    ax.set_xlim(0, None); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(out / f"fig6_shap_d{d}.pdf")
    plt.close(fig)
    pd.DataFrame(table).to_csv(out / f"fig6_shap_d{d}.csv", index=False)


def main(out, only=None):
    S.apply()
    out.mkdir(parents=True, exist_ok=True)
    jobs = {"fig3": lambda: fig_days(out), "fig4": lambda: fig_modules(out),
            "fig5": lambda: fig_prevalence(out), "fig6": lambda: fig_shap(out)}
    for k, f in jobs.items():
        if not only or k in only:
            f()
    print("figures written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--only", nargs="*", default=None, help="subset of fig3 fig4 fig5 fig6")
    a = ap.parse_args()
    main(Path(a.out), a.only)
