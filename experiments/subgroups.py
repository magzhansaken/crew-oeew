"""Learner subgroups under P1: discrimination and calibration of CREW against the benchmark and CatBoost.

Groups: gender, disability, age band and IMD band (deprivation index of the learner's area, grouped as 0-30 %,
30-70 % and 70-100 %; a missing band is its own group). Age 55 and over has fewer than 200 enrolments and is not
reported separately. All models use Platt calibration. For every group, the CREW-minus-model differences in AUROC and
ECE receive 95 % learner-cluster bootstrap intervals (all enrolments of a learner stay together).

Usage: python experiments/subgroups.py [--boot 1000]
Output: results/analysis/subgroups.csv, results/analysis/subgroup_gaps.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from oeew import metrics as Mx  # noqa: E402
from analysis import boot_arrays  # noqa: E402

EV, AN = ROOT / "results" / "eval", ROOT / "results" / "analysis"
METHODS = ["lim_lr", "cat_abs", "crew"]
IMD = {"0-10%": "0-30 %", "10-20": "0-30 %", "20-30%": "0-30 %", "30-40%": "30-70 %", "40-50%": "30-70 %",
       "50-60%": "30-70 %", "60-70%": "30-70 %", "70-80%": "70-100 %", "80-90%": "70-100 %", "90-100%": "70-100 %"}
MIN_N = 200


def groups(df: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for attr in ("gender", "disability", "age_band"):
        for v, idx in df.groupby(attr).indices.items():
            out[(attr, str(v))] = idx
    imd = df["imd_band"].map(IMD).fillna("not recorded")
    for v, idx in pd.Series(imd.to_numpy()).groupby(imd.to_numpy()).indices.items():
        out[("imd_group", str(v))] = idx
    return {k: v for k, v in out.items() if len(v) >= MIN_N}


def main(B: int):
    rows, gaps = [], []
    for d in (7, 14, 28, 42):
        R = {m: pd.read_parquet(EV / "rows" / f"P1_d{d}_{m}.parquet") for m in METHODS}
        key = ["code_module", "code_presentation", "id_student"]
        base = R["crew"][key + ["y", "gender", "disability", "age_band", "imd_band", "s_platt"]].rename(
            columns={"s_platt": "crew"})
        for m in ("lim_lr", "cat_abs"):
            base = base.merge(R[m][key + ["s_platt"]].rename(columns={"s_platt": m}), on=key, validate="1:1")
        assert len(base) == len(R["crew"])
        for (attr, g), idx in groups(base).items():
            sub = base.iloc[idx]
            y = sub["y"].to_numpy()
            r = dict(day=d, attr=attr, group=g, n=len(sub), prev=float(y.mean()))
            for m in METHODS:
                p = sub[m].to_numpy()
                r[f"auroc_{m}"] = Mx.auroc_fast(y, p)
                r[f"ece_{m}"] = Mx.ece(y, p)
                r[f"bias_{m}"] = float(p.mean() - y.mean())
            if B > 0:
                for m in ("lim_lr", "cat_abs"):
                    bd = boot_arrays(y, sub["crew"], sub[m], sub["id_student"], B, seed=0)
                    for k in ("d_auroc", "lo_auroc", "hi_auroc", "d_ece", "lo_ece", "hi_ece"):
                        r[f"{k}_vs_{m}"] = bd[k]
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(AN / "subgroups.csv", index=False)
    for (d, attr), g in S.groupby(["day", "attr"]):
        r = dict(day=d, attr=attr)
        for m in METHODS:
            r[f"auroc_gap_{m}"] = g[f"auroc_{m}"].max() - g[f"auroc_{m}"].min()
            r[f"ece_max_{m}"] = g[f"ece_{m}"].max()
        gaps.append(r)
    pd.DataFrame(gaps).to_csv(AN / "subgroup_gaps.csv", index=False)
    print("subgroups written to", AN)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    main(ap.parse_args().boot)
