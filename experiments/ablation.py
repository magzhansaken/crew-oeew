"""Ablation of CREW under P1: each variant against the variant it modifies (same rows, Platt calibration).

Differences in pooled AUROC and ECE receive 95 % learner-cluster bootstrap intervals (1,000 resamples by default).
Usage: python experiments/ablation.py [--boot 1000] [--day 28]
Output: results/analysis/ablation_d<day>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from analysis import boot_diff  # noqa: E402

EV, AN = ROOT / "results" / "eval", ROOT / "results" / "analysis"
# (variant, reference): the reference is the variant that the row modifies
PAIRS = [("lgbm_lim", None), ("lgbm_abs", "lgbm_lim"), ("lgbm_abs_noCAL", "lgbm_abs"), ("lgbm_abs_noTRD", "lgbm_abs"),
         ("lgbm_abs_noPRE", "lgbm_abs"), ("crew_lim", "lgbm_lim"), ("crew_rel", "lgbm_abs"),
         ("crew_noCAL", "lgbm_abs_noCAL"), ("crew", "lgbm_abs"), ("gru_abs", None), ("gru_rel", "gru_abs")]
# further contrasts quoted in the text: CAL inside CREW, the feature-set selection, the encoding on R, V, S only
EXTRA = [("crew", "crew_noCAL"), ("crew", "crew_rel"), ("crew", "crew_lim")]


def main(B: int, day: int):
    M = pd.read_parquet(EV / "metrics.parquet").query("proto == 'P1' and day == @day and calib == 's_platt'")
    M = M.set_index("method")
    out = []
    for m, ref in PAIRS:
        fp = EV / "rows" / f"P1_d{day}_{m}.parquet"
        if not fp.exists():
            continue
        r = dict(day=day, method=m, ref=ref or "", auroc=M.loc[m, "auroc"], ece=M.loc[m, "ece"],
                 mod_auroc=M.loc[m, "mod_auroc"], mod_ece=M.loc[m, "mod_ece"])
        if ref and (EV / "rows" / f"P1_d{day}_{ref}.parquet").exists():
            a, b = pd.read_parquet(fp), pd.read_parquet(EV / "rows" / f"P1_d{day}_{ref}.parquet")
            r |= boot_diff(a, b, "s_platt", "s_platt", B)
        out.append(r)
    pd.DataFrame(out).to_csv(AN / f"ablation_d{day}.csv", index=False)
    ex = []
    for m, ref in EXTRA:
        a = pd.read_parquet(EV / "rows" / f"P1_d{day}_{m}.parquet")
        b = pd.read_parquet(EV / "rows" / f"P1_d{day}_{ref}.parquet")
        ex.append(dict(day=day, method=m, ref=ref) | boot_diff(a, b, "s_platt", "s_platt", B))
    pd.DataFrame(ex).to_csv(AN / f"ablation_extra_d{day}.csv", index=False)
    print("ablation written to", AN / f"ablation_d{day}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--day", type=int, default=28)
    a = ap.parse_args()
    main(a.boot, a.day)
