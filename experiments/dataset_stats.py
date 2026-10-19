"""Descriptive numbers of the data quoted in the paper (risk sets, module rates, click scales) -> extra_numbers.json.
Usage: python experiments/dataset_stats.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "extra_numbers.json"
nums = json.load(open(OUT)) if OUT.exists() else {}
for d in (7, 14, 28, 42):
    X = pd.read_parquet(ROOT / "results" / "features" / f"feat_d{d}.parquet",
                        columns=["code_module", "presentation", "id_student", "y", "clicks_total"])
    nums[f"data.d{d}.n"] = f"{len(X):,}"
    nums[f"data.d{d}.npos"] = f"{int(X['y'].sum()):,}"
    nums[f"data.d{d}.prev"] = f"{X['y'].mean():.3f}"
    pm = X.groupby("code_module")["y"].mean()
    nums[f"data.d{d}.prevmin"], nums[f"data.d{d}.prevmax"] = f"{pm.min():.2f}", f"{pm.max():.2f}"
    nums[f"data.d{d}.prevminmod"], nums[f"data.d{d}.prevmaxmod"] = pm.idxmin(), pm.idxmax()
    clicks = np.expm1(X["clicks_total"])
    mc = clicks.groupby(X["code_module"]).median()
    nums[f"data.d{d}.medclicksmin"], nums[f"data.d{d}.medclicksmax"] = f"{mc.min():.0f}", f"{mc.max():.0f}"
    nums[f"data.d{d}.medclicksminmod"], nums[f"data.d{d}.medclicksmaxmod"] = mc.idxmin(), mc.idxmax()
    for m, v in pm.items():
        nums[f"data.d{d}.prev.{m}"] = f"{v:.2f}"
    for m, v in mc.items():
        nums[f"data.d{d}.medclicks.{m}"] = f"{v:.0f}"
json.dump(nums, open(OUT, "w"), indent=1, sort_keys=True)
print({k: v for k, v in nums.items() if k.startswith("data.d28.")})

# learners removed by the learner-disjoint filter (day 28), as reported in the paper
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from oeew import protocols as P  # noqa: E402
X = pd.read_parquet(ROOT / "results" / "features" / "feat_d28.parquet", columns=["code_module", "code_presentation", "id_student"])
rem = {f.split(":")[1]: int((X.code_module != f.split(":")[1]).sum() - len(tr)) for f, tr, te in P.p1_lomo(X)}
nums["p1.d28.removedmin"], nums["p1.d28.removedmax"] = f"{min(rem.values()):,}", f"{max(rem.values()):,}"
nums["p1.d28.removedmaxmod"] = max(rem, key=rem.get)
(_, tr, te), = list(P.p2_forward(X))
nums["p2.d28.ntrain"], nums["p2.d28.ntest"] = f"{len(tr):,}", f"{len(te):,}"
nums["p2.d28.removed"] = f"{int(X.code_presentation.str.startswith('2013').sum() - len(tr)):,}"
json.dump(nums, open(OUT, "w"), indent=1, sort_keys=True)
