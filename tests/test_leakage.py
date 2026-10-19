"""Leakage checks on the real data (one module, to keep the test fast).

1. Features on day d do not change when every record dated on or after d is altered.
2. Features and cohort statistics do not change when outcome labels are permuted (no label use, incl. REL).
"""
import numpy as np
import pandas as pd
import pytest

from oeew import calibration as C, data, features as F

pytestmark = pytest.mark.data
D = 14
MOD = "GGG"


def _subset(t):
    out = {}
    for k, v in t.items():
        out[k] = v[v["code_module"] == MOD].copy() if "code_module" in v.columns else v.copy()
    out["sass"] = t["sass"][t["sass"]["id_assessment"].isin(out["assess"]["id_assessment"])].copy()
    return out


@pytest.fixture(scope="module")
def small(oulad):
    t = _subset(oulad)
    return t, data.base_table(t), F.top_types(oulad["svle"])


def _feats(t, base, types):
    X, fam = F.build(t, base, D, types)
    cols = [c for v in fam.values() for c in v]
    return X.set_index(["code_module", "code_presentation", "id_student"])[cols].sort_index()


def test_future_records_do_not_change_features(small):
    t, base, types = small
    ref = _feats(t, base, types)
    t2 = {k: v.copy() for k, v in t.items()}
    sv = t2["svle"]
    fut = sv["date"] >= D
    sv.loc[fut, "sum_click"] = sv.loc[fut, "sum_click"] * 9 + 3          # alter future clicks
    sa = t2["sass"]
    t2["sass"] = sa[sa["date_submitted"] < D]                             # drop future submissions
    late = base["date_unregistration"] >= D
    base2 = base.copy()
    base2.loc[late, "date_unregistration"] = base2.loc[late, "date_unregistration"] + 5  # later withdrawals move
    pd.testing.assert_frame_equal(ref, _feats(t2, base2, types))


def test_labels_are_not_used(small):
    t, base, types = small
    ref = _feats(t, base, types)
    rng = np.random.default_rng(0)
    base2 = base.copy()
    base2["y"] = rng.permutation(base2["y"].to_numpy())
    base2["final_result"] = rng.permutation(base2["final_result"].to_numpy())
    pd.testing.assert_frame_equal(ref, _feats(t, base2, types))
    X1, _ = F.build(t, base, D, types)
    X2, _ = F.build(t, base2, D, types)
    pd.testing.assert_frame_equal(C.cohort_stats(X1, base, D), C.cohort_stats(X2, base2, D))


def test_risk_set_excludes_early_withdrawals(small):
    t, base, types = small
    X, _ = F.build(t, base, D, types)
    assert not (X["date_unregistration"] < D).any()
    assert len(X) == int((~(base["date_unregistration"] < D)).sum())
