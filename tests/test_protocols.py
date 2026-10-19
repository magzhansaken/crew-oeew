"""Splits never share learners between training and test; P1 holds out whole modules; P2 is forward in time."""
import numpy as np
import pandas as pd
import pytest

from oeew import protocols as P


@pytest.fixture
def toy():
    rng = np.random.default_rng(1)
    rows = []
    for m in "ABCDE":
        for p in ("2013B", "2013J", "2014B", "2014J"):
            for s in rng.choice(400, 60, replace=False):  # learners re-enrol across modules and presentations
                rows.append(dict(code_module=m, code_presentation=p, id_student=int(s), y=int(rng.random() < 0.4)))
    df = pd.DataFrame(rows).drop_duplicates(["code_module", "code_presentation", "id_student"]).reset_index(drop=True)
    df["presentation"] = df["code_module"] + "_" + df["code_presentation"]
    return df


@pytest.mark.parametrize("proto", ["P1", "P2", "P3"])
def test_student_disjoint(toy, proto):
    for _, tr, te in P.PROTOCOLS[proto](toy):
        assert len(tr) and len(te)
        assert not set(toy.iloc[tr]["id_student"]) & set(toy.iloc[te]["id_student"])


def test_p1_holds_out_module(toy):
    folds = list(P.p1_lomo(toy))
    assert len(folds) == toy["code_module"].nunique()
    for name, tr, te in folds:
        m = name.split(":")[1]
        assert set(toy.iloc[te]["code_module"]) == {m}
        assert m not in set(toy.iloc[tr]["code_module"])
        assert (toy["code_module"] == m).sum() == len(te)  # every enrolment of the module is tested


def test_p2_forward(toy):
    (_, tr, te), = list(P.p2_forward(toy))
    assert toy.iloc[tr]["code_presentation"].str.startswith("2013").all()
    assert toy.iloc[te]["code_presentation"].str.startswith("2014").all()


def test_p3_covers_all_rows_once(toy):
    seen = np.concatenate([te for _, _, te in P.p3_random(toy)])
    assert sorted(seen) == list(range(len(toy)))


def test_inner_folds_are_course_and_student_disjoint(toy):
    for tr, va in P.inner_module_folds(toy, 3):
        assert not set(toy.iloc[tr]["code_module"]) & set(toy.iloc[va]["code_module"])
        assert not set(toy.iloc[tr]["id_student"]) & set(toy.iloc[va]["id_student"])
