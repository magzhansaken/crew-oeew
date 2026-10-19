"""Download and load the Open University Learning Analytics Dataset (OULAD).

Source: UCI Machine Learning Repository, dataset 349, DOI 10.24432/C5KK69, licence CC BY 4.0.
Kuzilek, Hlosta, Zdrahal, "Open University Learning Analytics dataset", Scientific Data 4:170171, 2017.
The data are not redistributed with this repository; this module downloads the official archive and checks
its SHA-256 against the value recorded when the experiments were run.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
SHA256 = "f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4"
TABLES = ["assessments", "courses", "studentAssessment", "studentInfo", "studentRegistration", "studentVle", "vle"]
KEY = ["code_module", "code_presentation", "id_student"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir: str | Path = "data") -> Path:
    """Download the UCI archive (if absent), verify SHA-256 and extract the CSV tables."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / "oulad_uci.zip"
    if not zpath.exists():
        urllib.request.urlretrieve(UCI_URL, zpath)
    digest = sha256(zpath)
    if digest != SHA256:
        raise ValueError(f"SHA-256 mismatch for {zpath}: {digest} != {SHA256}")
    raw = data_dir / "raw"
    if not all((raw / f"{t}.csv").exists() for t in TABLES):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw)
    return raw


def load(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the seven OULAD tables with compact dtypes; '?' marks missing values in the UCI release."""
    raw = Path(raw_dir)
    t = {}
    t["info"] = pd.read_csv(raw / "studentInfo.csv", na_values="?")
    t["reg"] = pd.read_csv(raw / "studentRegistration.csv", na_values="?")
    t["courses"] = pd.read_csv(raw / "courses.csv", na_values="?")
    t["assess"] = pd.read_csv(raw / "assessments.csv", na_values="?")
    t["sass"] = pd.read_csv(raw / "studentAssessment.csv", na_values="?")
    t["vle"] = pd.read_csv(raw / "vle.csv", na_values="?")
    t["svle"] = pd.read_csv(
        raw / "studentVle.csv",
        dtype={"code_module": "category", "code_presentation": "category", "id_student": "int32",
               "id_site": "int32", "date": "int16", "sum_click": "int32"},
    )
    t["svle"] = t["svle"].merge(t["vle"][["id_site", "activity_type"]], on="id_site", how="left")
    return t


def base_table(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per enrolment with the binary target: 1 = Fail or Withdrawn, 0 = Pass or Distinction."""
    b = t["info"].merge(t["reg"], on=KEY, how="left")
    b["y"] = b["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    b["presentation"] = b["code_module"] + "_" + b["code_presentation"]
    return b


if __name__ == "__main__":  # python -m oeew.data [data_dir]
    import sys
    raw = download(sys.argv[1] if len(sys.argv) > 1 else "data")
    print("OULAD ready in", raw)
