import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the OULAD archive in data/ (skipped when absent)")


@pytest.fixture(scope="session")
def oulad():
    from oeew import data
    z = ROOT / "data" / "oulad_uci.zip"
    if not z.exists():
        pytest.skip("OULAD archive not present; run `python -m oeew.data` first")
    return data.load(data.download(ROOT / "data"))
