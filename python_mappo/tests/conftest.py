from pathlib import Path
import sys

import pytest
import yaml

# Allow `pytest` to be run from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="session")
def cfg() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))
