import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from evalaware import data


@pytest.fixture(scope="session")
def corpus():
    return data.load_corpus()
