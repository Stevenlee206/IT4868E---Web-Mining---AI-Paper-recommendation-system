"""Shared, disposable test fixtures."""
from pathlib import Path
import shutil
import pytest

@pytest.fixture
def raw_dir(tmp_path):
    source = Path(__file__).parent / 'fixtures' / 'citeulike-mini'
    target = tmp_path / 'raw'
    shutil.copytree(source, target)
    return target

@pytest.fixture
def prepared(raw_dir):
    from data_loader.preprocessing import prepare_dataset
    return prepare_dataset(raw_dir, seed=42)

@pytest.fixture(scope='session',autouse=True)
def deterministic_test_environment():
    from pipeline.reproducibility import setup
    setup(42,1,True)
