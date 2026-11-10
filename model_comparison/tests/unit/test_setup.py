from pathlib import Path

def test_project_metadata():
    assert '>=3.10,<3.11' in Path('pyproject.toml').read_text()
