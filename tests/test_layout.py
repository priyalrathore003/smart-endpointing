import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_python_is_3_11():
    assert sys.version_info[:2] == (3, 11)


@pytest.mark.parametrize("module", ["endpointing", "endpointing.data", "endpointing.models",
                                    "endpointing.evals", "endpointing.serve"])
def test_package_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("path", [".env", ".env.local", "runs/x/config.json", "data/raw/x.jsonl"])
def test_secret_and_output_paths_are_gitignored(path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert result.returncode == 0, f"{path} is not ignored by .gitignore"


def test_requirements_are_pinned():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l for l in lines)
