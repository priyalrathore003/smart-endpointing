import ast
from pathlib import Path

import pytest

from endpointing.text import normalize

SRC = Path(__file__).resolve().parent.parent / "src" / "endpointing"


@pytest.mark.parametrize("raw, expected", [
    ("I don't KNOW.", "i dont know"),
    ("I’m waiting, okay?", "im waiting okay"),
    ("मेरा पैसा कट गया।", "मेरा पैसा कट गया"),
    ("नंबर ९८७६ है", "नंबर 9876 है"),
    ("Rs. 1,500 paid", "rs 1500 paid"),
    ("₹500 - kal tak", "500 kal tak"),
    ("e-mail id", "e mail id"),
    ("  many\t\nspaces  ", "many spaces"),
    ("zero​width", "zerowidth"),
    ("", ""),
    ("?!।", ""),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_vowel_signs_survive():
    assert normalize("किताब चाहिए") == "किताब चाहिए"


def test_nfc_makes_composed_and_decomposed_equal():
    assert normalize("café") == normalize("café")


def test_idempotent():
    s = "Haan ji, mera A/C no. 12,345 hai!"
    assert normalize(normalize(s)) == normalize(s)


def test_rejects_non_string():
    with pytest.raises(TypeError):
        normalize(None)


def test_normalize_is_defined_only_in_endpointing_text():
    """Train/inference skew guard: no other module may define or reimplement a normaliser."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path == SRC / "text.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and "normali" in node.name:
                offenders.append(f"{path.name}: def {node.name}")
            if isinstance(node, ast.Attribute) and node.attr == "normalize" and \
                    isinstance(node.value, ast.Name) and node.value.id == "unicodedata":
                offenders.append(f"{path.name}: unicodedata.normalize")
    assert offenders == []


def test_source_coerces_strings_and_rejects_unknown_values():
    from endpointing.data.types import Language, Source
    s = Source(batch_id="b", language="hi", domain="banking", intent="x", style="plain")
    assert s.language is Language.HI
    with pytest.raises(ValueError):
        Source(batch_id="b", language="tamil", domain="banking", intent="x", style="plain")
    with pytest.raises(ValueError):
        Source(batch_id="", language="hi", domain="banking", intent="x", style="plain")
