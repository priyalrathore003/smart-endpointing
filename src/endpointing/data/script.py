"""Script checks on normalised text.

Hindi may contain a few Latin-script words, as real Hindi speech does ("मेरा parcel कहां है"):
up to MAX_LATIN_SHARE_HI of its tokens. Those utterances are tagged code_mixed.
English and Hinglish must contain no Devanagari. Tokens without letters (numbers) are neutral.
"""

import re
from typing import Sequence

from .types import Language

MAX_LATIN_SHARE_HI = 0.25
_LATIN = re.compile(r"[a-z]")
_DEVANAGARI = re.compile("[\u0900-\u097f]")


def latin_tokens(tokens: Sequence[str]) -> int:
    return sum(bool(_LATIN.search(t)) for t in tokens)


def script_ok(text: str, language: Language) -> bool:
    tokens = text.split()
    if language is Language.HI:
        return not tokens or latin_tokens(tokens) / len(tokens) <= MAX_LATIN_SHARE_HI
    return not _DEVANAGARI.search(text)


def is_code_mixed(tokens: Sequence[str], language: Language) -> bool:
    """A Hindi utterance that kept at least one Latin-script word."""
    return language is Language.HI and latin_tokens(tokens) > 0
