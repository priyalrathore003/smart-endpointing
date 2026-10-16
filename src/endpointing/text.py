"""The one text normaliser. Training data and inference must both call `normalize`.

Output looks like ASR text: NFC, lowercase, no punctuation, apostrophes dropped
inside words ("don't" -> "dont"), Devanagari digits as ASCII, digit-group commas
removed ("1,500" -> "1500"). Every other punctuation mark or symbol is a word break.
Punctuation is removed because a final "?" or "।" would give the label away; ASR
output has no such cue.
"""

import re
import unicodedata

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))
_APOSTROPHES = re.compile("['‘’`]")
_DIGIT_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d)")


def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    t = unicodedata.normalize("NFC", text)
    t = t.translate(_DEVANAGARI_DIGITS).translate(_ZERO_WIDTH)
    t = _APOSTROPHES.sub("", t)
    t = _DIGIT_GROUP_COMMA.sub("", t)
    # P* punctuation, S* symbols, Z* separators, C* control -> space.
    # Letters, digits and combining marks (Devanagari vowel signs) are kept.
    t = "".join(" " if unicodedata.category(ch)[0] in "PSZC" else ch for ch in t)
    return " ".join(t.lower().split())
