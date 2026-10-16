"""Find digit runs and spell some of them out digit by digit, as ASR sometimes does.

A digit run is a maximal sequence of all-digit tokens ("98765 43210" is one run of
two tokens). Only identifier-like runs are spelled: 4+ digits in total (phone, account,
OTP, order ID) and not a date or an amount. Dates and amounts are never read digit by
digit ("30 अप्रैल" is never "तीन ज़ीरो अप्रैल"). A run is a date if a date word sits
next to it or it has a day-month(-year) shape; an amount if a currency or
amount word sits next to it.

Each spellable run is spelled with probability `spell_prob`, using a random generator
seeded by (seed, utterance_id), so the result does not depend on processing order.
"""

import random
import re
from dataclasses import replace
from typing import Sequence

from .lexicon import AMOUNT_WORDS, DATE_WORDS, DIGIT_WORDS
from .types import Utterance

_DIGITS = re.compile(r"[0-9]+")


def digit_runs(tokens: Sequence[str]) -> list[tuple[int, int]]:
    """[start, end) token spans of consecutive all-digit tokens. O(n)."""
    runs, start = [], None
    for i, tok in enumerate(tokens):
        if _DIGITS.fullmatch(tok):
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(tokens)))
    return runs


MIN_SPELL_DIGITS = 4


def _neighbours(tokens: Sequence[str], start: int, end: int) -> list[str]:
    return [tokens[i] for i in (start - 1, end) if 0 <= i < len(tokens)]


def _date_shaped(run: Sequence[str]) -> bool:
    """dd mm or dd mm yyyy / dd mm yy, as people say dates with spaces."""
    lens = [len(t) for t in run]
    return len(lens) in (2, 3) and lens[0] <= 2 and lens[1] <= 2 and (len(lens) == 2 or lens[2] in (2, 4))


def is_spellable(tokens: Sequence[str], start: int, end: int) -> bool:
    run = tokens[start:end]
    if sum(len(t) for t in run) < MIN_SPELL_DIGITS:
        return False
    near = _neighbours(tokens, start, end)
    if any(w in DATE_WORDS for w in near) or _date_shaped(run):
        return False
    return not any(w in AMOUNT_WORDS for w in near)


class DigitSpeller:
    def __init__(self, spell_prob: float = 0.25, seed: int = 0):
        if not 0.0 <= spell_prob <= 1.0:
            raise ValueError("spell_prob must be in [0, 1]")
        self.spell_prob, self.seed = spell_prob, seed

    def apply(self, utt: Utterance) -> Utterance:
        """Returns the utterance with some runs spelled and digit/spelled spans filled in."""
        rng = random.Random(f"{self.seed}:{utt.utterance_id}")
        words = DIGIT_WORDS[utt.source.language]
        tokens = utt.text.split()
        out: list[str] = []
        digit_spans, spelled_spans = [], []
        cursor = 0
        for start, end in digit_runs(tokens):
            out += tokens[cursor:start]
            new_start = len(out)
            if is_spellable(tokens, start, end) and rng.random() < self.spell_prob:
                out += [words[int(d)] for tok in tokens[start:end] for d in tok]
                spelled_spans.append((new_start, len(out)))
            else:
                out += tokens[start:end]
            digit_spans.append((new_start, len(out)))
            cursor = end
        out += tokens[cursor:]
        return replace(utt, text=" ".join(out), digit_spans=tuple(digit_spans),
                       spelled_spans=tuple(spelled_spans))
