"""Find digit runs and spell some of them out digit by digit, as ASR sometimes does.

A digit run is a maximal sequence of all-digit tokens ("98765 43210" is one run of
two tokens). Each run is spelled with probability `spell_prob`, using a random
generator seeded by (seed, utterance_id), so the result does not depend on the
order utterances are processed in.
"""

import random
import re
from dataclasses import replace
from typing import Sequence

from .lexicon import DIGIT_WORDS
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
            if rng.random() < self.spell_prob:
                out += [words[int(d)] for tok in tokens[start:end] for d in tok]
                spelled_spans.append((new_start, len(out)))
            else:
                out += tokens[start:end]
            digit_spans.append((new_start, len(out)))
            cursor = end
        out += tokens[cursor:]
        return replace(utt, text=" ".join(out), digit_spans=tuple(digit_spans),
                       spelled_spans=tuple(spelled_spans))
