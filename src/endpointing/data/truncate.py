"""Turn one complete utterance into labelled rows: the full text (END) plus prefixes (NOT_END).

Prefixes are cut at word boundaries. Kept per utterance:
- the END row;
- every prefix that ends in a hard negative (they are rare and the most useful);
- up to `max_random_prefixes` other prefixes, sampled with a seeded generator.
A complete utterance that itself ends on a hard-negative word ("...i did not do that")
stays END and is tagged end_on_hard_negative: a real counterexample to the lexicon.
Keeping every prefix would make NOT_END outnumber END about 10:1 and over-weight
long utterances.
"""

import random
from dataclasses import dataclass

from .lexicon import ends_with_hard_negative, heuristic_complete
from .script import is_code_mixed
from .types import Example, Label, Utterance


@dataclass(frozen=True)
class TruncationConfig:
    max_random_prefixes: int = 3
    short_max_words: int = 4

    def __post_init__(self) -> None:
        if self.max_random_prefixes < 0 or self.short_max_words < 1:
            raise ValueError("invalid truncation config")


def _inside(spans: tuple[tuple[int, int], ...], k: int) -> bool:
    """True if a cut after k tokens falls strictly inside a span."""
    return any(s < k < e for s, e in spans)


def _overlaps(spans: tuple[tuple[int, int], ...], k: int) -> bool:
    """True if the first k tokens include part of a span."""
    return any(s < k for s, _ in spans)


class Truncator:
    def __init__(self, config: TruncationConfig = TruncationConfig(), seed: int = 0):
        self.config, self.seed = config, seed

    def examples(self, utt: Utterance) -> list[Example]:
        tokens = utt.text.split()
        n = len(tokens)
        lang = utt.source.language
        hard = {k for k in range(1, n) if ends_with_hard_negative(tokens[:k], lang)}
        others = [k for k in range(1, n) if k not in hard]
        rng = random.Random(f"{self.seed}:{utt.utterance_id}:prefixes")
        sampled = rng.sample(others, min(self.config.max_random_prefixes, len(others)))
        return [self._row(utt, tokens, k, k in hard) for k in sorted(hard | set(sampled) | {n})]

    def _row(self, utt: Utterance, tokens: list[str], k: int, hard: bool) -> Example:
        n = len(tokens)
        return Example(
            utterance_id=utt.utterance_id, source=utt.source, text=" ".join(tokens[:k]), context=utt.context,
            label=Label.END if k == n else Label.NOT_END, k=k, n=n,
            hard_negative=hard,
            heuristic_complete=heuristic_complete(tokens[:k], utt.source.language),
            has_digits=_overlaps(utt.digit_spans, k),
            digits_mid=_inside(utt.digit_spans, k),
            digits_spelled=_overlaps(utt.spelled_spans, k),
            short=n <= self.config.short_max_words,
            code_mixed=is_code_mixed(tokens, utt.source.language),
            end_on_hard_negative=k == n and ends_with_hard_negative(tokens, utt.source.language))
