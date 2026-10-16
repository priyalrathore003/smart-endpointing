"""Turn one complete utterance into labelled rows: the full text (END) plus prefixes (NOT_END).

Prefixes are cut at word boundaries. Which NOT_END prefixes are kept is a strategy:
- CappedSampler (default, all splits): at most 3 per utterance. One where the prefix
  ends on a hard negative or filler (if any), one in the last 30% of the words, one
  random; seeded, no duplicates. Without the cap, long utterances dominate and most
  prefixes are 1-2 word stubs.
- HardNegativeOnlySampler (short replies): only hard-negative prefixes, up to the cap.
  Random cuts of a short answer ("kal", "nahi") are usually complete answers given the
  question, so they would be mislabelled NOT_END.
- AllHardPlusRandom: the previous scheme, kept only for the before/after comparison.
`all_prefixes` returns every boundary, for the analysis-only file.

A complete utterance that itself ends on a hard-negative word ("...i did not do that")
stays END and is tagged end_on_hard_negative: a real counterexample to the lexicon.
"""

import math
import random
from dataclasses import dataclass
from typing import Protocol, Sequence

from .lexicon import ends_with_filler, ends_with_hard_negative, heuristic_complete
from .script import is_code_mixed
from .types import Example, Label, Style, Utterance


@dataclass(frozen=True)
class TruncationConfig:
    max_prefixes: int = 3
    tail_share: float = 0.3
    short_max_words: int = 4

    def __post_init__(self) -> None:
        if self.max_prefixes < 0 or self.short_max_words < 1 or not 0.0 < self.tail_share <= 1.0:
            raise ValueError("invalid truncation config")


class PrefixSampler(Protocol):
    def choose(self, tokens: Sequence[str], utt: Utterance, rng: random.Random) -> set[int]:
        """Prefix lengths k (1 <= k < n) to keep as NOT_END rows."""


def _hard(tokens: Sequence[str], utt: Utterance) -> list[int]:
    lang = utt.source.language
    return [k for k in range(1, len(tokens)) if ends_with_hard_negative(tokens[:k], lang)]


class CappedSampler:
    def __init__(self, config: TruncationConfig):
        self.config = config

    def choose(self, tokens, utt, rng):
        n, lang = len(tokens), utt.source.language
        chosen: set[int] = set()
        anchors = [k for k in range(1, n)
                   if ends_with_hard_negative(tokens[:k], lang) or ends_with_filler(tokens[:k], lang)]
        tail_start = max(1, math.ceil(n * (1 - self.config.tail_share)))
        for pool in (anchors, list(range(tail_start, n)), list(range(1, n))):
            if len(chosen) >= self.config.max_prefixes:
                break
            options = [k for k in pool if k not in chosen]
            if options:
                chosen.add(rng.choice(options))
        return chosen


class HardNegativeOnlySampler:
    def __init__(self, config: TruncationConfig):
        self.config = config

    def choose(self, tokens, utt, rng):
        hard = _hard(tokens, utt)
        return set(rng.sample(hard, min(self.config.max_prefixes, len(hard))))


class AllHardPlusRandom:
    """Previous scheme: every hard-negative prefix plus 3 random others."""

    def __init__(self, n_random: int = 3):
        self.n_random = n_random

    def choose(self, tokens, utt, rng):
        hard = set(_hard(tokens, utt))
        others = [k for k in range(1, len(tokens)) if k not in hard]
        return hard | set(rng.sample(others, min(self.n_random, len(others))))


def _inside(spans: tuple[tuple[int, int], ...], k: int) -> bool:
    """True if a cut after k tokens falls strictly inside a span."""
    return any(s < k < e for s, e in spans)


def _ends_in(spans: tuple[tuple[int, int], ...], k: int) -> bool:
    """True if the k-th token (the last one of the prefix) belongs to a span."""
    return any(s < k <= e for s, e in spans)


def _overlaps(spans: tuple[tuple[int, int], ...], k: int) -> bool:
    """True if the first k tokens include part of a span."""
    return any(s < k for s, _ in spans)


class Truncator:
    def __init__(self, config: TruncationConfig = TruncationConfig(), seed: int = 0,
                 sampler: PrefixSampler | None = None, reply_sampler: PrefixSampler | None = None):
        self.config, self.seed = config, seed
        self.sampler = sampler or CappedSampler(config)
        self.reply_sampler = reply_sampler or HardNegativeOnlySampler(config)

    def examples(self, utt: Utterance) -> list[Example]:
        tokens = utt.text.split()
        sampler = self.reply_sampler if utt.source.style is Style.SHORT_REPLIES else self.sampler
        rng = random.Random(f"{self.seed}:{utt.utterance_id}:prefixes")
        ks = sampler.choose(tokens, utt, rng) | {len(tokens)}
        return [self._row(utt, tokens, k) for k in sorted(ks)]

    def all_prefixes(self, utt: Utterance) -> list[Example]:
        """Every word boundary plus the END row. Analysis only: never used for training."""
        tokens = utt.text.split()
        return [self._row(utt, tokens, k) for k in range(1, len(tokens) + 1)]

    def _row(self, utt: Utterance, tokens: list[str], k: int) -> Example:
        n, lang = len(tokens), utt.source.language
        return Example(
            utterance_id=utt.utterance_id, source=utt.source, text=" ".join(tokens[:k]), context=utt.context,
            label=Label.END if k == n else Label.NOT_END, k=k, n=n,
            hard_negative=k < n and ends_with_hard_negative(tokens[:k], lang),
            # "char do" (four two) is a number, not the imperative "do"
            heuristic_complete=heuristic_complete(tokens[:k], lang) and not _ends_in(utt.digit_spans, k),
            has_digits=_overlaps(utt.digit_spans, k),
            digits_mid=_inside(utt.digit_spans, k),
            digits_spelled=_overlaps(utt.spelled_spans, k),
            short=n <= self.config.short_max_words,
            code_mixed=is_code_mixed(tokens, lang),
            end_on_hard_negative=k == n and ends_with_hard_negative(tokens, lang),
            prefix_of_reply=k < n and utt.source.style is Style.SHORT_REPLIES)
