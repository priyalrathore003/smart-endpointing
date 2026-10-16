"""Remove exact and near-duplicate utterances before splitting.

Turns are compared by `key`: the text, plus the agent's question when there is one,
so "haan ji" answering two different questions is two different turns.
1. Exact: identical key anywhere in the set (keeps the first seen).
2. Near: word-3-gram Jaccard >= threshold, compared only within one
   (domain, intent, language) group, where paraphrases come from.
   Cost O(sum of group_size^2): about 96 groups x 80^2 = 0.6M comparisons.
   Near-duplicates across different groups are not caught (documented in data/README.md).
Input order decides which copy survives, so callers pass a deterministic order.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .types import Utterance


def shingles(text: str, n: int = 3) -> frozenset[tuple[str, ...]]:
    tokens = text.split()
    if len(tokens) < n:
        return frozenset([tuple(tokens)])
    return frozenset(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


@dataclass(frozen=True)
class DedupReport:
    input: int
    exact_removed: int
    near_removed: int
    kept: int
    near_examples: tuple[tuple[str, str], ...]   # (kept text, removed text), a few for inspection


class Deduplicator:
    def __init__(self, threshold: float = 0.8, max_examples: int = 10):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold, self.max_examples = threshold, max_examples

    def dedupe(self, utts: Sequence[Utterance]) -> tuple[list[Utterance], DedupReport]:
        seen: set[str] = set()
        exact_kept = []
        for u in utts:
            if u.key not in seen:
                seen.add(u.key)
                exact_kept.append(u)

        kept_by_group: dict[tuple, list[tuple[Utterance, frozenset]]] = defaultdict(list)
        kept, near_examples, near_removed = [], [], 0
        for u in exact_kept:
            group = kept_by_group[(u.source.domain, u.source.intent, u.source.language)]
            sh = shingles(u.key)
            match = next((k for k, ksh in group if jaccard(sh, ksh) >= self.threshold), None)
            if match is None:
                group.append((u, sh))
                kept.append(u)
            else:
                near_removed += 1
                if len(near_examples) < self.max_examples:
                    near_examples.append((match.key, u.key))

        report = DedupReport(input=len(utts), exact_removed=len(utts) - len(exact_kept),
                             near_removed=near_removed, kept=len(kept),
                             near_examples=tuple(near_examples))
        return kept, report
