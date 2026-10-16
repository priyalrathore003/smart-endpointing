"""Checks that nothing seen in training reappears in dev or test.

The build fails if any check finds a problem. Full turns are compared by `key` (text plus agent question, if any). Prefix texts are NOT compared
across splits: short prefixes like "i want" legitimately occur in many utterances.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .dedup import jaccard, shingles
from .types import Example, Label, Split

SplitRow = tuple[Split, Example]


@dataclass
class LeakageReport:
    utterance_ids_in_several_splits: list[str] = field(default_factory=list)
    batch_ids_in_several_splits: list[str] = field(default_factory=list)
    full_texts_in_several_splits: list[str] = field(default_factory=list)
    prefixes_without_end_row: list[str] = field(default_factory=list)
    near_duplicates_across_splits: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(vars(self).values())


def _in_several(rows: Sequence[SplitRow], key) -> list[str]:
    splits: dict[str, set[Split]] = defaultdict(set)
    for split, ex in rows:
        splits[key(ex)].add(split)
    return sorted(k for k, s in splits.items() if len(s) > 1)


def check_leakage(rows: Sequence[SplitRow], near_threshold: float = 0.8) -> LeakageReport:
    ends = [(s, ex) for s, ex in rows if ex.label is Label.END]
    report = LeakageReport(
        utterance_ids_in_several_splits=_in_several(rows, lambda ex: ex.utterance_id),
        batch_ids_in_several_splits=_in_several(rows, lambda ex: ex.source.batch_id),
        full_texts_in_several_splits=_in_several(ends, lambda ex: ex.key),
    )
    end_ids = {ex.utterance_id for _, ex in ends}
    report.prefixes_without_end_row = sorted({ex.utterance_id for _, ex in rows if ex.utterance_id not in end_ids})

    # Near-duplicates: held-out full utterances vs training ones in the same group.
    train_by_group = defaultdict(list)
    for s, ex in ends:
        if s is Split.TRAIN:
            train_by_group[(ex.source.domain, ex.source.intent, ex.source.language)].append(
                (ex.key, shingles(ex.key)))
    for s, ex in ends:
        if s is Split.TRAIN:
            continue
        sh = shingles(ex.key)
        for text, tsh in train_by_group[(ex.source.domain, ex.source.intent, ex.source.language)]:
            if jaccard(sh, tsh) >= near_threshold:
                report.near_duplicates_across_splits.append((text, ex.key))
    return report
