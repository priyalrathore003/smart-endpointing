"""Assign whole generation batches to train/dev/test.

Splitting by batch (one LLM call) rather than by utterance keeps paraphrases from
the same call on one side. Every utterance ID and every prefix lives inside one
batch, so the spec's "split by ID, never by prefix" rule holds as well.

Within each language, batches are ordered by sha1(batch_id) and cut 80/10/10, so
the split is exact per language, reproducible, and independent of input order.
"""

import hashlib
from collections import defaultdict
from typing import Iterable

from .types import Language, Source, Split


class BatchSplitter:
    def __init__(self, train: float = 0.8, dev: float = 0.1, test: float = 0.1):
        if min(train, dev, test) < 0 or abs(train + dev + test - 1.0) > 1e-9:
            raise ValueError("split ratios must be non-negative and sum to 1")
        self.train, self.dev = train, dev

    def assign(self, sources: Iterable[Source]) -> dict[str, Split]:
        by_lang: dict[Language, set[str]] = defaultdict(set)
        for s in sources:
            by_lang[s.language].add(s.batch_id)
        out: dict[str, Split] = {}
        for ids in by_lang.values():
            ordered = sorted(ids, key=lambda b: hashlib.sha1(b.encode()).hexdigest())
            n_train = round(len(ordered) * self.train)
            n_dev = round(len(ordered) * self.dev)
            for i, b in enumerate(ordered):
                out[b] = Split.TRAIN if i < n_train else Split.DEV if i < n_train + n_dev else Split.TEST
        return out
