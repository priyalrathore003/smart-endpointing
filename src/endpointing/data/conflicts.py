"""Label-conflict report: the same normalised text labelled END in one place and NOT_END in another.

Example: "haan" is a complete answer in one utterance and the start of
"haan mera number ..." in another. These rows are kept (the conflict is real
ambiguity, which the model's probability should reflect); the rate is reported.
Counted per (language, text), over distinct texts.
"""

from collections import defaultdict
from typing import Iterable

from .types import Example, Label


def label_conflicts(examples: Iterable[Example], max_examples: int = 10) -> dict:
    labels: dict[tuple[str, str], set[Label]] = defaultdict(set)
    for ex in examples:
        labels[(ex.source.language.value, ex.text)].add(ex.label)

    def summary(keys: list[tuple[str, str]]) -> dict:
        conflicting = [k for k in keys if len(labels[k]) > 1]
        return {"distinct_texts": len(keys), "conflicting_texts": len(conflicting),
                "rate": round(len(conflicting) / len(keys), 4) if keys else 0.0}

    keys = sorted(labels)
    by_language = {lang: summary([k for k in keys if k[0] == lang]) for lang in sorted({k[0] for k in keys})}
    examples_out = [text for lang, text in keys if len(labels[(lang, text)]) > 1][:max_examples]
    return {"overall": summary(keys), "by_language": by_language, "examples": examples_out}
