"""Fixed value sets and plain records for the data pipeline."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class Language(StrEnum):
    EN = "en"
    HI = "hi"              # Hindi, Devanagari script
    HINGLISH = "hinglish"  # Hindi-English code-mixed, Latin script


class Domain(StrEnum):
    BANKING = "banking"
    COLLECTIONS = "collections"
    DELIVERY = "delivery"
    TELECOM = "telecom"


class Style(StrEnum):
    PLAIN = "plain"
    FILLERS = "fillers"
    NUMBERS = "numbers"
    NAMES = "names"
    SHORT_REPLIES = "short_replies"   # (agent question, customer reply) pairs; the question is `context`


class Label(StrEnum):
    END = "END"
    NOT_END = "NOT_END"


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass(frozen=True)
class Source:
    """Where an utterance came from: one generation call (batch) and its cell."""

    batch_id: str
    language: Language
    domain: Domain
    intent: str
    style: Style

    def __post_init__(self) -> None:
        # Coerce plain strings (e.g. read back from JSON); unknown values raise ValueError.
        object.__setattr__(self, "language", Language(self.language))
        object.__setattr__(self, "domain", Domain(self.domain))
        object.__setattr__(self, "style", Style(self.style))
        if not self.batch_id or not self.intent:
            raise ValueError("batch_id and intent must be non-empty")


def _is_clean_text(s: str) -> bool:
    return bool(s) and s == " ".join(s.split())


def dedup_key(text: str, context: Optional[str]) -> str:
    """Identity of a turn: the same reply to a different question is a different turn.
    Normalised text has no punctuation, so "|" cannot occur in either part."""
    return text if context is None else f"{context} | {text}"


@dataclass(frozen=True)
class Utterance:
    """A complete, normalised customer turn.

    context is the agent's question the turn answers (short-reply style only), else None.
    digit_spans / spelled_spans are [start, end) token ranges of digit runs; they
    are filled by DigitSpeller and used to tag prefixes cut inside a number.
    """

    utterance_id: str
    source: Source
    text: str
    context: Optional[str] = None
    digit_spans: tuple[tuple[int, int], ...] = ()
    spelled_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not _is_clean_text(self.text):
            raise ValueError("Utterance.text must be non-empty and whitespace-normalised")
        if self.context is not None and not _is_clean_text(self.context):
            raise ValueError("Utterance.context must be None or non-empty and whitespace-normalised")

    @property
    def key(self) -> str:
        return dedup_key(self.text, self.context)


@dataclass(frozen=True)
class Example:
    """One training row: a prefix of k of the utterance's n words, with its label."""

    utterance_id: str
    source: Source
    text: str
    label: Label
    k: int
    n: int
    hard_negative: bool
    heuristic_complete: bool
    has_digits: bool
    digits_mid: bool
    digits_spelled: bool
    short: bool
    code_mixed: bool
    end_on_hard_negative: bool
    prefix_of_reply: bool
    context: Optional[str] = None

    @property
    def key(self) -> str:
        return dedup_key(self.text, self.context)

    def __post_init__(self) -> None:
        if not 1 <= self.k <= self.n:
            raise ValueError(f"need 1 <= k <= n, got k={self.k} n={self.n}")
        if (self.label is Label.END) != (self.k == self.n):
            raise ValueError("label must be END exactly when the prefix is the whole utterance")
