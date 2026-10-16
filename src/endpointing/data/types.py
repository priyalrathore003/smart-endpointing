"""Fixed value sets and plain records for the data pipeline."""

from dataclasses import dataclass
from enum import StrEnum


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


@dataclass(frozen=True)
class Utterance:
    """A complete, normalised customer turn.

    digit_spans / spelled_spans are [start, end) token ranges of digit runs; they
    are filled by DigitSpeller and used to tag prefixes cut inside a number.
    """

    utterance_id: str
    source: Source
    text: str
    digit_spans: tuple[tuple[int, int], ...] = ()
    spelled_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.text or self.text != " ".join(self.text.split()):
            raise ValueError("Utterance.text must be non-empty and whitespace-normalised")


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

    def __post_init__(self) -> None:
        if not 1 <= self.k <= self.n:
            raise ValueError(f"need 1 <= k <= n, got k={self.k} n={self.n}")
        if (self.label is Label.END) != (self.k == self.n):
            raise ValueError("label must be END exactly when the prefix is the whole utterance")
