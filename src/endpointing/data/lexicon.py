"""Word lists per language. Every entry passes through `normalize`, so it matches data exactly.

HARD_NEGATIVES: a prefix ending in one of these is almost never a finished turn.
"na" and "hai" are left out on purpose: they often end complete Hindi sentences.

VERB_FINAL: a prefix ending in one of these *looks* finished (Hindi is verb-final).
This drives the `heuristic_complete` tag only. It is not a label; it lets us count
NOT_END prefixes that a reader would call complete. The English list is the
weakest: English has no reliable sentence-final word class.
"""

from typing import Iterable, Sequence

from ..text import normalize
from .types import Language


def _phrases(entries: Iterable[str]) -> frozenset[tuple[str, ...]]:
    return frozenset(tuple(normalize(e).split()) for e in entries)


def _words(entries: Iterable[str]) -> frozenset[str]:
    return frozenset(normalize(e) for e in entries)


HARD_NEGATIVES: dict[Language, frozenset[tuple[str, ...]]] = {
    Language.EN: _phrases([
        "and", "but", "because", "so", "or", "then", "that", "which", "like", "um", "uh", "umm",
        "i mean", "actually", "to", "the", "of", "for", "with", "my number is", "the amount is"]),
    Language.HI: _phrases([
        "और", "लेकिन", "क्योंकि", "तो", "कि", "मतलब", "यानी", "जैसे", "अम्म", "वो",
        "मेरा नंबर है", "अमाउंट है"]),
    Language.HINGLISH: _phrases([
        "aur", "lekin", "kyunki", "toh", "ki", "matlab", "yaani", "jaise", "umm", "woh",
        "mera number hai", "amount hai"]),
}

VERB_FINAL: dict[Language, frozenset[str]] = {
    # Requested forms first, then common spelling/inflection variants of the same verbs.
    Language.HI: _words(["है", "था", "चाहिए", "किया", "हूँ", "हैं", "दीजिए", "गया",
                         "हूं", "थी", "थे", "गई", "गए", "रहा", "रही", "रहे"]),
    Language.HINGLISH: _words(["hoon", "hain", "dijiye", "gaya", "raha", "karo",
                               "hu", "hun", "hai", "dijie", "gayi", "gaye", "rahi", "rahe", "tha", "thi"]),
    Language.EN: _words(["please", "now", "today", "tomorrow", "yesterday", "yet", "it", "me", "them",
                         "him", "her", "us", "done", "ok", "okay", "yes", "no", "thanks", "there",
                         "here", "again", "already", "sir", "madam"]),
}

DIGIT_WORDS: dict[Language, tuple[str, ...]] = {
    Language.EN: tuple(normalize(w) for w in "zero one two three four five six seven eight nine".split()),
    Language.HI: tuple(normalize(w) for w in "ज़ीरो एक दो तीन चार पाँच छह सात आठ नौ".split()),
    Language.HINGLISH: tuple(normalize(w) for w in "zero ek do teen char paanch chhe saat aath nau".split()),
}

_MAX_PHRASE = max(len(p) for ps in HARD_NEGATIVES.values() for p in ps)


def ends_with_hard_negative(tokens: Sequence[str], language: Language) -> bool:
    phrases = HARD_NEGATIVES[language]
    return any(tuple(tokens[-m:]) in phrases for m in range(1, min(_MAX_PHRASE, len(tokens)) + 1))


def heuristic_complete(tokens: Sequence[str], language: Language) -> bool:
    """True if the text ends like a finished sentence and not like a hard negative."""
    return bool(tokens) and tokens[-1] in VERB_FINAL[language] \
        and not ends_with_hard_negative(tokens, language)
