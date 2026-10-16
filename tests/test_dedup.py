import pytest

from conftest import make_utt
from endpointing.data.dedup import Deduplicator, jaccard, shingles
from endpointing.data.types import Domain, Language

BASE = "mera account se paise kat gaye but transfer nahi hua abhi tak"


def test_exact_duplicates_removed_across_groups():
    a = make_utt("Haan ji.", language=Language.HINGLISH, batch_id="b1")
    b = make_utt("haan ji", language=Language.HINGLISH, batch_id="b2", domain=Domain.TELECOM)
    kept, rep = Deduplicator().dedupe([a, b])
    assert kept == [a] and rep.exact_removed == 1 and rep.near_removed == 0


def test_near_duplicate_in_same_group_removed_first_kept():
    a = make_utt(BASE, language=Language.HINGLISH)
    b = make_utt(BASE + " please", language=Language.HINGLISH)    # 10 of 11 shingles shared
    kept, rep = Deduplicator().dedupe([a, b])
    assert kept == [a] and rep.near_removed == 1 and rep.near_examples == ((a.text, b.text),)


def test_near_duplicate_in_other_group_is_kept():
    a = make_utt(BASE, language=Language.HINGLISH)
    b = make_utt(BASE + " please", language=Language.HINGLISH, domain=Domain.TELECOM)
    kept, _ = Deduplicator().dedupe([a, b])
    assert kept == [a, b]


def test_below_threshold_kept():
    a = make_utt("i want to block my debit card right now")
    b = make_utt("i want to check my loan emi due date")
    kept, rep = Deduplicator().dedupe([a, b])
    assert kept == [a, b] and rep.kept == 2 and rep.input == 2


def test_short_texts_only_match_exactly():
    assert shingles("haan ji") == frozenset([("haan", "ji")])
    kept, _ = Deduplicator().dedupe([make_utt("haan ji"), make_utt("haan")])
    assert len(kept) == 2


def test_jaccard_bounds():
    s = shingles(BASE)
    assert jaccard(s, s) == 1.0 and jaccard(s, shingles("x y z")) == 0.0


def test_invalid_threshold():
    with pytest.raises(ValueError):
        Deduplicator(threshold=0)
