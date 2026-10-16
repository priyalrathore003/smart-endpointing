from collections import Counter
from dataclasses import replace

import pytest

from conftest import make_source, make_utt
from endpointing.data.conflicts import label_conflicts
from endpointing.data.leakage import check_leakage
from endpointing.data.split import BatchSplitter
from endpointing.data.truncate import Truncator
from endpointing.data.types import Language, Split


# --- split -------------------------------------------------------------------

def sources(n_per_lang):
    return [make_source(lang, f"{lang}-{i}") for lang, n in n_per_lang.items() for i in range(n)]


def test_split_is_exact_per_language():
    assign = BatchSplitter().assign(sources({Language.EN: 128, Language.HINGLISH: 154}))
    for lang, expected in [(Language.EN, (102, 13, 13)), (Language.HINGLISH, (123, 15, 16))]:
        c = Counter(s for b, s in assign.items() if b.startswith(lang.value + "-"))
        assert (c[Split.TRAIN], c[Split.DEV], c[Split.TEST]) == expected


def test_split_ignores_input_order_and_duplicates():
    srcs = sources({Language.HI: 40})
    a = BatchSplitter().assign(srcs)
    assert a == BatchSplitter().assign(list(reversed(srcs)) + srcs)


def test_single_batch_goes_to_train():
    assert BatchSplitter().assign(sources({Language.EN: 1})) == {"en-0": Split.TRAIN}


@pytest.mark.parametrize("ratios", [(0.8, 0.1, 0.2), (1.1, -0.1, 0.0)])
def test_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        BatchSplitter(*ratios)


# --- leakage -----------------------------------------------------------------

def rows(utts, split_of):
    t = Truncator()
    return [(split_of[u.source.batch_id], ex) for u in utts for ex in t.examples(u)]


A = make_utt("i want to block my debit card right now please", batch_id="b1")
B = make_utt("where is my order it was due yesterday", batch_id="b2", intent="where_is_order")
CLEAN = {"b1": Split.TRAIN, "b2": Split.TEST}


def test_clean_split_passes():
    assert check_leakage(rows([A, B], CLEAN)).ok


def test_detects_prefix_of_one_utterance_in_two_splits():
    r = rows([A], {"b1": Split.TRAIN})
    r[0] = (Split.TEST, r[0][1])                        # move one prefix to test
    report = check_leakage(r)
    assert report.utterance_ids_in_several_splits == [A.utterance_id]
    assert report.batch_ids_in_several_splits == ["b1"] and not report.ok


def test_detects_same_full_text_in_two_splits():
    copy = replace(A, utterance_id="other", source=replace(A.source, batch_id="b9"))
    report = check_leakage(rows([A, copy], {"b1": Split.TRAIN, "b9": Split.DEV}))
    assert report.full_texts_in_several_splits == [A.text]


def test_detects_near_duplicate_across_splits():
    near = make_utt("i want to block my debit card right now please sir", batch_id="b3")
    report = check_leakage(rows([A, near], {"b1": Split.TRAIN, "b3": Split.TEST}))
    assert report.near_duplicates_across_splits == [(A.text, near.text)]


def test_detects_prefix_without_its_end_row():
    r = [x for x in rows([A], {"b1": Split.TRAIN}) if x[1].k < x[1].n]
    assert check_leakage(r).prefixes_without_end_row == [A.utterance_id]


# --- label conflicts ---------------------------------------------------------

def test_label_conflicts_counted_per_language():
    utts = [make_utt("haan", language=Language.HINGLISH, batch_id="b1"),
            make_utt("haan mera number hai 98765", language=Language.HINGLISH, batch_id="b2"),
            make_utt("yes", batch_id="b3")]
    exs = [ex for u in utts for ex in Truncator(seed=0).examples(u)]
    exs = [e for e in exs if e.k in (1, e.n)]          # keep "haan" both as END and as a prefix
    rep = label_conflicts(exs)
    assert rep["by_language"]["hinglish"]["conflicting_texts"] == 1
    assert rep["by_language"]["en"] == {"distinct_texts": 1, "conflicting_texts": 0, "rate": 0.0}
    assert rep["overall"]["conflicting_texts"] == 1 and rep["examples"] == ["haan"]


def test_label_conflicts_empty():
    assert label_conflicts([])["overall"] == {"distinct_texts": 0, "conflicting_texts": 0, "rate": 0.0}
