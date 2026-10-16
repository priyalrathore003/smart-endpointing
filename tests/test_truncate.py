import pytest

from conftest import make_utt
from endpointing.data.digits import DigitSpeller, digit_runs
from endpointing.data.lexicon import ends_with_hard_negative, heuristic_complete
from endpointing.data.truncate import TruncationConfig, Truncator
from endpointing.data.types import Example, Label, Language

EN, HI, HG = Language.EN, Language.HI, Language.HINGLISH


# --- lexicon -----------------------------------------------------------------

@pytest.mark.parametrize("text, lang, expected", [
    ("i want to pay and", EN, True),
    ("my number is", EN, True),
    ("so my number is", EN, True),
    ("i want to pay", EN, False),
    ("mera number hai", HG, True),
    ("paise kat gaye lekin", HG, True),
    ("theek hai na", HG, False),
    ("मेरा नंबर है", HI, True),
    ("पैसा कट गया लेकिन", HI, True),
    ("पैसा कट गया", HI, False),
    ("", EN, False),
])
def test_hard_negative_endings(text, lang, expected):
    assert ends_with_hard_negative(text.split(), lang) is expected


@pytest.mark.parametrize("text, lang, expected", [
    ("मुझे नया कार्ड चाहिए", HI, True),
    ("पैसा कट गया", HI, True),
    ("मैं बैंक से बात कर रहा हूँ", HI, True),
    ("मेरा नंबर है", HI, False),         # verb-final but also a hard negative
    ("मुझे नया", HI, False),
    ("refund kab milega bata dijiye", HG, True),
    ("mera number hai", HG, False),
    ("mera account", HG, False),
    ("please block it", EN, True),
    ("please block the", EN, False),
    ("", EN, False),
    # added verb-final forms
    ("मेरा रिफंड कब तक मिलेगा", HI, True),
    ("पेमेंट कल तक हो जाएगा या नहीं पता नहीं क्या हुआ", HI, True),
    ("ऑर्डर कब डिलीवर होगा", HI, True),
    ("कृपया जल्दी कार्ड ब्लॉक कर दें", HI, True),
    ("recharge karwaya par credit nahi hua", HG, True),
    ("transfer ki thi par paise kat gaye aur refund nahi hui", HG, True),
    ("mere bhai ke saath bhi aisa hi hue", HG, True),
    ("refund kab tak hoga", HG, True),
    ("naya card kab milega", HG, True),
    ("link bhej dein", HG, True),
    # future first person, "sakte ho", negation-final endings
    ("मैं कल तक पूरा पैसा दे दूंगा", HI, True),
    ("main kal tak payment kar dunga", HG, True),
    ("main shaam tak transfer karunga", HG, True),
    ("क्या आप मेरा पार्सल ट्रैक कर सकते हो", HI, True),
    ("kya aap mera parcel track kar sakte ho", HG, True),
    ("पता नहीं", HI, True),
    ("pata nahi", HG, True),
    ("otp abhi tak aaya nahi", HG, True),
    ("हो", HI, False),                                 # "हो" alone is not "सकते हो"
    ("नहीं तो", HI, False),                            # hard negative wins
    # English structural rule
    ("i want to report an unauthorized transfer", EN, True),
    ("there is a debit of four thousand rupees", EN, True),
    ("i want to report an", EN, False),
    ("i did not initiate the deduction of", EN, False),
    ("can you explain why eighteen hundred", EN, False),
    ("i just saw a debit of five thousand five", EN, False),
    ("my account number is 1234", EN, False),
    ("block card", EN, False),                       # too short for the structural rule
    ("i want to pay because", EN, False),            # hard negative wins
])
def test_heuristic_complete(text, lang, expected):
    assert heuristic_complete(text.split(), lang) is expected


# --- digits ------------------------------------------------------------------

def test_digit_runs():
    assert digit_runs("call 98765 43210 now 42".split()) == [(1, 3), (4, 5)]
    assert digit_runs([]) == []
    assert digit_runs(["12a"]) == []


@pytest.mark.parametrize("lang, expected", [
    (EN, "otp is four two"),
    (HG, "otp is char do"),
    (HI, "otp is चार दो"),
])
def test_spelling_per_language(lang, expected):
    out = DigitSpeller(spell_prob=1.0).apply(make_utt("otp is 42", language=lang))
    assert out.text == expected
    assert out.digit_spans == out.spelled_spans == ((2, 4),)


def test_spell_prob_zero_keeps_digits_but_records_spans():
    out = DigitSpeller(spell_prob=0.0).apply(make_utt("call 98765 43210 now"))
    assert out.text == "call 98765 43210 now"
    assert out.digit_spans == ((1, 3),) and out.spelled_spans == ()


def test_spelling_rate_is_about_a_quarter_and_deterministic():
    utts = [make_utt(f"otp {i} please") for i in range(2000)]
    speller = DigitSpeller(spell_prob=0.25, seed=7)
    spelled = [bool(speller.apply(u).spelled_spans) for u in utts]
    assert 0.22 < sum(spelled) / len(spelled) < 0.28
    assert spelled == [bool(DigitSpeller(0.25, seed=7).apply(u).spelled_spans) for u in utts]


def test_invalid_spell_prob():
    with pytest.raises(ValueError):
        DigitSpeller(spell_prob=1.5)


# --- truncation --------------------------------------------------------------

def rows_for(text, lang=EN, spell=0.0, **cfg):
    utt = DigitSpeller(spell_prob=spell).apply(make_utt(text, language=lang))
    return Truncator(TruncationConfig(**cfg)).examples(utt)


def test_end_row_is_the_full_utterance():
    rows = rows_for("i want to check my balance please")
    ends = [r for r in rows if r.label is Label.END]
    assert len(ends) == 1 and ends[0].text == "i want to check my balance please" and ends[0].k == ends[0].n


def test_prefix_count_and_labels():
    rows = rows_for("i want to check my balance please")   # hard negative only at "to"
    not_end = [r for r in rows if r.label is Label.NOT_END]
    assert len(not_end) == 1 + 3                            # all hard + 3 random
    assert all(r.k < r.n for r in not_end)
    assert len({r.k for r in rows}) == len(rows)            # no duplicate cuts


def test_every_hard_negative_prefix_is_kept():
    rows = rows_for("umm and so because i mean the thing", max_random_prefixes=0)
    assert sorted(r.k for r in rows if r.hard_negative) == [1, 2, 3, 4, 6, 7]


def test_one_word_utterance_gives_only_end():
    rows = rows_for("yes")
    assert [(r.label, r.short) for r in rows] == [(Label.END, True)]


def test_digits_mid_and_spelled_tags():
    rows = rows_for("my number is 98765 43210", spell=1.0, max_random_prefixes=20)
    by_k = {r.k: r for r in rows}
    assert by_k[3].hard_negative and not by_k[3].has_digits
    assert by_k[5].digits_mid and by_k[5].digits_spelled and by_k[5].label is Label.NOT_END
    assert not by_k[13].digits_mid and by_k[13].label is Label.END     # 3 words + 10 digit words


def test_truncation_is_deterministic():
    a = rows_for("please send me the payment link on whatsapp today")
    b = rows_for("please send me the payment link on whatsapp today")
    assert a == b


def test_example_rejects_invalid_states():
    row = rows_for("check balance")[0]
    with pytest.raises(ValueError):
        Example(**{**row.__dict__, "k": 0})
    with pytest.raises(ValueError):
        Example(**{**row.__dict__, "label": Label.END if row.label is Label.NOT_END else Label.NOT_END})


def test_invalid_config():
    with pytest.raises(ValueError):
        TruncationConfig(max_random_prefixes=-1)


def test_code_mixed_tag_is_utterance_level_and_hindi_only():
    hi_rows = rows_for("मेरा parcel कहां तक पहुंचा है", lang=HI)
    assert hi_rows and all(r.code_mixed for r in hi_rows)
    assert not any(r.code_mixed for r in rows_for("मेरा पार्सल कहां तक पहुंचा है", lang=HI))
    assert not any(r.code_mixed for r in rows_for("mera parcel kahan hai", lang=HG))


def test_complete_utterance_ending_on_hard_negative_stays_end_and_is_tagged():
    rows = rows_for("i did not make this payment and i did not do that", max_random_prefixes=20)
    end = [r for r in rows if r.label is Label.END]
    assert len(end) == 1 and end[0].end_on_hard_negative and not end[0].hard_negative
    assert not any(r.end_on_hard_negative for r in rows if r.label is Label.NOT_END)
    assert any(r.hard_negative for r in rows if r.label is Label.NOT_END)   # "...payment and"


def test_ordinary_end_row_is_not_tagged():
    assert not any(r.end_on_hard_negative for r in rows_for("please block my card"))
