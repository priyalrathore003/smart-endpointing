import json
import re
import threading
from collections import Counter
from decimal import Decimal

import pytest

import endpointing.text
import endpointing.data.build as build_module
from endpointing.data.budget import GPT_41_MINI, Pricing, SpendLedger
from endpointing.data.build import BuildConfig, ContextBalanceError, build, context_balance, script_ok
from endpointing.data.catalogue import PILOT_BATCH_IDS, length_bucket, pilot_batches, plan_batches
from endpointing.data.cli import estimate, main
from endpointing.data.generator import Completion, GeneratedItem, parse_response
from endpointing.data.prompts import prompt_version, render, shape_of
from endpointing.data.runner import BatchRunner, StaleCacheError
from endpointing.data.types import Language, Style

FAKE_TURNS = {
    "en": ["I want to block my card because it was stolen.", "My number is 98765 43210.",
           "Can you check my balance please?", "Yes.", "I did not make this payment of 5,000 rupees."],
    "hi": ["मेरा कार्ड खो गया है।", "मैं कल तक पैसे जमा कर दूँगा।", "हाँ जी।", "मेरा parcel कहां तक पहुंचा है?",
           "Latin text in a Hindi batch"],
    "hinglish": ["mera recharge credit nahi hua", "haan ji", "otp 4521 nahi aa raha",
                 "maine complaint ki", "kal tak payment kar dunga"],
}
FAKE_PAIRS = {
    "en": [("will you pay by friday", "yes"), ("when can you pay", "by friday"),
           ("can i confirm your number", "98765 43210")],
    "hi": [("क्या आप कल भुगतान करेंगे", "हाँ जी"), ("कब तक करेंगे", "कल तक"), ("which date", "कल")],
    "hinglish": [("kya aap kal payment karenge", "haan ji"), ("kab tak karenge", "kal tak"),
                 ("otp bataiye", "4521")],
}


class FakeGenerator:
    model, temperature, max_output_tokens = "fake-model", 1.0, 2000

    def __init__(self, bad_responses=0):
        self.calls, self.bad_left = 0, bad_responses
        self.lock = threading.Lock()

    def complete(self, system, user, seed, response_format):
        with self.lock:
            self.calls += 1
            bad = self.bad_left > 0
            self.bad_left -= int(bad)
        lang = "hi" if "Devanagari" in user else "hinglish" if "code-mixed" in user else "en"
        if response_format["json_schema"]["name"] == "pairs":
            body = {"short": [{"agent_question": q, "reply": r} for q, r in FAKE_PAIRS[lang]], "medium": []}
        else:
            turns = [{"text": x, "context": "aapki kya madad karun" if i == 0 else ""}
                     for i, x in enumerate(FAKE_TURNS[lang])]
            body = {"short": turns[:1], "medium": turns[1:], "long": []}
        text = "not json" if bad else json.dumps(body, ensure_ascii=False)
        return Completion(text=text, input_tokens=400, output_tokens=600, model=self.model)


def ledger(tmp_path, cap="1.20"):
    return SpendLedger(tmp_path / "raw" / "spend.jsonl", GPT_41_MINI, Decimal(cap))


# --- catalogue and prompts ---------------------------------------------------

def test_plan_has_480_unique_batches_and_30_30_40_mix():
    specs = plan_batches()
    assert len(specs) == 480 and len({s.batch_id for s in specs}) == 480
    by_lang = Counter()
    for s in specs:
        by_lang[s.language] += s.n
    assert by_lang == {Language.EN: 2880, Language.HI: 2880, Language.HINGLISH: 3840}
    assert Counter(s.style for s in specs)[Style.SHORT_REPLIES] == 96


def test_pilot_has_the_full_runs_turn_to_reply_mix_in_every_language():
    pilot = pilot_batches()
    assert [s.batch_id for s in pilot] == list(PILOT_BATCH_IDS) and len(pilot) == 15
    for lang in Language:
        styles = [s.style for s in pilot if s.language is lang]
        assert styles.count(Style.SHORT_REPLIES) == 1 and len(styles) == 5      # 4 turn batches : 1


@pytest.mark.parametrize("style, n, expected", [(Style.PLAIN, 18, 5), (Style.NAMES, 24, 7),
                                                (Style.SHORT_REPLIES, 18, 18)])
def test_context_count(style, n, expected):
    spec = next(s for s in plan_batches() if s.style is style and s.n == n)
    assert spec.context_count() == expected


@pytest.mark.parametrize("style, n, expected", [
    (Style.PLAIN, 18, {"short": 4, "medium": 10, "long": 4}),
    (Style.FILLERS, 24, {"short": 6, "medium": 12, "long": 6}),
    (Style.SHORT_REPLIES, 18, {"short": 14, "medium": 4, "long": 0}),
    (Style.SHORT_REPLIES, 24, {"short": 18, "medium": 6, "long": 0}),
])
def test_bucket_counts_are_exact_and_sum_to_n(style, n, expected):
    spec = next(s for s in plan_batches() if s.style is style and s.n == n)
    assert spec.bucket_counts() == expected and sum(expected.values()) == n


@pytest.mark.parametrize("n_words, bucket", [(1, "short"), (4, "short"), (5, "medium"), (12, "medium"),
                                             (13, "long"), (30, "long"), (45, "long"), (0, "empty")])
def test_length_bucket(n_words, bucket):
    assert length_bucket(n_words) == bucket


def test_turn_prompt_asks_for_exact_counts_with_examples_in_script():
    spec = next(s for s in plan_batches() if s.batch_id == "hi.delivery.where_is_order.numbers")
    p = render(spec)
    assert "complete customer turn" in p.system and p.response_format["json_schema"]["name"] == "turns"
    assert "exactly 4 short turns of 1-4 words (for example 'हाँ जी', 'कल तक', 'ठीक है')" in p.user
    assert "exactly 10 medium" in p.user and "exactly 4 long" in p.user
    assert "1 in 4 turns keep one common English noun" in p.user
    assert 'For exactly 5 of the turns, spread over the three lengths, also write in "context"' in p.user
    assert "Colloquial spoken Hindi" in p.user and "कृपया" in p.user and "one topic" in p.user


def test_short_reply_prompt_asks_for_pairs_and_ambiguous_replies():
    spec = next(s for s in plan_batches() if s.batch_id == "hinglish.delivery.reschedule.short_replies")
    p = render(spec)
    assert p.response_format["json_schema"]["name"] == "pairs"
    assert "exactly 18 pairs whose reply is 1-4 words" in p.user and "exactly 6 pairs whose reply is 5-12" in p.user
    assert "At least 8 replies" in p.user and "'haan ji', 'theek hai', 'kal tak'" in p.user
    assert "bare number" in p.user


def test_prompt_versions_are_per_shape_and_stable():
    assert prompt_version("turns") == prompt_version("turns") and len(prompt_version("pairs")) == 16
    assert prompt_version("turns") != prompt_version("pairs")
    assert shape_of(Style.SHORT_REPLIES) == "pairs" and shape_of(Style.PLAIN) == "turns"
    with pytest.raises(ValueError):
        prompt_version("other")


# --- parsing and money -------------------------------------------------------

def test_parse_turns():
    items = parse_response('{"short": [{"text": " haan ji ", "context": "kal tak karenge?"},'
                           ' {"text": "", "context": ""}],'
                           ' "medium": [{"text": "a b c d e", "context": "  "}], "long": []}', Style.PLAIN)
    assert items == [GeneratedItem("haan ji", "kal tak karenge?", "short"),
                     GeneratedItem("a b c d e", None, "medium")]


def test_parse_pairs():
    items = parse_response('{"short": [{"agent_question": "kab tak?", "reply": "kal tak"},'
                           ' {"agent_question": "", "reply": "x"}],'
                           ' "medium": [{"agent_question": "q", "reply": "kal shaam tak kar dunga sir"}]}',
                           Style.SHORT_REPLIES)
    assert items == [GeneratedItem("kal tak", "kab tak?", "short"),
                     GeneratedItem("kal shaam tak kar dunga sir", "q", "medium")]


@pytest.mark.parametrize("bad, style", [
    ("not json", Style.PLAIN), ("[]", Style.PLAIN), ('{"short": [], "medium": []}', Style.PLAIN),
    ('{"short": [1], "medium": [], "long": []}', Style.PLAIN),
    ('{"short": ["plain string"], "medium": [], "long": []}', Style.PLAIN),
    ('{"short": [{"text": "x"}], "medium": [], "long": []}', Style.PLAIN),
    ('{"short": "x", "medium": []}', Style.SHORT_REPLIES),
    ('{"short": [{"reply": 3, "agent_question": "q"}], "medium": []}', Style.SHORT_REPLIES),
    ('{"pairs": []}', Style.SHORT_REPLIES), ('{"short": [], "long": []}', Style.SHORT_REPLIES),
])
def test_parse_rejects_wrong_shapes(bad, style):
    with pytest.raises(ValueError):
        parse_response(bad, style)


def test_pricing_uses_exact_decimals():
    assert GPT_41_MINI.cost(1_000_000, 1_000_000) == Decimal("2.00")
    assert GPT_41_MINI.cost(400, 600) == Decimal("0.00112")
    with pytest.raises(ValueError):
        Pricing(Decimal(1), Decimal(1)).cost(-1, 0)


def test_ledger_persists_spend(tmp_path):
    l1 = ledger(tmp_path)
    r = Decimal("0.01")
    assert l1.try_reserve(r)
    l1.settle(r, "b", "m", 400, 600, True)
    assert ledger(tmp_path).spent == Decimal("0.00112")


def test_ledger_refuses_reservation_over_cap(tmp_path):
    l = ledger(tmp_path, cap="0.01")
    assert l.try_reserve(Decimal("0.006"))
    assert not l.try_reserve(Decimal("0.006"))     # 0.012 > 0.01 while the first is still reserved
    l.release(Decimal("0.006"))
    assert l.try_reserve(Decimal("0.006"))


# --- runner ------------------------------------------------------------------

def test_runner_writes_raw_and_caches(tmp_path):
    gen, l = FakeGenerator(), ledger(tmp_path)
    s1 = BatchRunner(gen, l, tmp_path / "raw").run(pilot_batches())
    assert len(s1.called) == 15 and gen.calls == 15
    raw = json.loads((tmp_path / "raw" / f"{PILOT_BATCH_IDS[0]}.json").read_text())
    assert [i["text"] for i in raw["items"]] == [t.strip() for t in FAKE_TURNS["en"]]
    assert raw["items"][0]["context"] == "aapki kya madad karun" and raw["items"][1]["context"] is None
    assert raw["prompt_version"] == prompt_version("turns") and raw["prompt_shape"] == "turns"
    assert raw["bucket_counts_requested"]["short"] == 4 and raw["context_count_requested"] == 5
    pairs = json.loads((tmp_path / "raw" / "hinglish.delivery.reschedule.short_replies.json").read_text())
    assert pairs["items"][0] == {"text": "haan ji", "context": "kya aap kal payment karenge",
                                 "requested_bucket": "short"}
    s2 = BatchRunner(gen, l, tmp_path / "raw").run(pilot_batches())
    assert len(s2.cached) == 15 and gen.calls == 15
    assert l.spent == 15 * GPT_41_MINI.cost(400, 600)


def test_runner_refuses_stale_cache_before_any_call(tmp_path):
    BatchRunner(FakeGenerator(), ledger(tmp_path), tmp_path / "raw").run(pilot_batches())
    path = tmp_path / "raw" / f"{PILOT_BATCH_IDS[0]}.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "prompt_version": "old"}))
    gen = FakeGenerator()
    with pytest.raises(StaleCacheError):
        BatchRunner(gen, ledger(tmp_path), tmp_path / "raw").run(plan_batches())
    assert gen.calls == 0


def test_runner_stops_at_budget_cap(tmp_path):
    # Reserve worst case $0.0038 per call, settle at actual $0.00112. Sequentially with a $0.005 cap:
    # call 1 (0 + 0.0038 ok), call 2 (0.00112 + 0.0038 ok), call 3 (0.00224 + 0.0038 > cap) -> stop.
    gen = FakeGenerator()
    s = BatchRunner(gen, ledger(tmp_path, cap="0.005"), tmp_path / "raw", workers=1).run(pilot_batches())
    assert len(s.called) == 2 and len(s.skipped_budget) == 13 and gen.calls == 2


def test_concurrent_workers_never_exceed_cap(tmp_path):
    gen, l = FakeGenerator(), ledger(tmp_path, cap="0.02")
    s = BatchRunner(gen, l, tmp_path / "raw", workers=8).run(plan_batches()[:40])
    # The exact count depends on thread timing; the invariant does not.
    assert l.spent <= Decimal("0.02") and s.skipped_budget and gen.calls == len(s.called)


def test_bad_response_is_retried_once_and_still_paid_for(tmp_path):
    gen, l = FakeGenerator(bad_responses=1), ledger(tmp_path)
    s = BatchRunner(gen, l, tmp_path / "raw", workers=1).run(pilot_batches()[:1])
    assert s.called == [PILOT_BATCH_IDS[0]] and gen.calls == 2
    assert [e["parsed_ok"] for e in l.entries()] == [False, True]


def test_twice_bad_response_marks_batch_failed(tmp_path):
    s = BatchRunner(FakeGenerator(bad_responses=2), ledger(tmp_path), tmp_path / "raw",
                    workers=1).run(pilot_batches()[:1])
    assert s.failed == [PILOT_BATCH_IDS[0]] and not (tmp_path / "raw" / f"{PILOT_BATCH_IDS[0]}.json").exists()


def test_estimate_uses_observed_usage(tmp_path):
    assert estimate(pilot_batches(), tmp_path / "raw")["basis"]["en"] == "prior"
    BatchRunner(FakeGenerator(), ledger(tmp_path), tmp_path / "raw").run(pilot_batches())
    e = estimate(plan_batches(), tmp_path / "raw")
    assert e["cached"] == 15 and e["calls"] == 465 and e["basis"]["hi"] == "observed over 5 calls"


# --- build -------------------------------------------------------------------

def run_pilot_build(tmp_path, name="pilot", **cfg):
    BatchRunner(FakeGenerator(), ledger(tmp_path), tmp_path / "raw").run(pilot_batches())
    # The fake data is not balanced; the balance check has its own tests below.
    return build(tmp_path / "raw", tmp_path, name, BuildConfig(**{"max_context_gap": 1.0, **cfg}))


def test_build_end_to_end(tmp_path):
    m = run_pilot_build(tmp_path)
    assert m["leakage_checks"] == "passed"
    # per hi turn batch: the Latin line + the Latin-script context on the first turn (4 x 2), + 1 Latin question
    assert m["drops"]["script_mismatch"] == {"hi": 9}
    assert "ends_with_hard_negative" not in m["drops"]                 # kept as END, tagged instead
    assert m["counts"]["slices"]["train"]["end_on_hard_negative"] >= 1  # "maine complaint ki"
    assert m["dedup"]["exact_removed"] > 0                             # same fake text in both batches
    assert (tmp_path / "manifests" / "pilot.json").exists()
    rows = [json.loads(l) for l in (tmp_path / "processed" / "pilot" / "train.jsonl").read_text().splitlines()]
    assert all(r["text"] == endpointing.text.normalize(r["text"]) for r in rows)
    assert {r["label"] for r in rows} == {"END", "NOT_END"}
    replies = [r for r in rows if r["style"] == "short_replies"]
    assert replies and all(r["context"] for r in replies)
    turn_rows = [r for r in rows if r["style"] != "short_replies"]
    assert any(r["context"] for r in turn_rows) and any(r["context"] is None for r in turn_rows)
    assert set(m["context_balance"]["by_language"]) == {"en", "hi", "hinglish"}
    assert m["counts"]["slices"]["train"]["code_mixed"] >= 1               # "मेरा parcel ..."
    lb = m["length_buckets"]
    assert lb["short_replies"]["hinglish"]["short"] >= 2 and lb["requested_bucket_agreement"]["asked"] > 0
    assert m["utterances_by_style"]["short_replies"] > 0


def test_build_is_deterministic(tmp_path):
    a = run_pilot_build(tmp_path, "a")["data_hash"]
    b = build(tmp_path / "raw", tmp_path, "a", BuildConfig(max_context_gap=1.0))["data_hash"]
    assert a == b


def test_build_uses_the_shared_normaliser():
    assert build_module.normalize is endpointing.text.normalize


@pytest.mark.parametrize("text, lang, ok", [
    ("मेरा कार्ड खो गया", Language.HI, True), ("mera card", Language.HI, False),
    ("mera card", Language.HINGLISH, True), ("मेरा card", Language.HINGLISH, False),
    ("98765", Language.HI, True),
    ("मेरा parcel कहां तक पहुंचा है", Language.HI, True),        # 1 of 6 tokens Latin
    ("मेरा parcel status बताओ", Language.HI, False),             # 2 of 4 (50%)
    ("मेरा ओटीपी नहीं आया yaar", Language.HI, True),             # 1 of 5 (20%)
    ("आज ओटीपी नहीं yaar", Language.HI, True),                   # exactly 25% is allowed
    ("haan matlab main tuesday ko payment kar dunga", Language.HI, False),   # wholly Latin
    ("", Language.HI, True),
])
def test_script_check(text, lang, ok):
    assert script_ok(text, lang) is ok


# --- cli ---------------------------------------------------------------------

def test_generate_without_yes_makes_no_call(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "generate", "--pilot"]) == 2
    assert "Not calling the API" in capsys.readouterr().out
    assert not (tmp_path / "raw").exists()


def test_plan_command_runs_offline(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "plan"]) == 0
    out = capsys.readouterr().out
    assert "480 calls" in out and "of cap $1.20" in out


@pytest.mark.parametrize("batch_id, script, must_have, must_not_have", [
    ("hi.collections.promise_to_pay.fillers", "Devanagari script, apart from that occasional English noun",
     "मंगलवार, नहीं, बुधवार", "Tuesday"),
    ("hinglish.telecom.recharge_not_credited.fillers", "Latin script", "Tuesday, nahi, Wednesday", "मतलब"),
    ("en.banking.lost_card.fillers", "Latin script", "I mean", "matlab"),
])
def test_filler_examples_and_script_rule_are_per_language(batch_id, script, must_have, must_not_have):
    spec = next(s for s in plan_batches() if s.batch_id == batch_id)
    user = render(spec).user
    assert user.rstrip().endswith(f"Every turn and question must be in {script}.")
    assert must_have in user and must_not_have not in user


def test_every_planned_prompt_renders_without_placeholders():
    for spec in plan_batches():
        assert not re.search(r"\{[a-z_]+\}", render(spec).user), spec.batch_id


def test_build_refuses_mixed_prompt_versions(tmp_path):
    BatchRunner(FakeGenerator(), ledger(tmp_path), tmp_path / "raw").run(pilot_batches())
    path = tmp_path / "raw" / f"{PILOT_BATCH_IDS[0]}.json"
    raw = json.loads(path.read_text())
    path.write_text(json.dumps({**raw, "prompt_version": "old"}))
    with pytest.raises(ValueError, match="mixes prompt versions"):
        build(tmp_path / "raw", tmp_path, "x", BuildConfig(max_context_gap=1.0))


def test_build_fails_on_context_gap_but_writes_the_manifest_first(tmp_path):
    with pytest.raises(ContextBalanceError, match="gap with vs without context"):
        run_pilot_build(tmp_path, max_context_gap=0.0)
    m = json.loads((tmp_path / "manifests" / "pilot.json").read_text())
    assert m["context_balance"]["failed"]


def test_context_balance_numbers():
    from conftest import make_utt
    from endpointing.data.truncate import Truncator
    t = Truncator()
    ex = [e for u in [make_utt("haan ji", context="kya aap kal payment karenge", style="short_replies"),
                      make_utt("i want to check my balance please and then block my card")]
          for e in t.all_prefixes(u)]
    b = context_balance(ex, max_gap=0.15)["by_language"]["en"]
    assert b["with_context"] == {"rows": 2, "p_end": 0.5}
    assert b["without_context"] == {"rows": 12, "p_end": round(1 / 12, 4)} and b["gap"] == round(0.5 - 1 / 12, 4)
    assert context_balance(ex, max_gap=0.15)["failed"] == ["en"]


def test_build_caps_prefixes_in_every_split_and_writes_analysis_file(tmp_path):
    m = run_pilot_build(tmp_path)
    rows = [json.loads(l) for split in ("train", "dev", "test")
            for l in (tmp_path / "processed" / "pilot" / f"{split}.jsonl").read_text().splitlines()]
    per_utt = Counter(r["utterance_id"] for r in rows if r["label"] == "NOT_END")
    assert max(per_utt.values()) <= 3
    assert all(r["prefix_of_reply"] == (r["style"] == "short_replies" and r["label"] == "NOT_END") for r in rows)

    analysis = tmp_path / "processed" / "pilot" / "analysis" / "all_prefixes.jsonl"
    every = [json.loads(l) for l in analysis.read_text().splitlines()]
    assert sum(r["sampled"] for r in every) == len(rows) and len(every) > len(rows)
    assert m["analysis_files"][0]["rows"] == len(every)
    assert all(f["path"] != m["analysis_files"][0]["path"] for f in m["files"])   # not in data_hash

    ps = m["prefix_sampling"]
    assert ps["after"]["all"]["END"] == ps["before_all_hard_plus_3_random"]["all"]["END"]
    assert ps["after"]["all"]["NOT_END"] <= ps["before_all_hard_plus_3_random"]["all"]["NOT_END"]
    assert ps["every_boundary"]["all"]["NOT_END"] >= ps["before_all_hard_plus_3_random"]["all"]["NOT_END"]
