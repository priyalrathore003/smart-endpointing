import json
import threading
from collections import Counter
from decimal import Decimal

import pytest

import endpointing.data.build as build_module
import endpointing.text
from endpointing.data.budget import GPT_41_MINI, Pricing, SpendLedger
from endpointing.data.build import BuildConfig, build, script_ok
from endpointing.data.catalogue import PILOT_BATCH_IDS, pilot_batches, plan_batches
from endpointing.data.cli import estimate, main
from endpointing.data.generator import Completion, parse_utterances
from endpointing.data.prompts import prompt_version, render
from endpointing.data.runner import BatchRunner
from endpointing.data.types import Language

FAKE_UTTERANCES = {
    "en": ["I want to block my card because it was stolen.", "My number is 98765 43210.",
           "Can you check my balance please?", "Yes.", "I did not make this payment of 5,000 rupees."],
    "hi": ["मेरा कार्ड खो गया है।", "मैं कल तक पैसे जमा कर दूँगा।", "हाँ जी।", "मेरा ऑर्डर कहाँ है?",
           "Latin text in a Hindi batch"],
    "hinglish": ["mera recharge credit nahi hua", "haan ji", "otp 4521 nahi aa raha",
                 "maine complaint ki", "kal tak payment kar dunga"],
}


class FakeGenerator:
    model, temperature, max_output_tokens = "fake-model", 1.0, 2000

    def __init__(self, bad_responses=0):
        self.calls, self.bad_left = 0, bad_responses
        self.lock = threading.Lock()

    def complete(self, system, user, seed):
        with self.lock:
            self.calls += 1
            bad = self.bad_left > 0
            self.bad_left -= int(bad)
        lang = "hi" if "Devanagari" in user else "hinglish" if "code-mixed" in user else "en"
        text = "not json" if bad else json.dumps({"utterances": FAKE_UTTERANCES[lang]})
        return Completion(text=text, input_tokens=400, output_tokens=600, model=self.model)


def ledger(tmp_path, cap="1.20"):
    return SpendLedger(tmp_path / "raw" / "spend.jsonl", GPT_41_MINI, Decimal(cap))


# --- catalogue and prompts ---------------------------------------------------

def test_plan_has_384_unique_batches_and_30_30_40_mix():
    specs = plan_batches()
    assert len(specs) == 384 and len({s.batch_id for s in specs}) == 384
    by_lang = Counter()
    for s in specs:
        by_lang[s.language] += s.n
    assert by_lang == {Language.EN: 2304, Language.HI: 2304, Language.HINGLISH: 3072}


def test_pilot_is_five_planned_batches_covering_all_languages():
    pilot = pilot_batches()
    assert [s.batch_id for s in pilot] == list(PILOT_BATCH_IDS)
    assert {s.language for s in pilot} == set(Language)


def test_render_fills_every_field():
    spec = pilot_batches()[2]
    system, user = render(spec)
    assert "complete customer turn" in system
    assert "Devanagari" in user and "where_is_order" in user and "Write 18 different" in user
    assert "{" not in user.replace('{"utterances"', "")


def test_prompt_version_is_stable():
    assert prompt_version() == prompt_version() and len(prompt_version()) == 16


# --- parsing and money -------------------------------------------------------

def test_parse_utterances():
    assert parse_utterances('{"utterances": [" a ", "", "b"]}') == ["a", "b"]
    for bad in ["not json", '{"other": []}', '{"utterances": [1, 2]}', "[]"]:
        with pytest.raises(ValueError):
            parse_utterances(bad)


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
    assert len(s1.called) == 5 and gen.calls == 5
    raw = json.loads((tmp_path / "raw" / f"{PILOT_BATCH_IDS[0]}.json").read_text())
    assert raw["utterances"] == FAKE_UTTERANCES["en"] and raw["prompt_version"] == prompt_version()
    s2 = BatchRunner(gen, l, tmp_path / "raw").run(pilot_batches())
    assert len(s2.cached) == 5 and gen.calls == 5
    assert l.spent == 5 * GPT_41_MINI.cost(400, 600)


def test_runner_stops_at_budget_cap(tmp_path):
    # Reserve worst case $0.0038 per call, settle at actual $0.00112. Sequentially with a $0.005 cap:
    # call 1 (0 + 0.0038 ok), call 2 (0.00112 + 0.0038 ok), call 3 (0.00224 + 0.0038 > cap) -> stop.
    gen = FakeGenerator()
    s = BatchRunner(gen, ledger(tmp_path, cap="0.005"), tmp_path / "raw", workers=1).run(pilot_batches())
    assert len(s.called) == 2 and len(s.skipped_budget) == 3 and gen.calls == 2


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
    assert e["cached"] == 5 and e["calls"] == 379 and e["basis"]["hi"] == "observed over 2 calls"


# --- build -------------------------------------------------------------------

def run_pilot_build(tmp_path, name="pilot"):
    BatchRunner(FakeGenerator(), ledger(tmp_path), tmp_path / "raw").run(pilot_batches())
    return build(tmp_path / "raw", tmp_path, name, BuildConfig())


def test_build_end_to_end(tmp_path):
    m = run_pilot_build(tmp_path)
    assert m["leakage_checks"] == "passed"
    assert m["drops"]["script_mismatch"] == {"hi": 2}                   # Latin line, in both hi batches
    assert "ends_with_hard_negative" not in m["drops"]                 # kept as END, tagged instead
    assert m["counts"]["slices"]["train"]["end_on_hard_negative"] >= 1  # "maine complaint ki"
    assert m["dedup"]["exact_removed"] > 0                             # same fake text in both batches
    assert (tmp_path / "manifests" / "pilot.json").exists()
    rows = [json.loads(l) for l in (tmp_path / "processed" / "pilot" / "train.jsonl").read_text().splitlines()]
    assert all(r["text"] == endpointing.text.normalize(r["text"]) for r in rows)
    assert {r["label"] for r in rows} == {"END", "NOT_END"}


def test_build_is_deterministic(tmp_path):
    a = run_pilot_build(tmp_path, "a")["data_hash"]
    b = build(tmp_path / "raw", tmp_path, "a", BuildConfig())["data_hash"]
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
    assert "384 calls" in out and "of cap $1.20" in out


@pytest.mark.parametrize("batch_id, script, must_have, must_not_have", [
    ("hi.collections.promise_to_pay.fillers", "Devanagari script", "मंगलवार, नहीं, बुधवार", "Tuesday"),
    ("hinglish.telecom.recharge_not_credited.fillers", "Latin script", "Tuesday, nahi, Wednesday", "मतलब"),
    ("en.banking.lost_card.fillers", "Latin script", "I mean", "matlab"),
])
def test_filler_examples_and_script_rule_are_per_language(batch_id, script, must_have, must_not_have):
    spec = next(s for s in plan_batches() if s.batch_id == batch_id)
    _, user = render(spec)
    assert user.rstrip().endswith(f"Every turn must be in {script}.")
    assert must_have in user and must_not_have not in user


def test_every_planned_prompt_renders_without_placeholders():
    for spec in plan_batches():
        _, user = render(spec)
        assert "{" not in user.replace('{"utterances"', "")


def test_build_refuses_mixed_prompt_versions(tmp_path):
    BatchRunner(FakeGenerator(), ledger(tmp_path), tmp_path / "raw").run(pilot_batches())
    path = tmp_path / "raw" / f"{PILOT_BATCH_IDS[0]}.json"
    raw = json.loads(path.read_text())
    path.write_text(json.dumps({**raw, "prompt_version": "old"}))
    with pytest.raises(ValueError, match="mixes prompt versions"):
        build(tmp_path / "raw", tmp_path, "x", BuildConfig())
