import json

from conftest import make_utt
from endpointing.data.manifest import build_manifest, example_record, write_jsonl, write_manifest
from endpointing.data.truncate import Truncator
from endpointing.data.types import Split

UTT = make_utt("my number is 98765 and the otp is 1234")


def build(tmp_path, utt=UTT):
    rows = [(Split.TRAIN, ex) for ex in Truncator().examples(utt)]
    f = tmp_path / "processed" / "train.jsonl"
    write_jsonl(f, (example_record(s, ex) for s, ex in rows))
    return build_manifest("t", tmp_path, [f], rows, {"seed": 0}), rows


def test_manifest_hash_is_reproducible(tmp_path):
    m1, _ = build(tmp_path)
    m2, _ = build(tmp_path)
    assert m1["data_hash"] == m2["data_hash"] and len(m1["data_hash"]) == 64


def test_manifest_hash_changes_with_data(tmp_path):
    m1, _ = build(tmp_path)
    m2, _ = build(tmp_path, make_utt("a different utterance entirely"))
    assert m1["data_hash"] != m2["data_hash"]


def test_manifest_counts_and_paths(tmp_path):
    m, rows = build(tmp_path)
    assert m["files"][0]["path"] == "processed/train.jsonl"
    assert m["counts"]["rows"] == len(rows)
    assert m["counts"]["by_split_label"]["train"]["END"] == 1
    assert m["seed"] == 0


def test_records_round_trip_unicode(tmp_path):
    utt = make_utt("मेरा पैसा कट गया", language="hi")
    rec = example_record(Split.DEV, Truncator().examples(utt)[-1])
    write_jsonl(tmp_path / "x.jsonl", [rec])
    assert json.loads((tmp_path / "x.jsonl").read_text(encoding="utf-8")) == rec
    write_manifest(tmp_path / "m.json", {"text": rec["text"]})
    assert "मेरा" in (tmp_path / "m.json").read_text(encoding="utf-8")
