"""Raw LLM responses -> labelled, deduplicated, split dataset + manifest. No API calls.

Besides the train/dev/test files, the build writes analysis/all_prefixes.jsonl (every
word boundary of every utterance, flagged `sampled`). It is for analysis only: never train
on it. It is listed in the manifest but not part of data_hash.

Order matters: normalise -> drop (empty, wrong script in text or question) -> dedup -> spell digits -> split -> truncate.
Dedup runs before digit spelling so a random spelling can't hide a duplicate.
"""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ..text import normalize
from .catalogue import length_bucket
from .conflicts import label_conflicts
from .dedup import Deduplicator
from .digits import DigitSpeller
from .leakage import check_leakage
from .manifest import build_manifest, example_record, sha256_file, write_jsonl, write_manifest
from .script import script_ok
from .split import BatchSplitter
from .truncate import AllHardPlusRandom, TruncationConfig, Truncator
from .types import Example, Label, Source, Split, Style, Utterance, dedup_key


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 0
    spell_prob: float = 0.25
    dedup_threshold: float = 0.8
    max_prefixes: int = 3
    tail_share: float = 0.3


def utterance_id(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def sampling_stats(examples: list[Example]) -> dict:
    """END:NOT_END ratio and the share of NOT_END prefixes that are only 1-2 words long."""
    end = sum(ex.label is Label.END for ex in examples)
    not_end = [ex for ex in examples if ex.label is Label.NOT_END]
    stubs = sum(ex.k <= 2 for ex in not_end)
    return {"END": end, "NOT_END": len(not_end),
            "not_end_per_end": round(len(not_end) / end, 3) if end else None,
            "share_1_2_word_prefixes": round(stubs / len(not_end), 4) if not_end else None}


def _length_report(cleaned: list[tuple[Utterance, str | None]]) -> dict:
    """Actual length buckets of complete turns (before digit spelling), and how often the
    model's turn landed in the bucket it was asked for."""
    counts: dict[str, dict[str, Counter]] = {"turns": defaultdict(Counter), "short_replies": defaultdict(Counter)}
    agree = asked = 0
    for u, requested in cleaned:
        actual = length_bucket(len(u.text.split()))
        group = "short_replies" if u.source.style is Style.SHORT_REPLIES else "turns"
        counts[group][u.source.language.value][actual] += 1
        if requested is not None:
            asked += 1
            agree += requested == actual
    out = {g: {lang: dict(sorted(c.items())) for lang, c in sorted(v.items())} for g, v in counts.items()}
    out["requested_bucket_agreement"] = {"agree": agree, "asked": asked,
                                         "rate": round(agree / asked, 4) if asked else None}
    return out


def load_raw(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(raw_dir).glob("*.json"))]


def build(raw_dir: Path, data_root: Path, name: str, config: BuildConfig = BuildConfig(),
          generation_meta: dict | None = None) -> dict:
    raws = load_raw(raw_dir)
    versions = sorted({r["prompt_version"] for r in raws})
    if len(versions) > 1:
        raise ValueError(f"raw data mixes prompt versions {versions}; regenerate or move the old batches")
    raw_by_lang: Counter = Counter()
    drops: dict[str, Counter] = defaultdict(Counter)
    cleaned: list[tuple[Utterance, str | None]] = []
    for raw in raws:
        src = Source(raw["batch_id"], raw["language"], raw["domain"], raw["intent"], raw["style"])
        for item in raw["items"]:
            raw_by_lang[src.language.value] += 1
            t = normalize(item["text"])
            c = normalize(item["context"]) if item["context"] is not None else None
            reason = ("empty" if not t or c == "" else
                      "script_mismatch" if not script_ok(t, src.language)
                      or (c is not None and not script_ok(c, src.language)) else None)
            if reason:
                drops[reason][src.language.value] += 1
                continue
            cleaned.append((Utterance(utterance_id(dedup_key(t, c)), src, t, c), item["requested_bucket"]))

    kept, dedup_report = Deduplicator(config.dedup_threshold).dedupe([u for u, _ in cleaned])
    kept_ids = {u.utterance_id for u in kept}
    speller = DigitSpeller(config.spell_prob, config.seed)
    utts = [speller.apply(u) for u in kept]

    split_of = BatchSplitter().assign(u.source for u in utts)
    tconfig = TruncationConfig(max_prefixes=config.max_prefixes, tail_share=config.tail_share)
    truncator = Truncator(tconfig, config.seed)
    rows = [(split_of[u.source.batch_id], ex) for u in utts for ex in truncator.examples(u)]
    old_scheme = Truncator(tconfig, config.seed, sampler=AllHardPlusRandom(), reply_sampler=AllHardPlusRandom())
    every = [(split_of[u.source.batch_id], ex) for u in utts for ex in truncator.all_prefixes(u)]
    sampled_keys = {(ex.utterance_id, ex.k) for _, ex in rows}

    leakage = check_leakage(rows, config.dedup_threshold)
    if not leakage.ok:
        raise LeakageError(f"leakage found: {leakage}")

    out_dir = Path(data_root) / "processed" / name
    files = []
    for split in Split:
        path = out_dir / f"{split.value}.jsonl"
        write_jsonl(path, (example_record(s, ex) for s, ex in rows if s is split))
        files.append(path)

    analysis_path = out_dir / "analysis" / "all_prefixes.jsonl"
    n_every = write_jsonl(analysis_path, ({**example_record(s, ex), "sampled": (ex.utterance_id, ex.k) in sampled_keys}
                                          for s, ex in every))

    def by_group(examples: list[Example]) -> dict:
        is_reply = lambda ex: ex.source.style is Style.SHORT_REPLIES
        return {"all": sampling_stats(examples),
                "turns": sampling_stats([ex for ex in examples if not is_reply(ex)]),
                "short_replies": sampling_stats([ex for ex in examples if is_reply(ex)])}

    manifest = build_manifest(name, Path(data_root), files, rows, {
        "analysis_files": [{"path": str(analysis_path.relative_to(data_root)), "sha256": sha256_file(analysis_path),
                            "rows": n_every, "note": "every word boundary; analysis only, never train on it"}],
        "prefix_sampling": {
            "scheme": f"capped: <= {config.max_prefixes} NOT_END per utterance (hard-negative/filler, "
                      f"last {int(config.tail_share * 100)}% of words, random); short replies: hard-negative only",
            "after": by_group([ex for _, ex in rows]),
            "before_all_hard_plus_3_random": by_group([ex for u in utts for ex in old_scheme.examples(u)]),
            "every_boundary": by_group([ex for _, ex in every]),
        },
        "build_config": asdict(config),
        "generation": generation_meta or {},
        "raw": {"batches": len(raws), "prompt_versions": versions,
                "utterances_by_language": dict(sorted(raw_by_lang.items()))},
        "drops": {k: dict(sorted(v.items())) for k, v in sorted(drops.items())},
        "dedup": {"input": dedup_report.input, "exact_removed": dedup_report.exact_removed,
                  "near_removed": dedup_report.near_removed, "kept": dedup_report.kept,
                  "near_examples": [list(p) for p in dedup_report.near_examples]},
        "utterances_by_language": dict(sorted(Counter(u.source.language.value for u in utts).items())),
        "utterances_by_style": dict(sorted(Counter(u.source.style.value for u in utts).items())),
        "length_buckets": _length_report([(u, r) for u, r in cleaned if u.utterance_id in kept_ids]),
        "batches_by_split": dict(sorted(Counter(s.value for s in split_of.values()).items())),
        "label_conflicts": label_conflicts(ex for _, ex in rows),
        "leakage_checks": "passed",
    })
    write_manifest(Path(data_root) / "manifests" / f"{name}.json", manifest)
    return manifest
