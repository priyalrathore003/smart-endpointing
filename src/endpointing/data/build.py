"""Raw LLM responses -> labelled, deduplicated, split dataset + manifest. No API calls.

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
from .manifest import build_manifest, example_record, write_jsonl, write_manifest
from .script import script_ok
from .split import BatchSplitter
from .truncate import TruncationConfig, Truncator
from .types import Source, Split, Style, Utterance, dedup_key


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 0
    spell_prob: float = 0.25
    dedup_threshold: float = 0.8
    max_random_prefixes: int = 3


def utterance_id(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:12]


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
    truncator = Truncator(TruncationConfig(max_random_prefixes=config.max_random_prefixes), config.seed)
    rows = [(split_of[u.source.batch_id], ex) for u in utts for ex in truncator.examples(u)]

    leakage = check_leakage(rows, config.dedup_threshold)
    if not leakage.ok:
        raise LeakageError(f"leakage found: {leakage}")

    out_dir = Path(data_root) / "processed" / name
    files = []
    for split in Split:
        path = out_dir / f"{split.value}.jsonl"
        write_jsonl(path, (example_record(s, ex) for s, ex in rows if s is split))
        files.append(path)

    manifest = build_manifest(name, Path(data_root), files, rows, {
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
