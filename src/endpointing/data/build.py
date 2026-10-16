"""Raw LLM responses -> labelled, deduplicated, split dataset + manifest. No API calls.

Order matters: normalise -> drop (empty, wrong script) -> dedup -> spell digits -> split -> truncate.
Dedup runs before digit spelling so a random spelling can't hide a duplicate.
"""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ..text import normalize
from .conflicts import label_conflicts
from .dedup import Deduplicator
from .digits import DigitSpeller
from .leakage import check_leakage
from .manifest import build_manifest, example_record, write_jsonl, write_manifest
from .script import script_ok
from .split import BatchSplitter
from .truncate import TruncationConfig, Truncator
from .types import Source, Split, Utterance


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 0
    spell_prob: float = 0.25
    dedup_threshold: float = 0.8
    max_random_prefixes: int = 3


def utterance_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


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
    cleaned: list[Utterance] = []
    for raw in raws:
        src = Source(raw["batch_id"], raw["language"], raw["domain"], raw["intent"], raw["style"])
        for text in raw["utterances"]:
            raw_by_lang[src.language.value] += 1
            t = normalize(text)
            reason = "empty" if not t else "script_mismatch" if not script_ok(t, src.language) else None
            if reason:
                drops[reason][src.language.value] += 1
                continue
            cleaned.append(Utterance(utterance_id(t), src, t))

    kept, dedup_report = Deduplicator(config.dedup_threshold).dedupe(cleaned)
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
        "batches_by_split": dict(sorted(Counter(s.value for s in split_of.values()).items())),
        "label_conflicts": label_conflicts(ex for _, ex in rows),
        "leakage_checks": "passed",
    })
    write_manifest(Path(data_root) / "manifests" / f"{name}.json", manifest)
    return manifest
