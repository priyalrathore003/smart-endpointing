"""Write dataset files and a manifest that names the exact data version.

`data_hash` is a sha256 over the sorted (path, file sha256) pairs, so results
can cite one string. Files are written with sorted keys and a fixed row order,
so rebuilding from the same raw data gives the same hash.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from .types import Example, Split

SLICES = ("hard_negative", "heuristic_complete", "has_digits", "digits_mid", "digits_spelled", "short")


def example_record(split: Split, ex: Example) -> dict:
    return {"split": split.value, "utterance_id": ex.utterance_id, "batch_id": ex.source.batch_id,
            "language": ex.source.language.value, "domain": ex.source.domain.value,
            "intent": ex.source.intent, "style": ex.source.style.value, "text": ex.text,
            "label": ex.label.value, "k": ex.k, "n": ex.n, **{s: getattr(ex, s) for s in SLICES}}


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows: Sequence[tuple[Split, Example]]) -> dict:
    by_split_label: dict = defaultdict(Counter)
    by_split_language: dict = defaultdict(Counter)
    slices: dict = defaultdict(Counter)
    for split, ex in rows:
        by_split_label[split.value][ex.label.value] += 1
        by_split_language[split.value][ex.source.language.value] += 1
        for s in SLICES:
            slices[split.value][s] += int(getattr(ex, s))
    as_dict = lambda d: {k: dict(sorted(v.items())) for k, v in sorted(d.items())}
    return {"rows": len(rows), "by_split_label": as_dict(by_split_label),
            "by_split_language": as_dict(by_split_language), "slices": as_dict(slices)}


def build_manifest(name: str, root: Path, files: Sequence[Path],
                   rows: Sequence[tuple[Split, Example]], extra: dict) -> dict:
    file_entries = [{"path": str(f.relative_to(root)), "sha256": sha256_file(f)} for f in sorted(files)]
    data_hash = hashlib.sha256("".join(f"{e['path']}:{e['sha256']}\n" for e in file_entries).encode()).hexdigest()
    return {"name": name, "data_hash": data_hash, "files": file_entries, "counts": counts(rows), **extra}


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
