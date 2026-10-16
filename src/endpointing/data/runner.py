"""Run generation calls: cache raw responses, enforce the budget, retry a bad response once.

A batch whose raw file exists is never called again, so the pilot's batches are
reused by the full run and rebuilding the dataset is free.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from .budget import SpendLedger
from .catalogue import BatchSpec
from .generator import UtteranceGenerator, parse_response
from .prompts import prompt_version, render, shape_of

WORST_CASE_INPUT_TOKENS = 1500   # rendered prompts are ~400 tokens; this is a safe ceiling


class StaleCacheError(RuntimeError):
    pass


@dataclass
class RunSummary:
    called: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped_budget: list[str] = field(default_factory=list)
    errored: list[str] = field(default_factory=list)   # the API call itself failed (timeout, network)


def call_seed(batch_id: str, attempt: int) -> int:
    return int(hashlib.sha1(f"{batch_id}:{attempt}".encode()).hexdigest()[:8], 16)


class BatchRunner:
    def __init__(self, generator: UtteranceGenerator, ledger: SpendLedger, raw_dir: Path,
                 max_attempts: int = 2, workers: int = 4):
        if max_attempts < 1 or workers < 1:
            raise ValueError("max_attempts and workers must be >= 1")
        self.generator, self.ledger, self.raw_dir = generator, ledger, Path(raw_dir)
        self.max_attempts, self.workers = max_attempts, workers

    def raw_path(self, spec: BatchSpec) -> Path:
        return self.raw_dir / f"{spec.batch_id}.json"

    def run(self, specs: Sequence[BatchSpec]) -> RunSummary:
        summary = RunSummary()
        todo, stale = [], []
        for spec in specs:
            path = self.raw_path(spec)
            if not path.exists():
                todo.append(spec)
                continue
            cached_version = json.loads(path.read_text(encoding="utf-8"))["prompt_version"]
            (summary.cached if cached_version == self._version(spec) else stale).append(spec)
        if stale:   # never mix prompt versions silently; checked before any call is made
            raise StaleCacheError(f"{len(stale)} cached batches were made with an older prompt, e.g. "
                                  f"{stale[0].batch_id}; move them aside before generating")
        summary.cached = [s.batch_id for s in summary.cached]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for batch_id, outcome in pool.map(self._run_one, todo):
                getattr(summary, outcome).append(batch_id)
        return summary

    @staticmethod
    def _version(spec: BatchSpec) -> str:
        shape = shape_of(spec.style)
        return prompt_version(shape, spec.language if shape == "turns" else None)

    def _run_one(self, spec: BatchSpec) -> tuple[str, str]:
        prompt = render(spec)
        worst = self.ledger.pricing.cost(WORST_CASE_INPUT_TOKENS, self.generator.max_output_tokens)
        for attempt in range(self.max_attempts):
            if not self.ledger.try_reserve(worst):
                return spec.batch_id, "skipped_budget"
            seed = call_seed(spec.batch_id, attempt)
            try:
                completion = self.generator.complete(prompt.system, prompt.user, seed, prompt.response_format)
            except Exception as e:   # one bad call must not abort the run; it may still be billed
                self.ledger.settle_unknown(worst, spec.batch_id, self.generator.model, type(e).__name__)
                return spec.batch_id, "errored"
            try:
                items, ok = parse_response(completion.text, spec.style), True
            except ValueError:
                items, ok = [], False
            self.ledger.settle(worst, spec.batch_id, completion.model, completion.input_tokens,
                               completion.output_tokens, ok)
            if ok:
                self._write_raw(spec, prompt, seed, completion, items)
                return spec.batch_id, "called"
        return spec.batch_id, "failed"

    def _write_raw(self, spec, prompt, seed, completion, items) -> None:
        record = {
            "batch_id": spec.batch_id, "language": spec.language.value, "domain": spec.domain.value,
            "intent": spec.intent, "style": spec.style.value, "n_requested": spec.n,
            "bucket_counts_requested": spec.bucket_counts(),
            "prompt_version": self._version(spec), "prompt_shape": shape_of(spec.style),
            "context_count_requested": spec.context_count(), "model": completion.model,
            "request": {"system": prompt.system, "user": prompt.user, "seed": seed,
                        "response_format": prompt.response_format["json_schema"]["name"],
                        "temperature": self.generator.temperature,
                        "max_completion_tokens": self.generator.max_output_tokens},
            "usage": {"input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens},
            "response_text": completion.text, "items": [asdict(i) for i in items],
        }
        path = self.raw_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)                      # atomic: a crash never leaves a half-written cache file
