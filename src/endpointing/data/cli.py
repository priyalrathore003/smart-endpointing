"""Stage 1 command line.

    python -m endpointing.data.cli plan                       # calls and cost estimate; no API
    python -m endpointing.data.cli generate --pilot --yes     # 5 batches
    python -m endpointing.data.cli generate --all --yes       # remaining batches (cached ones skipped)
    python -m endpointing.data.cli build --name pilot
    python -m endpointing.data.cli sample --name pilot --per-language 20

Every API call is budgeted against --cap (default $1.20, across pilot and full run).
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from .budget import GPT_41_MINI, SpendLedger
from .build import BuildConfig, build
from .catalogue import BatchSpec, pilot_batches, plan_batches
from .generator import MODEL, OpenAIGenerator
from .prompts import prompt_version
from .runner import BatchRunner
from .types import Language

DEFAULT_CAP = Decimal("1.20")
# Before any spend is recorded: ~450 input tokens per call, ~60 output tokens per utterance
# (Devanagari Hindi is the most expensive). Replaced by observed numbers once calls exist.
PRIOR_INPUT_PER_CALL = 450
PRIOR_OUTPUT_PER_UTTERANCE = 60


def _paths(data_root: Path) -> tuple[Path, Path]:
    return data_root / "raw", data_root / "raw" / "spend.jsonl"


def estimate(specs: list[BatchSpec], raw_dir: Path) -> dict:
    """Projected cost of the uncached batches, from observed token usage where available."""
    observed: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for p in raw_dir.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        observed[r["language"]].append((r["usage"]["input_tokens"], r["usage"]["output_tokens"], r["n_requested"]))
    todo = [s for s in specs if not (raw_dir / f"{s.batch_id}.json").exists()]
    total, basis = Decimal(0), {}
    for lang in Language:
        lang_specs = [s for s in todo if s.language is lang]
        obs = observed.get(lang.value)
        if obs:
            in_per_call = sum(o[0] for o in obs) / len(obs)
            out_per_utt = sum(o[1] for o in obs) / sum(o[2] for o in obs)
            basis[lang.value] = f"observed over {len(obs)} calls"
        else:
            in_per_call, out_per_utt = PRIOR_INPUT_PER_CALL, PRIOR_OUTPUT_PER_UTTERANCE
            basis[lang.value] = "prior"
        for s in lang_specs:
            total += GPT_41_MINI.cost(round(in_per_call), round(out_per_utt * s.n))
    return {"calls": len(todo), "cached": len(specs) - len(todo),
            "utterances": sum(s.n for s in todo), "projected_usd": total.quantize(Decimal("0.0001")),
            "basis": basis}


def cmd_plan(args) -> int:
    raw_dir, ledger_path = _paths(args.data_dir)
    ledger = SpendLedger(ledger_path, GPT_41_MINI, args.cap)
    for label, specs in [("pilot", pilot_batches()), ("all", plan_batches())]:
        e = estimate(specs, raw_dir)
        print(f"{label:5}: {e['calls']} calls to make ({e['cached']} cached), {e['utterances']} utterances, "
              f"projected ${e['projected_usd']}  [basis: {e['basis']}]")
    print(f"spent so far: ${ledger.spent.quantize(Decimal('0.0001'))} of cap ${args.cap}")
    return 0


def cmd_generate(args) -> int:
    specs = pilot_batches() if args.pilot else plan_batches()
    raw_dir, ledger_path = _paths(args.data_dir)
    ledger = SpendLedger(ledger_path, GPT_41_MINI, args.cap)
    e = estimate(specs, raw_dir)
    print(f"{e['calls']} calls, projected ${e['projected_usd']}; spent ${ledger.spent:.4f} of cap ${args.cap}")
    if not args.yes:
        print("Not calling the API without --yes.")
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        return 2
    print("OPENAI_API_KEY: present")
    runner = BatchRunner(OpenAIGenerator(), ledger, raw_dir, workers=args.workers)
    s = runner.run(specs)
    print(f"called {len(s.called)}, cached {len(s.cached)}, failed {s.failed}, skipped for budget {s.skipped_budget}")
    print(f"spent so far: ${ledger.spent:.6f} of cap ${args.cap}")
    return 0 if not (s.failed or s.skipped_budget) else 1


def cmd_build(args) -> int:
    raw_dir, _ = _paths(args.data_dir)
    current = {f"turns/{lang.value}": prompt_version("turns", lang) for lang in Language}
    current["pairs"] = prompt_version("pairs")
    meta = {"model": MODEL, "prompt_versions_current": current,
            "pricing_usd_per_million": {"input": str(GPT_41_MINI.input_per_million),
                                        "output": str(GPT_41_MINI.output_per_million)}}
    m = build(raw_dir, args.data_dir, args.name, BuildConfig(), meta)
    print(json.dumps({k: m[k] for k in ("data_hash", "raw", "drops", "dedup", "utterances_by_language",
                                        "utterances_by_style", "length_buckets", "context_balance",
                                        "batches_by_split",
                                        "label_conflicts", "counts")},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_sample(args) -> int:
    """Random rows per group: each language (turn styles), then short replies with their question."""
    rows = []
    for p in sorted((args.data_dir / "processed" / args.name).glob("*.jsonl")):
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    rng = random.Random(args.seed)
    groups = [(lang.value, [r for r in rows if r["language"] == lang.value and r["style"] != "short_replies"])
              for lang in Language]
    groups.append(("short_replies", [r for r in rows if r["style"] == "short_replies"]))
    for name, pool in groups:
        if args.group and name not in args.group:
            continue
        picked = sorted(rng.sample(pool, min(args.per_group, len(pool))), key=lambda r: (r["language"], r["label"]))
        print(f"\n=== {name}: {len(picked)} of {len(pool)} rows ===")
        print(f"{'label':8} {'h_comp':6} {'spelled':7} {'c_mixed':7} {'end_hn':6}  text")
        for r in picked:
            ctx = f"   [Q: {r['context']}]" if r["context"] else ""
            print(f"{r['label']:8} {str(r['heuristic_complete']):6} {str(r['digits_spelled']):7} "
                  f"{str(r['code_mixed']):7} {str(r['end_on_hard_negative']):6}  {r['text']}{ctx}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="endpointing.data.cli")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--cap", type=Decimal, default=DEFAULT_CAP, help="hard spend cap in USD")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    g = sub.add_parser("generate")
    which = g.add_mutually_exclusive_group(required=True)
    which.add_argument("--pilot", action="store_true")
    which.add_argument("--all", action="store_true")
    g.add_argument("--yes", action="store_true", help="actually call the API")
    g.add_argument("--workers", type=int, default=4)
    g.set_defaults(fn=cmd_generate)
    b = sub.add_parser("build")
    b.add_argument("--name", default="synth_v1")
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("sample")
    s.add_argument("--name", default="synth_v1")
    s.add_argument("--per-group", type=int, default=20)
    s.add_argument("--group", action="append", help="en, hi, hinglish or short_replies; repeatable")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
