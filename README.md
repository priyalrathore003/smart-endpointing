# smart-endpointing

A turn-detection model that lets a voice agent reply sooner without cutting people off.
The plan is to test it against a silence-threshold baseline on a hand-labelled Hindi, Hinglish and English set.
See `SPEC.md` for the full plan and stages.

## Status

**There is no model yet, so there are no model results.** Only the synthetic training data exists.

| Stage | State |
|---|---|
| 0: repo, CI, rules | Done |
| 1: synthetic data | **Done** (`synth_v1`; details below and in `data/README.md`) |
| 2: gold set and labelling CLI | Not started |
| 3: M0/M1 and the evaluation library | Not started |
| 4: M2, a transformer trained from scratch | Not started |
| 5: M3 fine-tune and M4 distillation | Not started |
| 6: ONNX, int8 and Lambda | Not started |
| 7: shadow mode in Auvia, report | Not started |

**Stage 1 (`synth_v1`, `data_hash 5dc7cd3a…`)**
- **Source:** generated with `gpt-4.1-mini-2025-04-14`, 480 calls. Banking, collections, delivery and telecom; 8 intents
  per domain; 5 styles, one of which is (agent question, short reply) pairs.
- **Utterances:** 9,899 complete ones after dropping wrong-script text and duplicates: English 3,166, Hindi
  (Devanagari) 2,684 and Hinglish 4,049. Hindi is a little under its 30% target because of the script drops.
- **Rows:** 28,529 labelled rows, 9,899 END and 18,630 NOT_END (1 : 1.88). That ratio comes from the prefix
  cap, not from real calls. The split is by generation batch: 22,907 train, 2,741 dev and 2,881 test rows.
- **Evaluation slices (all splits):** hard negatives 4,423; short utterances 4,749; code-mixed Hindi 1,837;
  rows with digits 2,496; digits spelled out 244; short-reply prefixes 260; END rows ending on a hard
  negative 66.
- **Label conflicts:** 0.54% of distinct texts appear as both END and NOT_END (135 of 25,114).
- **Context balance:** P(END) with vs without an agent question differs by 9.9 / 10.4 / 10.6 points
  (en / hi / hinglish). The build fails above 15.
- **Checks:** leakage checks passed. The data rebuilds exactly from the committed raw responses, and a CI
  test verifies this.
- **Cost:** $0.55 of LLM spend for the whole stage (519 calls, including five pilot rounds).
- **Known limits:** the text is LLM-written rather than real speech, and the labels come from where
  prefixes are cut. See `data/README.md` for the full list.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## Layout

| Path | Contents | Stage |
|---|---|---|
| `src/endpointing/data/` | Generation, truncation, hard negatives, dedup, split, manifest | 1 |
| `src/endpointing/models/` | M0 silence threshold to M4 distilled student | 3-5 |
| `src/endpointing/evals/` | Metrics, calibration, slices, bootstrap, simulated conversations | 3 |
| `src/endpointing/serve/` | FastAPI and Lambda prediction API | 6 |
| `tests/` | pytest suite, run in CI on every push | all |
