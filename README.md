# smart-endpointing

A turn-detection model that lets a voice agent reply sooner without cutting people off.
It is tested against a silence-threshold baseline on a hand-labelled Hindi, Hinglish and English set.
See `SPEC.md` for the full plan and stages.

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
