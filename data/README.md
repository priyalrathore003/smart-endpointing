# Datasheet: synthetic endpointing data (`synth_v1`)

Exact counts, file hashes, drop counts and label-conflict rates are in
`data/manifests/<name>.json`. Results cite its `data_hash`.

## Source
- **Generator:** OpenAI `gpt-4.1-mini-2025-04-14`, temperature 1.0, fixed seed per call.
  Prompts are in `src/endpointing/data/prompts.py`, and their hash is in the manifest.
- **Content:** what customers say to a support voice agent in India. Four domains (banking,
  collections, delivery, telecom), with 8 intents each and 4 styles (plain, fillers and
  self-corrections, numbers, names/places).
- **Licence:** generated text, no third-party corpus. Use is subject to OpenAI's terms for API output.
- **Raw responses** are stored under `data/raw/`, one file per call, together with the token
  usage and cost of every call (`data/raw/spend.jsonl`). Rebuilding never calls the API.

## Size and language mix
- **Planned:** 384 calls, one per domain × intent × style × language.
- **Utterances per call:** 18 for English, 18 for Hindi and 24 for Hinglish, so about
  7,680 raw utterances split **EN 30% / HI 30% / Hinglish 40%**.
- **Scripts:** Hindi is written in Devanagari; Hinglish is code-mixed Hindi-English in Latin script.

## Processing and labels
1. **Normalise** with `endpointing.text.normalize`, the same function used at inference.
   It produces ASR-style text: lowercase, no punctuation, "dont" for "don't", ASCII digits.
2. **Drop** an utterance if it is empty, if its script doesn't match its language, or if it ends on a
   hard-negative word (an ambiguous label).
3. **Deduplicate:** exact duplicates are removed everywhere; near-duplicates (word 3-gram Jaccard ≥ 0.8)
   are removed within each domain/intent/language group.
4. **Spell digits:** about 25% of digit runs are rewritten digit by digit ("four two" / "चार दो" /
   "char do"), seeded. These rows are tagged `digits_spelled`.
5. **Split** by generation batch, 80/10/10 per language. A batch's utterances and all their prefixes
   are in exactly one split.
6. **Truncate:** each complete utterance becomes one **END** row. Prefixes cut at word boundaries are
   **NOT_END**: every prefix ending in a hard negative, plus up to 3 random others. Mid-number prefixes
   are NOT_END and tagged `digits_mid`.

**Row tags:** `hard_negative`, `heuristic_complete`, `has_digits`, `digits_mid`, `digits_spelled`
and `short` (utterance of 4 words or fewer).

**Pause** is not part of this data or the text model. It is handled in the policy layer.

## Splits
- **train:** for training.
- **dev:** for model selection.
- **test:** a secondary synthetic check, including the hard-negative slice.
- **Primary test:** the hand-labelled gold set (Stage 2), which is not generated.

## Known biases and limits
- **Written by an LLM, not spoken:** the text is more fluent and regular than real speech. It has no
  ASR errors, and fillers and self-corrections are stylised.
- **Label noise from truncation:** a prefix that is itself a complete sentence ("i want to pay")
  is labelled NOT_END. `heuristic_complete` counts these, and M4's soft labels are meant to address them.
- **Label conflicts:** the same text can be END in one utterance and NOT_END in another
  (for example, "haan"). These rows are kept, and the conflict rate is in the manifest.
- **Ambiguous hard negatives:** "ki" and "toh" are on the Hinglish list, but "maine complaint ki"
  is a complete sentence. Utterances like that are dropped (counted in the manifest).
- **Digit spelling:** digits are spelled one by one, which is unnatural for amounts ("five zero zero zero").
- **Dedup gaps:** near-duplicates in different domain/intent groups are not removed.
- **Not covered:** Hindi in Latin script that isn't code-mixed, other Indian languages, and real
  telephony ASR output.
