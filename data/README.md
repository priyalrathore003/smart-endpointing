# Datasheet: synthetic endpointing data (`synth_v1`)

Exact counts, file hashes, drop counts and label-conflict rates are in
`data/manifests/<name>.json`. Results cite its `data_hash`.

## Source
- **Generator:** OpenAI `gpt-4.1-mini`, pinned to the snapshot **`gpt-4.1-mini-2025-04-14`**,
  temperature 1.0, fixed seed per call. OpenAI marks `gpt-4.1-mini` as **deprecated** (checked
  2026-10-16). The snapshot was still served then, at $0.40 / $1.60 per 1M input / output tokens.
  If it is retired, the raw responses in `data/raw/` are still the dataset; regenerating would need
  a new model and a new data version.
- **Prompts:** in `src/endpointing/data/prompts.py`. The manifest records their hash
  (`raw.prompt_versions`), and a build refuses raw data that mixes versions. Filler and
  self-correction examples are given per language, in that language's script.
- **Prompt history:** in pilot v0, the Hindi prompt used Latin-script filler examples, and the model
  answered in Hinglish (18 of 18 in one batch). Those responses are kept in `data/raw_pilot_v0/`
  and are not part of any dataset.
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
   hard-negative word (an ambiguous label). Script rules: English and Hinglish must contain no
   Devanagari. Hindi may contain Latin-script words for up to 25% of its tokens
   ("मेरा parcel कहां तक पहुंचा है"); above that, including wholly Latin text, it is dropped.
3. **Deduplicate:** exact duplicates are removed everywhere; near-duplicates (word 3-gram Jaccard ≥ 0.8)
   are removed within each domain/intent/language group.
4. **Spell digits:** about 25% of digit runs are rewritten digit by digit ("four two" / "चार दो" /
   "char do"), seeded. These rows are tagged `digits_spelled`.
5. **Split** by generation batch, 80/10/10 per language. A batch's utterances and all their prefixes
   are in exactly one split.
6. **Truncate:** each complete utterance becomes one **END** row. Prefixes cut at word boundaries are
   **NOT_END**: every prefix ending in a hard negative, plus up to 3 random others. Mid-number prefixes
   are NOT_END and tagged `digits_mid`.

**Row tags:** `hard_negative`, `heuristic_complete`, `has_digits`, `digits_mid`, `digits_spelled`,
`short` (utterance of 4 words or fewer) and `code_mixed` (a Hindi utterance that kept at least one
Latin-script word; the tag is on all of its rows). Each tag is an evaluation slice.

`heuristic_complete` is a rule, not a label:
- **Hindi and Hinglish:** the prefix ends on a verb-final form (है, हुआ, होगा, मिलेगा, दें… / hai, hua, hoga, milega, dein…).
- **English:** the prefix ends on a listed final word, or it has at least 3 words and ends on a content
  word (not a function word, filler or number).
- **Never:** a prefix that ends on a hard negative.

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
