# Datasheet: synthetic endpointing data (`synth_v1`)

Exact counts, file hashes, drop counts and label-conflict rates are in
`data/manifests/<name>.json`. Results cite its `data_hash`.

**What is in git:** the raw LLM responses (`data/raw/*.json`, one per batch, 480 for `synth_v1`), the
spend ledger (`data/raw/spend.jsonl`) and the manifest (`data/manifests/synth_v1.json`), about 4.7 MB.
The processed train/dev/test files (16 MB) are not committed. They rebuild exactly with
`python -m endpointing.data.cli build --name synth_v1` (no API calls, a few seconds), and a CI test checks
that the rebuild matches the manifest's `data_hash` (`5dc7cd3a…`).

## Source
- **Generator:** OpenAI `gpt-4.1-mini`, pinned to the snapshot **`gpt-4.1-mini-2025-04-14`**,
  temperature 1.0, fixed seed per call. OpenAI marks `gpt-4.1-mini` as **deprecated** (checked
  2026-10-16). The snapshot was still served then, at $0.40 / $1.60 per 1M input / output tokens.
  If it is retired, the raw responses in `data/raw/` are still the dataset; regenerating would need
  a new model and a new data version.
- **Prompts:** in `src/endpointing/data/prompts.py`. There are two shapes (turns and short-reply pairs),
  each with its own hash. The manifest records the hash per shape (`raw.prompt_versions`), a build refuses
  raw data that mixes versions within a shape, and the generator refuses to reuse a cached batch made
  with an older prompt. Filler and
  self-correction examples are given per language, in that language's script.
- **Prompt history:** in pilot v0, the Hindi prompt used Latin-script filler examples, and the model
  answered in Hinglish (18 of 18 in one batch). In pilot v1, there were no short turns and no code-mixed Hindi.
  In pilot v2, Hindi was formal and sometimes mixed topics, and Hindi short replies missed the 75/25 split.
  The three pilots are kept in `data/raw_pilot_v0/`, `v1/` and `v2/` and are not part of any dataset.
- **Content:** what customers say to a support voice agent in India. Four domains (banking,
  collections, delivery, telecom), with 8 intents each and 5 styles: plain, fillers and
  self-corrections, numbers, names/places, and **short replies**. Short replies are
  (agent_question, reply) pairs; the question is kept in the row's `context` field.
- **Context on turns:** for about 30% of turn-style utterances (exactly round(0.3 n) per call), the
  model also writes the agent question that came just before. Those turns are truncated as usual,
  so their prefixes are NOT_END rows *with* context. Without this, context would appear almost
  only on (mostly END) short replies, and a model could learn "question present means END".
  The manifest's `context_balance` reports P(END | context) and P(END | no context) per language,
  and **the build fails if the gap is over 15 points**.
- **Length buckets:** every turn prompt asks for exact counts: 25% short (1-4 words), 50% medium
  (5-12) and 25% long (13-30). Short replies ask for 75% of 1-4 words and 25% of 5-12, with at least a
  third ambiguous ("haan ji", "theek hai", "kal tak", a bare number). The manifest reports the
  actual buckets (`length_buckets`) and how often the model hit the requested one. In pilot v1, without
  exact counts, no turn was under 5 words.
- **Hindi code-mixing:** the Hindi prompt asks for one common English noun in Latin script in about
  1 in 4 turns (parcel, account, OTP, payment); all other text is Devanagari.
- **Hindi register:** colloquial spoken Hindi, not formal written Hindi (avoid कृपया, श्रीमान, रकम, जमा;
  use सर, मैडम, प्लीज़, जल्दी, पैसा), one topic per utterance. Short-reply pairs come back in `short`
  and `medium` arrays, so the 75/25 split is enforced by the response structure, as for turns.
- **Licence:** generated text, no third-party corpus. Use is subject to OpenAI's terms for API output.
- **Raw responses** are stored under `data/raw/`, one file per call, together with the token
  usage and cost of every call (`data/raw/spend.jsonl`). Rebuilding never calls the API.

## Size and language mix
- **Planned:** 480 calls, one per domain × intent × style × language.
- **Utterances per call:** 18 for English, 18 for Hindi and 24 for Hinglish, so about
  9,600 raw utterances split **EN 30% / HI 30% / Hinglish 40%**.
- **Scripts:** Hindi is written in Devanagari; Hinglish is code-mixed Hindi-English in Latin script.

## Processing and labels
1. **Normalise** with `endpointing.text.normalize`, the same function used at inference.
   It produces ASR-style text: lowercase, no punctuation, "dont" for "don't", ASCII digits.
2. **Drop** an utterance if it (or its agent question) is empty or in the wrong script. Script rules: English and Hinglish must contain no
   Devanagari. Hindi may contain Latin-script words for up to 25% of its tokens
   ("मेरा parcel कहां तक पहुंचा है"); above that, including wholly Latin text, it is dropped.
3. **Deduplicate:** turns are compared by text plus agent question, so "haan ji" answering two different
   questions is kept twice. Exact duplicates are removed everywhere; near-duplicates (word 3-gram
   Jaccard ≥ 0.8) are removed within each domain/intent/language group.
4. **Spell digits:** about 25% of identifier-like digit runs are rewritten digit by digit ("four two" /
   "चार दो" / "char do"), seeded. These rows are tagged `digits_spelled`. A run qualifies only if it has
   4+ digits (phone, account, OTP, order ID) and is not a date (a month or date word next to it, or a
   dd mm [yy]yy shape) or an amount (a currency or amount word next to it). "30 अप्रैल" and "5000 rupees"
   are never spelled out.
5. **Split** by generation batch, 80/10/10 per language. A batch's utterances and all their prefixes
   are in exactly one split.
6. **Truncate:** each complete utterance becomes one **END** row. Prefixes cut at word boundaries are
   **NOT_END**, with **at most 3 per utterance**, using the same scheme for train, dev and test: one prefix ending
   on a hard negative or filler (if there is one), one in the last 30% of the words, and one random (seeded, no
   duplicates). **Short replies** keep only prefixes that end on a hard negative; random cuts of an answer
   ("kal", "nahi") are usually complete given the question. The prefixes they keep are tagged `prefix_of_reply`.
   Mid-number prefixes are NOT_END and tagged `digits_mid`.
   Every boundary of every utterance is also written to `processed/<name>/analysis/all_prefixes.jsonl`
   (flagged `sampled`). That file is **analysis only, never used for training**, and is outside `data_hash`.
   The manifest's `prefix_sampling` compares the END:NOT_END ratio and the share of 1–2 word prefixes for
   the capped scheme, the previous one (all hard negatives plus 3 random) and every boundary. A complete turn that ends on a hard-negative word ("…and i did
   not do that") stays END and is tagged `end_on_hard_negative`.

**Row tags:** `hard_negative`, `heuristic_complete`, `has_digits`, `digits_mid`, `digits_spelled`,
`short` (utterance of 4 words or fewer), `code_mixed` (a Hindi utterance that kept at least one
Latin-script word; the tag is on all of its rows), `end_on_hard_negative` (END rows only) and
`prefix_of_reply` (NOT_END rows from short replies). Each tag is an evaluation slice, and so is
`style == short_replies`.

`heuristic_complete` is a rule, not a label:
- **Hindi and Hinglish:** the prefix ends on a verb-final form (है, हुआ, होगा, मिलेगा, दें, दूंगा, सकते हो… /
  hai, hua, hoga, milega, dein, dunga, karunga, sakte ho…), on a negation (पता नहीं, pata nahi), or on an
  imperative (बताओ, बताइए, करो, दो, करें, करिए / batao, karo, do, kariye, dijiye). A prefix whose last word is
  inside a number is never tagged, because "दो"/"do" is also the digit two.
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
- **The class ratio is set by the sampling, not by real calls.** END:NOT_END here comes from the
  prefix cap. The real ratio of finished to unfinished turns *at pauses* (what the policy faces) must
  come from recordings: the gold set and shadow logs. Calibration and thresholds must be checked
  against that ratio, not this one.
- **Written by an LLM, not spoken:** the text is more fluent and regular than real speech. It has no
  ASR errors, and fillers and self-corrections are stylised.
- **Label noise from truncation:** a prefix that is itself a complete sentence ("i want to pay")
  is labelled NOT_END. `heuristic_complete` counts these, and M4's soft labels are meant to address them.
- **Label conflicts:** the same text can be END in one utterance and NOT_END in another
  (for example, "haan"). These rows are kept, and the conflict rate is in the manifest.
- **Ambiguous hard negatives:** "ki" and "toh" are on the Hinglish list, but "maine complaint ki"
  is a complete sentence. Such turns stay END (`end_on_hard_negative`). Their prefixes that end on the same
  word elsewhere are NOT_END, which is exactly the ambiguity the label-conflict rate measures.
- **Label conflicts on text alone:** the conflict report ignores `context`. A reply like "haan ji" can be END
  after a question and NOT_END as the start of a longer turn; that is real ambiguity, which the context
  ablation (SPEC, Stages 3 and 5) measures.
- **Digit spelling:** digits are spelled one by one, which is unnatural for amounts ("five zero zero zero").
- **Dedup gaps:** near-duplicates in different domain/intent groups are not removed.
- **Not covered:** Hindi in Latin script that isn't code-mixed, other Indian languages, and real
  telephony ASR output.
