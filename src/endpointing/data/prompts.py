"""Prompt templates for utterance generation. Their hash is recorded with every raw response.

Two shapes:
- turns: one complete customer turn each, requested per length bucket;
- short replies: (agent_question, reply) pairs; the question becomes the row's `context`.
"""

import hashlib
import json
from dataclasses import dataclass

from .catalogue import (AMBIGUOUS_REPLY_EXAMPLES, FILLER_EXAMPLES, INTENTS, LANGUAGE_INSTRUCTIONS,
                        SCRIPT_NAMES, SELF_CORRECTION_EXAMPLES, SHORT_EXAMPLES, STYLE_INSTRUCTIONS,
                        BatchSpec)
from .types import Style

SYSTEM_PROMPT = """\
You write realistic things that customers say to a voice agent on a phone call in India.
Write what the customer says out loud, as speech recognition would transcribe it:
natural spoken word order, fillers kept, no punctuation needed.
Each utterance is one complete customer turn: the customer has finished speaking
and is waiting for the agent to reply."""

TURNS_TEMPLATE = """\
Domain: {domain}
Intent: {intent} - {intent_description}
Language: {language_instruction}
Style for this batch: {style_instruction}

Write exactly {n} different customer turns:
- exactly {short} short turns of 1-4 words (for example {short_examples}),
- exactly {medium} medium turns of 5-12 words,
- exactly {long} long turns of 13-30 words.
Vary people, amounts, dates, names and places. Do not number them. Do not start two turns the same way.
Return JSON: {{"short": [...], "medium": [...], "long": [...]}}
Every turn must be in {script}."""

PAIRS_TEMPLATE = """\
Domain: {domain}
Intent: {intent} - {intent_description}
Language: {language_instruction}

The voice agent has just asked the customer a question, and the customer replies.
Write exactly {n} different (agent_question, reply) pairs. agent_question is one short question
the agent asks during this call; reply is the customer's complete answer to it.
- exactly {short} replies of 1-4 words,
- exactly {medium} replies of 5-12 words.
At least {ambiguous} replies must be complete answers that could also be the start of a longer
one, such as {ambiguous_examples}, or a bare number (an amount, a date, an OTP).
Vary the questions. Do not number them.
Return JSON: {{"pairs": [{{"agent_question": "...", "reply": "..."}}, ...]}}
Every question and reply must be in {script}."""


def _schema(name: str, properties: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "properties": properties, "required": sorted(properties),
        "additionalProperties": False}}}


_STRINGS = {"type": "array", "items": {"type": "string"}}
TURNS_FORMAT = _schema("turns", {"short": _STRINGS, "medium": _STRINGS, "long": _STRINGS})
PAIRS_FORMAT = _schema("pairs", {"pairs": {"type": "array", "items": {
    "type": "object", "properties": {"agent_question": {"type": "string"}, "reply": {"type": "string"}},
    "required": ["agent_question", "reply"], "additionalProperties": False}}})


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    response_format: dict


def render(spec: BatchSpec) -> Prompt:
    counts = spec.bucket_counts()
    common = dict(domain=spec.domain.value, intent=spec.intent,
                  intent_description=INTENTS[spec.domain][spec.intent],
                  language_instruction=LANGUAGE_INSTRUCTIONS[spec.language],
                  script=SCRIPT_NAMES[spec.language], n=spec.n, **counts)
    if spec.style is Style.SHORT_REPLIES:
        user = PAIRS_TEMPLATE.format(ambiguous=spec.n // 3,
                                     ambiguous_examples=AMBIGUOUS_REPLY_EXAMPLES[spec.language], **common)
        return Prompt(SYSTEM_PROMPT, user, PAIRS_FORMAT)
    style = STYLE_INSTRUCTIONS[spec.style].format(
        fillers=FILLER_EXAMPLES[spec.language], correction=SELF_CORRECTION_EXAMPLES[spec.language])
    user = TURNS_TEMPLATE.format(style_instruction=style, short_examples=SHORT_EXAMPLES[spec.language], **common)
    return Prompt(SYSTEM_PROMPT, user, TURNS_FORMAT)


def prompt_version() -> str:
    tables = {"intents": {d.value: i for d, i in INTENTS.items()}}
    for name, table in [("languages", LANGUAGE_INSTRUCTIONS), ("styles", STYLE_INSTRUCTIONS),
                        ("fillers", FILLER_EXAMPLES), ("corrections", SELF_CORRECTION_EXAMPLES),
                        ("scripts", SCRIPT_NAMES), ("short", SHORT_EXAMPLES),
                        ("ambiguous", AMBIGUOUS_REPLY_EXAMPLES)]:
        tables[name] = {k.value: v for k, v in table.items()}
    payload = json.dumps({"system": SYSTEM_PROMPT, "turns": TURNS_TEMPLATE, "pairs": PAIRS_TEMPLATE,
                          "formats": [TURNS_FORMAT, PAIRS_FORMAT], **tables},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
