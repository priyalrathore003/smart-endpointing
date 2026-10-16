"""Prompt templates for utterance generation. Their hash is recorded with every raw response.

Two shapes, each with its own prompt version, so changing one does not invalidate the other:
- turns: one complete customer turn each, requested per length bucket; about 30% also
  carry the agent question that came just before (`context`), so that context is not
  a give-away for END (short replies are almost all END);
- short replies: (agent_question, reply) pairs; the question becomes the row's `context`.
"""

import hashlib
import json
from dataclasses import dataclass

from .catalogue import (AMBIGUOUS_REPLY_EXAMPLES, CONTEXT_SHARE_TURNS, FILLER_EXAMPLES, INTENTS, LANGUAGE_INSTRUCTIONS,
                        SCRIPT_NAMES, SELF_CORRECTION_EXAMPLES, SHORT_EXAMPLES, STYLE_INSTRUCTIONS,
                        BatchSpec)
from .types import Language, Style

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

Write exactly {n} different customer turns in these six groups:
- "short": exactly {short_plain} turns of 1-4 words (for example {short_examples}),
- "short_with_context": exactly {short_ctx} turns of 1-4 words, each with the agent's question before it,
- "medium": exactly {medium_plain} turns of 5-12 words,
- "medium_with_context": exactly {medium_ctx} turns of 5-12 words, each with the agent's question before it,
- "long": exactly {long_plain} turns of 13-30 words,
- "long_with_context": exactly {long_ctx} turns of 13-30 words, each with the agent's question before it.
In the *_with_context groups, "context" is the question the agent asked just before that turn.
Vary people, amounts, dates, names and places. Do not number them. Do not start two turns the same way.
Return JSON: {{"short": [{{"text": "..."}}, ...], "short_with_context": [{{"text": "...", "context": "..."}}, ...],
"medium": [...], "medium_with_context": [...], "long": [...], "long_with_context": [...]}}
Every turn and question must be in {script}."""

PAIRS_TEMPLATE = """\
Domain: {domain}
Intent: {intent} - {intent_description}
Language: {language_instruction}

The voice agent has just asked the customer a question, and the customer replies.
Write exactly {n} different (agent_question, reply) pairs. agent_question is one short question
the agent asks during this call; reply is the customer's complete answer to it.
- "short": exactly {short} pairs whose reply is 1-4 words,
- "medium": exactly {medium} pairs whose reply is 5-12 words.
At least {ambiguous} replies must be complete answers that could also be the start of a longer
one, such as {ambiguous_examples}, or a bare number (an amount, a date, an OTP).
Vary the questions. Do not number them.
Return JSON: {{"short": [{{"agent_question": "...", "reply": "..."}}, ...], "medium": [...]}}
Every question and reply must be in {script}."""


def _schema(name: str, properties: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "properties": properties, "required": sorted(properties),
        "additionalProperties": False}}}


_PLAIN = {"type": "array", "items": {
    "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}}
_WITH_CONTEXT = {"type": "array", "items": {
    "type": "object", "properties": {"text": {"type": "string"}, "context": {"type": "string"}},
    "required": ["context", "text"], "additionalProperties": False}}
# Turns with a question get their own arrays: asked in prose, pilot v4 got 2-11 questions for 5 requested.
TURN_GROUPS = ("short", "short_with_context", "medium", "medium_with_context", "long", "long_with_context")
TURNS_FORMAT = _schema("turns", {g: _WITH_CONTEXT if g.endswith("_with_context") else _PLAIN for g in TURN_GROUPS})
_PAIRS = {"type": "array", "items": {
    "type": "object", "properties": {"agent_question": {"type": "string"}, "reply": {"type": "string"}},
    "required": ["agent_question", "reply"], "additionalProperties": False}}
# One array per length bucket, as for turns: in pilot v2 a single list got 8 short / 10 medium in Hindi.
PAIRS_FORMAT = _schema("pairs", {"short": _PAIRS, "medium": _PAIRS})


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
    plan = {f"{b}_{kind}": v for b, (plain, ctx) in spec.turn_plan().items()
            for kind, v in (("plain", plain), ("ctx", ctx))}
    user = TURNS_TEMPLATE.format(style_instruction=style, short_examples=SHORT_EXAMPLES[spec.language],
                                 **plan, **common)
    return Prompt(SYSTEM_PROMPT, user, TURNS_FORMAT)


SHAPES = ("turns", "pairs")


def shape_of(style: Style) -> str:
    return "pairs" if style is Style.SHORT_REPLIES else "turns"


def prompt_version(shape: str, language: Language | None = None) -> str:
    """Hash of everything that goes into prompts of one shape. Turn prompts are versioned per
    language (their context share is per language); short-reply prompts are not."""
    if shape not in SHAPES:
        raise ValueError(f"unknown prompt shape {shape!r}")
    if shape == "turns" and language is None:
        raise ValueError("turn prompt versions are per language")
    common = [("languages", LANGUAGE_INSTRUCTIONS), ("scripts", SCRIPT_NAMES)]
    own = ([("styles", STYLE_INSTRUCTIONS), ("fillers", FILLER_EXAMPLES), ("corrections", SELF_CORRECTION_EXAMPLES),
            ("short", SHORT_EXAMPLES)] if shape == "turns" else [("ambiguous", AMBIGUOUS_REPLY_EXAMPLES)])
    tables = {"intents": {d.value: i for d, i in INTENTS.items()}}
    for name, table in common + own:
        tables[name] = {k.value: v for k, v in table.items()}
    template, fmt = (TURNS_TEMPLATE, TURNS_FORMAT) if shape == "turns" else (PAIRS_TEMPLATE, PAIRS_FORMAT)
    extra = {"language": language.value, "context_share_turns": CONTEXT_SHARE_TURNS[language]} \
        if shape == "turns" else {}
    payload = json.dumps({"system": SYSTEM_PROMPT, "template": template, "format": fmt, **tables, **extra},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
