"""Prompt templates for utterance generation. Their hash is recorded with every raw response."""

import hashlib
import json

from .catalogue import INTENTS, LANGUAGE_INSTRUCTIONS, STYLE_INSTRUCTIONS, BatchSpec

SYSTEM_PROMPT = """\
You write realistic things that customers say to a voice agent on a phone call in India.
Write what the customer says out loud, as speech recognition would transcribe it:
natural spoken word order, fillers kept, no punctuation needed.
Each utterance is one complete customer turn: the customer has finished speaking
and is waiting for the agent to reply."""

USER_TEMPLATE = """\
Domain: {domain}
Intent: {intent} - {intent_description}
Language: {language_instruction}
Style for this batch: {style_instruction}

Write {n} different customer turns. Lengths: about a quarter short (1-4 words),
half medium (5-12 words), a quarter long (13-30 words). Vary people, amounts, dates,
names and places. Do not number them. Do not start two turns the same way.
Return JSON: {{"utterances": [ ... ]}}"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "utterances",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"utterances": {"type": "array", "items": {"type": "string"}}},
            "required": ["utterances"],
            "additionalProperties": False,
        },
    },
}


def render(spec: BatchSpec) -> tuple[str, str]:
    user = USER_TEMPLATE.format(
        domain=spec.domain.value, intent=spec.intent,
        intent_description=INTENTS[spec.domain][spec.intent],
        language_instruction=LANGUAGE_INSTRUCTIONS[spec.language],
        style_instruction=STYLE_INSTRUCTIONS[spec.style], n=spec.n)
    return SYSTEM_PROMPT, user


def prompt_version() -> str:
    payload = json.dumps({
        "system": SYSTEM_PROMPT, "user": USER_TEMPLATE, "format": RESPONSE_FORMAT,
        "intents": {d.value: i for d, i in INTENTS.items()},
        "languages": {k.value: v for k, v in LANGUAGE_INSTRUCTIONS.items()},
        "styles": {k.value: v for k, v in STYLE_INSTRUCTIONS.items()},
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
