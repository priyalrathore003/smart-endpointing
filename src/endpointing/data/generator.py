"""LLM access behind a small interface, so everything else is tested without the network."""

import json
from dataclasses import dataclass
from typing import Optional, Protocol

from .types import Style

MODEL = "gpt-4.1-mini-2025-04-14"


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


@dataclass(frozen=True)
class GeneratedItem:
    text: str
    context: Optional[str]            # the agent's preceding question, if given
    requested_bucket: Optional[str]   # length bucket the model put it in


class UtteranceGenerator(Protocol):
    model: str
    temperature: float
    max_output_tokens: int

    def complete(self, system: str, user: str, seed: int, response_format: dict) -> Completion: ...


class OpenAIGenerator:
    """The only code that talks to the API. The key is read from OPENAI_API_KEY by the SDK."""

    def __init__(self, model: str = MODEL, temperature: float = 1.0, max_output_tokens: int = 2000,
                 client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(timeout=120, max_retries=0)   # retries are explicit and budgeted in BatchRunner
        self.client, self.model = client, model
        self.temperature, self.max_output_tokens = temperature, max_output_tokens

    def complete(self, system: str, user: str, seed: int, response_format: dict) -> Completion:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, seed=seed,
            max_completion_tokens=self.max_output_tokens, response_format=response_format,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return Completion(text=resp.choices[0].message.content or "",
                          input_tokens=resp.usage.prompt_tokens,
                          output_tokens=resp.usage.completion_tokens, model=resp.model)


def parse_response(text: str, style: Style) -> list[GeneratedItem]:
    """Raw JSON response -> items. Raises ValueError if the shape is wrong. Empty strings are skipped."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response is not JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    if style is Style.SHORT_REPLIES:
        items = []
        for bucket in ("short", "medium"):
            pairs = data.get(bucket)
            if not isinstance(pairs, list) or not all(isinstance(p, dict) for p in pairs):
                raise ValueError(f"{bucket} must be a list of {{agent_question, reply}} objects")
            for p in pairs:
                q, r = p.get("agent_question"), p.get("reply")
                if not isinstance(q, str) or not isinstance(r, str):
                    raise ValueError("agent_question and reply must be strings")
                if q.strip() and r.strip():
                    items.append(GeneratedItem(r.strip(), q.strip(), bucket))
        return items
    items = []
    for bucket in ("short", "medium", "long"):
        for group, needs_context in ((bucket, False), (f"{bucket}_with_context", True)):
            turns = data.get(group)
            if not isinstance(turns, list) or not all(isinstance(x, dict) for x in turns):
                raise ValueError(f"{group} must be a list of objects")
            for x in turns:
                text, ctx = x.get("text"), x.get("context", "") if needs_context else ""
                if not isinstance(text, str) or not isinstance(ctx, str):
                    raise ValueError("text and context must be strings")
                if needs_context and "context" not in x:
                    raise ValueError(f"{group} items need a context")
                if text.strip():
                    items.append(GeneratedItem(text.strip(), ctx.strip() or None, bucket))
    return items
