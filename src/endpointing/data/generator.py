"""LLM access behind a small interface, so everything else is tested without the network."""

import json
from dataclasses import dataclass
from typing import Protocol

from .prompts import RESPONSE_FORMAT

MODEL = "gpt-4.1-mini-2025-04-14"


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class UtteranceGenerator(Protocol):
    model: str
    temperature: float
    max_output_tokens: int

    def complete(self, system: str, user: str, seed: int) -> Completion: ...


class OpenAIGenerator:
    """The only code that talks to the API. The key is read from OPENAI_API_KEY by the SDK."""

    def __init__(self, model: str = MODEL, temperature: float = 1.0, max_output_tokens: int = 2000,
                 client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(timeout=120, max_retries=0)   # retries are explicit and budgeted in BatchRunner
        self.client, self.model = client, model
        self.temperature, self.max_output_tokens = temperature, max_output_tokens

    def complete(self, system: str, user: str, seed: int) -> Completion:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, seed=seed,
            max_completion_tokens=self.max_output_tokens, response_format=RESPONSE_FORMAT,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return Completion(text=resp.choices[0].message.content or "",
                          input_tokens=resp.usage.prompt_tokens,
                          output_tokens=resp.usage.completion_tokens, model=resp.model)


def parse_utterances(text: str) -> list[str]:
    """Raw JSON response -> list of non-empty strings. Raises ValueError on any other shape."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response is not JSON: {e}") from e
    items = data.get("utterances") if isinstance(data, dict) else None
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("response must be {\"utterances\": [string, ...]}")
    return [x.strip() for x in items if x.strip()]
