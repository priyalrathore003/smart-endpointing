"""Money: prices, an append-only spend ledger, and a hard cap.

Before each call the runner reserves the call's worst-case cost (max input + max
output tokens). The check against the cap and the reservation happen under one
lock, so concurrent workers can never jointly overshoot the cap. After the call
the reservation is replaced by the actual cost from the API's usage numbers.
"""

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class Pricing:
    input_per_million: Decimal
    output_per_million: Decimal

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        return (self.input_per_million * input_tokens + self.output_per_million * output_tokens) / 1_000_000


# OpenAI list price for gpt-4.1-mini, checked 2026-10-16.
GPT_41_MINI = Pricing(Decimal("0.40"), Decimal("1.60"))


class SpendLedger:
    def __init__(self, path: Path, pricing: Pricing, cap: Decimal):
        if cap <= 0:
            raise ValueError("cap must be positive")
        self.path, self.pricing, self.cap = Path(path), pricing, cap
        self._lock = threading.Lock()
        self._reserved = Decimal(0)
        self._spent = sum((Decimal(e["cost_usd"]) for e in self.entries()), Decimal(0))

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    @property
    def spent(self) -> Decimal:
        with self._lock:
            return self._spent

    def try_reserve(self, amount: Decimal) -> bool:
        with self._lock:
            if self._spent + self._reserved + amount > self.cap:
                return False
            self._reserved += amount
            return True

    def settle(self, reserved: Decimal, batch_id: str, model: str, input_tokens: int,
               output_tokens: int, ok: bool) -> Decimal:
        cost = self.pricing.cost(input_tokens, output_tokens)
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "batch_id": batch_id,
                 "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens,
                 "cost_usd": str(cost), "parsed_ok": ok}
        with self._lock:
            self._reserved -= reserved
            self._spent += cost
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        return cost

    def release(self, reserved: Decimal) -> None:
        """Drop a reservation for a call that never reached the API."""
        with self._lock:
            self._reserved -= reserved
