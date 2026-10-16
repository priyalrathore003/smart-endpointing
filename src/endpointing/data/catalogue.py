"""What to generate: domains x intents x styles x languages, one LLM call (batch) per cell.

Language mix EN 30 / HI 30 / Hinglish 40 comes from utterances per call (18/18/24)
over the same 160 cells per language: 2,880 + 2,880 + 3,840 = 9,600.
Every prompt asks for exact counts per length bucket: the pilot showed that "about a
quarter short" produced no turn under 5 words.
"""

from dataclasses import dataclass

from .types import Domain, Language, Source, Style

INTENTS: dict[Domain, dict[str, str]] = {
    Domain.BANKING: {
        "balance_enquiry": "asks for their account balance or recent transactions",
        "lost_card": "reports a lost or stolen card and wants it blocked",
        "failed_upi_refund": "a UPI or online payment failed but money was debited; wants a refund",
        "loan_emi_query": "asks about a loan EMI amount, due date or a deduction",
        "kyc_update": "wants to update KYC, address, PAN or Aadhaar details",
        "cheque_book": "requests a new cheque book or asks about its delivery",
        "unauthorised_debit": "reports a transaction they did not make",
        "login_otp_problem": "cannot log in to net or mobile banking or is not receiving the OTP",
    },
    Domain.COLLECTIONS: {
        "promise_to_pay": "replies to a payment reminder with a date or amount they will pay",
        "dispute_amount": "says the outstanding amount or a charge is wrong",
        "already_paid": "says they have already paid and gives details",
        "ask_extension": "asks for more time or to split the payment",
        "financial_hardship": "explains they cannot pay right now because of job loss, illness or similar",
        "wrong_person": "says the agent has the wrong person or number",
        "payment_link": "asks for a payment link or how to pay",
        "callback_request": "asks the agent to call back at another time",
    },
    Domain.DELIVERY: {
        "where_is_order": "asks where their order or parcel is",
        "reschedule": "wants the delivery on a different day or time",
        "change_address": "wants to change the delivery address or landmark",
        "wrong_or_damaged": "received a wrong or damaged item",
        "return_refund": "wants to return an item or asks about a refund",
        "agent_unreachable": "the delivery person is not answering or cannot find the address",
        "cancel_order": "wants to cancel an order",
        "cod_payment": "has a problem with cash on delivery or payment at the door",
    },
    Domain.TELECOM: {
        "recharge_not_credited": "recharged but the plan or balance was not credited",
        "no_network": "has no signal, call drops or slow internet",
        "data_balance": "asks how much data or validity is left",
        "bill_dispute": "says a postpaid bill or charge is wrong",
        "port_or_sim": "wants to port the number, replace or swap a SIM",
        "plan_change": "wants to change or understand a plan",
        "roaming": "wants to activate international or national roaming",
        "escalate_complaint": "is unhappy that an earlier complaint was not solved and wants escalation",
    },
}

LANGUAGE_INSTRUCTIONS: dict[Language, str] = {
    Language.EN: "Indian English, Latin script.",
    Language.HI: ("Colloquial spoken Hindi as on a real phone call, not written or formal Hindi. "
                  "Avoid कृपया, श्रीमान, रकम, जमा; callers say सर, मैडम, प्लीज़, जल्दी, पैसा. "
                  "Each utterance stays on one topic (no account number in a delivery query). "
                  "Devanagari script. Write most English loanwords in Devanagari as people "
                  "say them (डिलीवरी, रिफंड), but in about 1 in 4 turns keep one common English noun "
                  "in Latin script, as in real chats (parcel, account, OTP, payment)."),
    Language.HINGLISH: ("Hindi-English code-mixed, entirely in Latin script "
                        "(e.g. 'mera account se paise kat gaye but transfer nahi hua')."),
}

# Filler and self-correction examples are per language and in that language's script:
# Latin-script examples in a Hindi prompt made the model answer in Hinglish (pilot v0).
FILLER_EXAMPLES: dict[Language, str] = {
    Language.EN: "umm, uh, like, actually, I mean",
    Language.HI: "अम्म, मतलब, वो, अच्छा, हाँ जी",
    Language.HINGLISH: "umm, matlab, woh, accha, haan ji",
}

SELF_CORRECTION_EXAMPLES: dict[Language, str] = {
    Language.EN: "Tuesday, no, Wednesday",
    Language.HI: "मंगलवार, नहीं, बुधवार",
    Language.HINGLISH: "Tuesday, nahi, Wednesday",
}

SCRIPT_NAMES: dict[Language, str] = {
    Language.EN: "Latin script",
    Language.HI: "Devanagari script, apart from that occasional English noun",
    Language.HINGLISH: "Latin script",
}

STYLE_INSTRUCTIONS: dict[Style, str] = {
    Style.PLAIN: "plain and direct",
    Style.FILLERS: ("with fillers and hesitations ({fillers}); at least a third "
                    "include a self-correction ('{correction}')"),
    Style.NUMBERS: ("with numbers (account and phone numbers, OTPs, amounts, dates) written as numerals, "
                    "with long numbers spaced in groups the way people say them"),
    Style.NAMES: "with names, addresses or places; some phrased as questions",
    Style.SHORT_REPLIES: "short replies to a question the agent has just asked",
}

SHORT_EXAMPLES: dict[Language, str] = {
    Language.EN: "'yes please', 'tomorrow', 'card ending 4321'",
    Language.HI: "'हाँ जी', 'कल तक', 'ठीक है'",
    Language.HINGLISH: "'haan ji', 'kal tak', 'theek hai'",
}

# Replies that are complete as they stand but could also start a longer answer.
AMBIGUOUS_REPLY_EXAMPLES: dict[Language, str] = {
    Language.EN: "'yes', 'okay', 'tomorrow', 'by friday'",
    Language.HI: "'हाँ जी', 'ठीक है', 'कल तक', 'अभी नहीं'",
    Language.HINGLISH: "'haan ji', 'theek hai', 'kal tak', 'abhi nahi'",
}

LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (("short", 1, 4), ("medium", 5, 12), ("long", 13, 30))


def length_bucket(n_words: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= n_words <= hi:
            return name
    return "long" if n_words > 30 else "empty"

UTTERANCES_PER_CALL: dict[Language, int] = {Language.EN: 18, Language.HI: 18, Language.HINGLISH: 24}

PILOT_BATCH_IDS: tuple[str, ...] = (
    "en.banking.unauthorised_debit.numbers",
    "hi.collections.promise_to_pay.fillers",
    "hi.delivery.where_is_order.numbers",
    "hinglish.telecom.recharge_not_credited.fillers",
    "hinglish.banking.login_otp_problem.numbers",
    "en.telecom.data_balance.short_replies",
    "hi.collections.promise_to_pay.short_replies",
    "hinglish.delivery.reschedule.short_replies",
    # More turn batches so the pilot has the full run's 4 turn batches per short-reply
    # batch in each language; the context-balance check depends on that mix.
    "en.delivery.reschedule.plain",
    "en.collections.promise_to_pay.fillers",
    "en.telecom.bill_dispute.names",
    "hi.banking.lost_card.plain",
    "hi.telecom.no_network.names",
    "hinglish.collections.ask_extension.plain",
    "hinglish.delivery.change_address.names",
)

# Share of turn-style utterances that also get the preceding agent question. Pilot v4 at 0.3:
# Hindi's P(END) gap with vs without context was 16.9 points (limit 15). Per language so one
# language can be raised without regenerating the others.
CONTEXT_SHARE_TURNS: dict[Language, float] = {Language.EN: 0.45, Language.HI: 0.45, Language.HINGLISH: 0.45}


@dataclass(frozen=True)
class BatchSpec:
    language: Language
    domain: Domain
    intent: str
    style: Style
    n: int

    @property
    def batch_id(self) -> str:
        return f"{self.language}.{self.domain}.{self.intent}.{self.style}"

    @property
    def source(self) -> Source:
        return Source(self.batch_id, self.language, self.domain, self.intent, self.style)

    def bucket_counts(self) -> dict[str, int]:
        """Exact number of turns to request per length bucket. Sums to n.
        Turns: 25 / 50 / 25. Short replies: 75% of 1-4 words, 25% of 5-12, none long."""
        if self.style is Style.SHORT_REPLIES:
            medium = round(self.n / 4)
            return {"short": self.n - medium, "medium": medium, "long": 0}
        short = long = round(self.n / 4)
        return {"short": short, "medium": self.n - short - long, "long": long}

    def context_count(self) -> int:
        """How many items come with the agent's preceding question."""
        return self.n if self.style is Style.SHORT_REPLIES else round(self.n * CONTEXT_SHARE_TURNS[self.language])

    def turn_plan(self) -> dict[str, tuple[int, int]]:
        """Per length bucket: (turns without a question, turns with one). The questions are spread
        over the buckets in proportion to their size (largest remainder), so the totals are exact."""
        counts, total = self.bucket_counts(), self.context_count()
        exact = {b: counts[b] * total / self.n for b in counts}
        with_ctx = {b: int(exact[b]) for b in counts}
        for b in sorted(counts, key=lambda b: exact[b] - int(exact[b]), reverse=True)[:total - sum(with_ctx.values())]:
            with_ctx[b] += 1
        return {b: (counts[b] - with_ctx[b], with_ctx[b]) for b in ("short", "medium", "long")}


def plan_batches() -> list[BatchSpec]:
    specs = [BatchSpec(lang, domain, intent, style, UTTERANCES_PER_CALL[lang])
             for lang in Language for domain in Domain for intent in INTENTS[domain] for style in Style]
    return sorted(specs, key=lambda s: s.batch_id)


def pilot_batches() -> list[BatchSpec]:
    by_id = {s.batch_id: s for s in plan_batches()}
    return [by_id[b] for b in PILOT_BATCH_IDS]
