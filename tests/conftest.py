import hashlib

from endpointing.data.types import Domain, Language, Source, Style, Utterance
from endpointing.text import normalize


def make_source(language=Language.EN, batch_id="b0", domain=Domain.BANKING,
                intent="balance_enquiry", style=Style.PLAIN) -> Source:
    return Source(batch_id=batch_id, language=language, domain=domain, intent=intent, style=style)


def make_utt(text, language=Language.EN, batch_id="b0", **source_kw) -> Utterance:
    t = normalize(text)
    uid = hashlib.sha1(t.encode()).hexdigest()[:12]
    return Utterance(utterance_id=uid, source=make_source(language, batch_id, **source_kw), text=t)
