"""Conservative PII masking for inquiry text before it reaches a model."""

from __future__ import annotations

import re

MASK_TOKENS = {"email": "[EMAIL]", "phone": "[PHONE]", "order_id": "[ORDER_ID]"}

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:0(?:10|[2-6]\d?|70)[-\s]?\d{3,4}[-\s]?\d{4})(?!\d)")
ORDER_ID_PATTERN = re.compile(
    r"(?i)(?:주문\s*(?:번호|번)?|order\s*(?:id|number|no\.?)?)\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{5,}"
)


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace supported PII with stable tokens and return replacement counts.

    This is a defence-in-depth preprocessing step, not a guarantee that every
    identifying value has been removed. Raw data access still needs controls.
    """

    masked = str(text)
    counts: dict[str, int] = {}
    for name, pattern in (("email", EMAIL_PATTERN), ("phone", PHONE_PATTERN), ("order_id", ORDER_ID_PATTERN)):
        masked, count = pattern.subn(MASK_TOKENS[name], masked)
        counts[name] = count
    return masked, counts

