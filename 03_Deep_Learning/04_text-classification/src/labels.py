"""Stable inquiry-label metadata shared by training and serving documentation."""

from __future__ import annotations

LABEL_DISPLAY_NAMES = {
    "account": "계정",
    "delivery": "배송",
    "product": "상품",
    "refund": "환불",
}


def display_name(label: str) -> str:
    """Return the user-facing name when a documented label code is available."""

    return LABEL_DISPLAY_NAMES.get(label, label)

