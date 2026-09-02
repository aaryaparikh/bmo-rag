"""Local token counting with a documented fallback when tiktoken is unavailable."""

from __future__ import annotations

import requests


def count_text_tokens(text: str, model: str) -> tuple[int, str]:
    try:
        import tiktoken
    except ImportError:
        return _fallback_count(text), "utf8_bytes_div_4_estimate"

    try:
        try:
            encoding = tiktoken.encoding_for_model(model)
            method = f"tiktoken:{encoding.name}:model"
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
            method = "tiktoken:o200k_base:fallback"
    except (OSError, requests.RequestException):
        return _fallback_count(text), "utf8_bytes_div_4_estimate"
    return len(encoding.encode(text)), method


def _fallback_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)
