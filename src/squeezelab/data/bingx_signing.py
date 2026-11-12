"""Подпись приватных запросов BingX: HMAC-SHA256 от строки параметров.

Схема (по документации, на реальном запросе пока не проверена):
- параметры + `timestamp` (мс, UTC) + при необходимости `recvWindow` (мс);
- строка собирается как `k1=v1&k2=v2` по ключам в порядке ASCII, без URL-кодирования;
- подпись — hex-HMAC-SHA256 этой строки секретом API, добавляется последним параметром
  `signature`; ключ API передаётся заголовком `X-BX-APIKEY`.

Значения с символами `& = ? # пробел` запрещены: такие параметры требуют кодирования, а
подпись и строка запроса обязаны совпадать байт в байт.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping

API_KEY_HEADER = "X-BX-APIKEY"
DEFAULT_RECV_WINDOW_MS = 5000
_RESERVED = frozenset({"timestamp", "recvWindow", "signature"})
_FORBIDDEN = re.compile(r"[&=?#\s]")


def _as_text(name: str, value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise TypeError(
            f"параметр {name!r}: допустимы только str и int, получено {type(value).__name__}"
        )
    text = str(value)
    if not text or _FORBIDDEN.search(text):
        raise ValueError(
            f"параметр {name!r}: пустое значение или недопустимые символы (& = ? # пробел)"
        )
    return text


def sign(query: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()


def build_signed_query(
    params: Mapping[str, str | int],
    secret: str,
    *,
    timestamp_ms: int,
    recv_window_ms: int | None = DEFAULT_RECV_WINDOW_MS,
) -> str:
    """Возвращает готовую строку запроса `...&signature=<hex>`."""
    merged: dict[str, str] = {}
    for name, value in params.items():
        if name in _RESERVED:
            raise ValueError(f"параметр {name!r} добавляется подписью автоматически")
        if _FORBIDDEN.search(name) or not name:
            raise ValueError(f"недопустимое имя параметра {name!r}")
        merged[name] = _as_text(name, value)
    merged["timestamp"] = _as_text("timestamp", timestamp_ms)
    if recv_window_ms is not None:
        merged["recvWindow"] = _as_text("recvWindow", recv_window_ms)
    query = "&".join(f"{k}={v}" for k, v in sorted(merged.items()))
    return f"{query}&signature={sign(query, secret)}"
