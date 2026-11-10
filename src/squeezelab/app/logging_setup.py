"""Логирование с маскированием секретов.

Секрет, попавший в сообщение или в текст исключения, заменяется на «***».
Токен Telegram передаётся в пути URL, поэтому логгеры httpx и httpcore, которые пишут URL
целиком, переводятся на уровень WARNING.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

MASK = "***"
_MIN_SECRET_LEN = 6  # слишком короткие значения не маскируем, чтобы не портить обычный текст


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in sorted({s for s in secrets if len(s) >= _MIN_SECRET_LEN}, key=len, reverse=True):
        text = text.replace(secret, MASK)
    return text


class RedactingFormatter(logging.Formatter):
    def __init__(self, fmt: str, secrets: Iterable[str] = ()) -> None:
        super().__init__(fmt)
        self._secrets = list(secrets)

    def format(self, record: logging.LogRecord) -> str:
        # Маскируется итоговая строка вместе с трассировкой исключения.
        return redact(super().format(record), self._secrets)


def configure_logging(level: str = "INFO", secrets: Iterable[str] = ()) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        RedactingFormatter("%(asctime)sZ %(levelname)s %(name)s: %(message)s", secrets)
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
