"""Минимальный клиент Telegram Bot API: только sendMessage и только в чаты из белого списка.

Токен находится в пути URL (`/bot<TOKEN>/sendMessage`), поэтому:
- URL нигде не логируется и не попадает в тексты ошибок;
- исключения httpx не пробрасываются дальше (в их тексте бывает полный URL);
- сообщения об ошибках собираются только из типа ошибки и ответа Telegram.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import SecretStr

from squeezelab.app.http_clients import build_telegram_http_client
from squeezelab.app.logging_setup import redact

TELEGRAM_API_URL = "https://api.telegram.org"
MAX_TEXT_LEN = 4096  # лимит Telegram на длину сообщения


class TelegramError(Exception):
    """Ошибка отправки. Текст понятен человеку и не содержит токена."""


class ChatNotAllowedError(TelegramError):
    """chat_id не входит в белый список; запрос в сеть не отправлялся."""


def build_test_message(
    mode: str, strategy_hash: str, app_hash: str, now: datetime | None = None
) -> str:
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return (
        "SqueezeLab: тестовое сообщение\n"
        f"Режим: {mode}\n"
        f"Хэш стратегии: {strategy_hash}\n"
        f"Хэш приложения: {app_hash}\n"
        f"Время: {moment:%Y-%m-%d %H:%M:%S} UTC"
    )


class TelegramClient:
    def __init__(
        self,
        token: SecretStr,
        allowed_chat_ids: Sequence[int],
        *,
        base_url: str = TELEGRAM_API_URL,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        proxy: SecretStr | None = None,
    ) -> None:
        self._token = token
        self._allowed = frozenset(allowed_chat_ids)
        self._base_url = base_url.rstrip("/")
        # Прокси только явный (из TELEGRAM_PROXY_URL); HTTP(S)_PROXY процесса не используются.
        self._http = build_telegram_http_client(
            timeout=timeout,
            transport=transport,
            proxy=proxy.get_secret_value() if proxy is not None else None,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _clean(self, text: str) -> str:
        return redact(text, [self._token.get_secret_value()])

    def send_message(self, chat_id: int, text: str) -> None:
        """Отправляет простой текст (без разметки) в один чат из белого списка."""
        if chat_id not in self._allowed:
            raise ChatNotAllowedError(
                f"chat_id {chat_id} не входит в белый список TELEGRAM_CHAT_IDS: не отправлено"
            )
        if not text.strip():
            raise TelegramError("пустой текст сообщения")
        if len(text) > MAX_TEXT_LEN:
            raise TelegramError(f"текст длиннее {MAX_TEXT_LEN} символов ({len(text)})")

        url = f"{self._base_url}/bot{self._token.get_secret_value()}/sendMessage"
        try:
            response = self._http.post(url, json={"chat_id": chat_id, "text": text})
        except httpx.TimeoutException:
            raise TelegramError(
                "таймаут при обращении к Telegram API: проверь сеть; api.telegram.org "
                "может быть недоступен без VPN"
            ) from None
        except httpx.HTTPError as exc:
            # Текст исключения не берём: в нём может быть URL с токеном.
            raise TelegramError(
                f"сетевая ошибка при обращении к Telegram API ({type(exc).__name__})"
            ) from None

        self._check_response(response, chat_id)

    def send_to_all(self, text: str) -> list[int]:
        """Шлёт во все чаты белого списка; при первой ошибке прекращает. Возвращает chat_id."""
        sent: list[int] = []
        for chat_id in sorted(self._allowed):
            self.send_message(chat_id, text)
            sent.append(chat_id)
        return sent

    def _check_response(self, response: httpx.Response, chat_id: int) -> None:
        try:
            body: Any = response.json()
        except ValueError:
            raise TelegramError(
                f"ответ Telegram не является JSON (HTTP {response.status_code})"
            ) from None
        if isinstance(body, dict) and body.get("ok") is True:
            return

        code = body.get("error_code", response.status_code) if isinstance(body, dict) else None
        description = str(body.get("description", "")) if isinstance(body, dict) else ""
        description = self._clean(description)

        if code == 401:
            raise TelegramError("токен отклонён (401 Unauthorized): проверь TELEGRAM_BOT_TOKEN")
        if code == 403:
            raise TelegramError(
                f"бот не может писать в чат {chat_id} (403): добавь бота в группу или канал, "
                f"либо нажми Start в личном чате с ботом. Ответ Telegram: {description}"
            )
        if code == 400 and "chat not found" in description.lower():
            raise TelegramError(
                f"чат {chat_id} не найден (400): проверь chat_id; бот должен быть участником "
                "группы или канала, а в личном чате ты должен сначала написать боту"
            )
        if code == 429:
            retry = ""
            if isinstance(body, dict):
                params = body.get("parameters")
                if isinstance(params, dict) and "retry_after" in params:
                    retry = f", повтори через {params['retry_after']} с"
            raise TelegramError(f"слишком частые запросы (429){retry}")
        raise TelegramError(f"Telegram вернул ошибку {code}: {description or 'без описания'}")
