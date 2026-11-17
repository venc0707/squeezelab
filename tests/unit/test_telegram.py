"""Тесты P00-T03 и P01-T01: клиент Telegram на mock-транспорте (внешняя сеть не используется)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from squeezelab.app.cli import main
from squeezelab.app.logging_setup import RedactingFormatter
from squeezelab.notify.telegram import (
    ChatNotAllowedError,
    TelegramClient,
    TelegramError,
    build_test_message,
)

TOKEN = "123456789:AAH-super-secret-token-value"
CHAT = 555
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


class Recorder:
    """Транспорт-заглушка: запоминает запросы и отдаёт заданный ответ."""

    def __init__(self, handler: httpx.MockTransport | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.transport = handler or httpx.MockTransport(self._ok)

    def _ok(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})


def _client(transport: httpx.MockTransport, allowed: list[int] | None = None) -> TelegramClient:
    return TelegramClient(SecretStr(TOKEN), allowed or [CHAT], transport=transport)


def _error_transport(status: int, body: object) -> httpx.MockTransport:
    return httpx.MockTransport(lambda _request: httpx.Response(status, json=body))


def test_send_success() -> None:
    rec = Recorder()
    with _client(rec.transport) as client:
        client.send_message(CHAT, "привет")
    assert len(rec.requests) == 1
    request = rec.requests[0]
    assert request.method == "POST"
    assert request.url.path == f"/bot{TOKEN}/sendMessage"
    assert json.loads(request.content) == {"chat_id": CHAT, "text": "привет"}


def test_foreign_chat_id_is_refused_without_network() -> None:
    rec = Recorder()
    with _client(rec.transport) as client, pytest.raises(ChatNotAllowedError, match="999"):
        client.send_message(999, "текст")
    assert rec.requests == []


def test_send_to_all_only_whitelist() -> None:
    rec = Recorder()
    with _client(rec.transport, [CHAT, -1001]) as client:
        assert client.send_to_all("текст") == [-1001, CHAT]
    assert len(rec.requests) == 2


def test_empty_and_too_long_text() -> None:
    rec = Recorder()
    with _client(rec.transport) as client:
        with pytest.raises(TelegramError, match="пустой"):
            client.send_message(CHAT, "   ")
        with pytest.raises(TelegramError, match="длиннее"):
            client.send_message(CHAT, "x" * 5000)
    assert rec.requests == []


@pytest.mark.parametrize(
    ("status", "body", "fragment"),
    [
        (401, {"ok": False, "error_code": 401, "description": "Unauthorized"}, "токен отклонён"),
        (
            400,
            {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"},
            "не найден",
        ),
        (
            403,
            {"ok": False, "error_code": 403, "description": "Forbidden: bot was blocked"},
            "не может писать",
        ),
        (
            429,
            {
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests",
                "parameters": {"retry_after": 7},
            },
            "через 7 с",
        ),
        (
            500,
            {"ok": False, "error_code": 500, "description": "Internal Server Error"},
            "Internal Server Error",
        ),
    ],
)
def test_api_errors_are_explained(status: int, body: object, fragment: str) -> None:
    with _client(_error_transport(status, body)) as client, pytest.raises(TelegramError) as exc:
        client.send_message(CHAT, "текст")
    assert fragment in str(exc.value)
    assert TOKEN not in str(exc.value)


def test_non_json_response() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(502, text="<html>Bad Gateway</html>"))
    with _client(transport) as client, pytest.raises(TelegramError, match="не является JSON"):
        client.send_message(CHAT, "текст")


def test_token_in_description_is_masked() -> None:
    body = {"ok": False, "error_code": 400, "description": f"странно: {TOKEN}"}
    with _client(_error_transport(400, body)) as client, pytest.raises(TelegramError) as exc:
        client.send_message(CHAT, "текст")
    assert TOKEN not in str(exc.value)


def test_network_error_does_not_leak_token() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {request.url}")

    with _client(httpx.MockTransport(boom)) as client, pytest.raises(TelegramError) as exc:
        client.send_message(CHAT, "текст")
    assert "ConnectError" in str(exc.value)
    assert TOKEN not in str(exc.value)
    assert exc.value.__cause__ is None


def test_timeout_message() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with _client(httpx.MockTransport(slow)) as client, pytest.raises(TelegramError) as exc:
        client.send_message(CHAT, "текст")
    assert "таймаут" in str(exc.value)
    assert TOKEN not in str(exc.value)


def test_error_text_survives_redacting_log_formatter() -> None:
    formatter = RedactingFormatter("%(message)s", [TOKEN])
    record = logging.LogRecord(
        "t", logging.ERROR, __file__, 1, f"url=https://api.telegram.org/bot{TOKEN}/x", None, None
    )
    assert TOKEN not in formatter.format(record)


def test_build_test_message() -> None:
    now = datetime(2026, 11, 10, 12, 30, 5, tzinfo=UTC)
    text = build_test_message("signals_only", "a" * 64, "b" * 64, now)
    assert "Режим: signals_only" in text
    assert "Хэш стратегии: " + "a" * 64 in text
    assert "Хэш приложения: " + "b" * 64 in text
    assert "2026-11-10 12:30:05 UTC" in text


# ------------------------------------------------------------------ команда notify-test


def _env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, token: bool = True, proxy: str = ""
) -> Path:
    for name in (
        "BINGX_API_KEY",
        "BINGX_API_SECRET",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_IDS",
        "TELEGRAM_PROXY_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    lines = [f"TELEGRAM_BOT_TOKEN={TOKEN}"] if token else []
    lines.append(f"TELEGRAM_CHAT_IDS={CHAT}")
    if proxy:
        lines.append(f"TELEGRAM_PROXY_URL={proxy}")
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env


def test_cli_dry_run_sends_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("в сухом прогоне сеть не используется")

    monkeypatch.setattr(httpx.Client, "post", forbidden)
    code = main(
        ["notify-test", "--dry-run", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Сухой прогон" in out
    assert "Режим: signals_only" in out
    assert str(CHAT) in out
    assert "Прокси Telegram: не задан" in out
    assert TOKEN not in out


def test_cli_dry_run_reports_proxy_without_showing_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch, proxy="http://user:pa55word@127.0.0.1:10809")
    code = main(
        ["notify-test", "--dry-run", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Прокси Telegram: задан" in out
    assert "pa55word" not in out
    assert "127.0.0.1:10809" not in out


def test_cli_passes_proxy_from_env_to_client(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch, proxy="http://127.0.0.1:10809")
    seen: list[str | None] = []
    original_init = TelegramClient.__init__

    def spy_init(self: TelegramClient, *args: object, **kwargs: object) -> None:
        proxy = kwargs.get("proxy")
        seen.append(proxy.get_secret_value() if isinstance(proxy, SecretStr) else None)
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(TelegramClient, "__init__", spy_init)
    monkeypatch.setattr(TelegramClient, "send_message", lambda *_a, **_k: None)
    code = main(["notify-test", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    assert code == 0
    assert seen == ["http://127.0.0.1:10809"]
    assert "pa55word" not in capsys.readouterr().out


def test_cli_without_telegram_settings(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch, token=False)
    code = main(["notify-test", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    assert code == 2
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err


def test_cli_real_send_uses_whitelist_and_reports(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)
    sent: list[tuple[int, str]] = []

    def fake_send(self: TelegramClient, chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    monkeypatch.setattr(TelegramClient, "send_message", fake_send)
    code = main(["notify-test", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    assert code == 0
    assert [cid for cid, _ in sent] == [CHAT]
    assert "Хэш стратегии:" in sent[0][1]
    assert "Хэш приложения:" in sent[0][1]
    assert f"chat_id {CHAT}: отправлено" in capsys.readouterr().out


def test_cli_reports_api_failure_with_exit_code_1(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)

    def failing(self: TelegramClient, chat_id: int, text: str) -> None:
        raise TelegramError("токен отклонён (401 Unauthorized)")

    monkeypatch.setattr(TelegramClient, "send_message", failing)
    code = main(["notify-test", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    captured = capsys.readouterr()
    assert code == 1
    assert "ОШИБКА" in captured.err
    assert TOKEN not in captured.err + captured.out
