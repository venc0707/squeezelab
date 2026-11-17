"""Тесты P01-T01: фабрика HTTP-клиентов.

BingX всегда идёт напрямую (HTTP_PROXY и HTTPS_PROXY не действуют). Telegram использует только
явный параметр `proxy`. Внешняя сеть не используется: проверяется выбор транспорта и обращение
к фальшивому прокси на 127.0.0.1.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import httpx
import pytest
from pydantic import SecretStr

from squeezelab.app.http_clients import build_bingx_http_client, build_telegram_http_client
from squeezelab.data.bingx_rest import (
    LIVE_BASE_URL,
    VST_BASE_URL,
    BingXPublicClient,
    BingXReadOnlyClient,
)
from squeezelab.notify.telegram import TelegramClient, TelegramError

PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
BINGX_URL = "https://open-api.bingx.com/openApi/swap/v3/quote/klines"
TELEGRAM_URL = "https://api.telegram.org/bot1:x/sendMessage"
TOKEN = "123456789:AAH-super-secret-token-value"


@pytest.fixture
def proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Глобальные переменные прокси указывают на недостижимый адрес."""
    for name in PROXY_ENV_NAMES:
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)


def uses_proxy(client: httpx.Client, url: str) -> bool:
    """Пойдёт ли запрос на этот адрес через прокси, а не через основной транспорт клиента."""
    return client._transport_for_url(httpx.URL(url)) is not client._transport


class FakeProxy:
    """Принимает одно соединение, запоминает первую строку запроса и отвечает 403."""

    def __init__(self) -> None:
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(5)
        self.url = f"http://127.0.0.1:{self._server.getsockname()[1]}"
        self.first_lines: list[str] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            data = conn.recv(4096)
            self.first_lines.append(data.split(b"\r\n", 1)[0].decode("ascii", "replace"))
            conn.sendall(
                b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )

    def close(self) -> None:
        self._server.close()
        self._thread.join(timeout=5)


@pytest.fixture
def fake_proxy() -> Iterator[FakeProxy]:
    proxy = FakeProxy()
    try:
        yield proxy
    finally:
        proxy.close()


# --------------------------------------------------------------------------- контроль теста


def test_control_default_httpx_client_does_take_proxy_from_environment(proxy_env: None) -> None:
    """Контроль: без него тесты ниже проходили бы, даже если переменные ни на что не влияют."""
    with httpx.Client() as client:
        assert uses_proxy(client, BINGX_URL)
    with httpx.Client(trust_env=False) as client:
        assert not uses_proxy(client, BINGX_URL)


# ------------------------------------------------------------------------------- BingX


def test_bingx_factory_ignores_proxy_environment(proxy_env: None) -> None:
    with build_bingx_http_client(LIVE_BASE_URL) as client:
        assert not uses_proxy(client, BINGX_URL)
        assert client.trust_env is False


@pytest.mark.parametrize("base_url", [LIVE_BASE_URL, VST_BASE_URL])
def test_public_client_ignores_proxy_environment(proxy_env: None, base_url: str) -> None:
    with BingXPublicClient(base_url) as client:
        assert not uses_proxy(client._http, BINGX_URL)


def test_readonly_client_ignores_proxy_environment(proxy_env: None) -> None:
    with BingXReadOnlyClient(SecretStr("k" * 8), SecretStr("s" * 8), VST_BASE_URL) as client:
        assert not uses_proxy(client._http, BINGX_URL)


def test_bingx_factory_keeps_headers_timeouts_and_transport() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, json={}))
    with build_bingx_http_client(LIVE_BASE_URL + "/", timeout=20, transport=transport) as client:
        assert str(client.base_url).rstrip("/") == LIVE_BASE_URL
        assert client.headers["User-Agent"].startswith("squeezelab/")
        assert client.timeout.read == 20
        assert client.timeout.connect == 5.0
        assert client.get("/ping").status_code == 200


# ----------------------------------------------------------------------------- Telegram


def test_telegram_ignores_proxy_environment_by_default(proxy_env: None) -> None:
    with build_telegram_http_client() as client:
        assert not uses_proxy(client, TELEGRAM_URL)
    with TelegramClient(SecretStr(TOKEN), [1]) as tg:
        assert not uses_proxy(tg._http, TELEGRAM_URL)


def test_telegram_explicit_proxy_is_used(proxy_env: None) -> None:
    with build_telegram_http_client(proxy="http://127.0.0.1:10809") as client:
        assert uses_proxy(client, TELEGRAM_URL)
    with TelegramClient(SecretStr(TOKEN), [1], proxy=SecretStr("http://127.0.0.1:10809")) as tg:
        assert uses_proxy(tg._http, TELEGRAM_URL)


def test_telegram_request_goes_through_explicit_proxy(
    proxy_env: None, fake_proxy: FakeProxy
) -> None:
    with (
        TelegramClient(SecretStr(TOKEN), [1], proxy=SecretStr(fake_proxy.url), timeout=3) as tg,
        pytest.raises(TelegramError) as exc,
    ):
        tg.send_message(1, "проверка прокси")
    fake_proxy.close()
    assert fake_proxy.first_lines == ["CONNECT api.telegram.org:443 HTTP/1.1"]
    assert TOKEN not in str(exc.value)


def test_telegram_mock_transport_still_works_without_proxy(proxy_env: None) -> None:
    transport = httpx.MockTransport(
        lambda _r: httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
    )
    with TelegramClient(SecretStr(TOKEN), [1], transport=transport) as tg:
        tg.send_message(1, "привет")
