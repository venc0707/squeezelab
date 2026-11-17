"""Фабрика HTTP-клиентов (httpx). Единственное место, где создаются `httpx.Client`.

Правила проекта (решение мастера, Phase 01):

- Клиенты BingX всегда создаются с `trust_env=False`: переменные HTTP_PROXY, HTTPS_PROXY,
  ALL_PROXY и NO_PROXY их не затрагивают. BingX блокирует часть регионов и VPN, поэтому
  запросы к нему идут напрямую.
- Telegram тоже с `trust_env=False`. Прокси для него задаётся только явным параметром
  `proxy` (в проекте из `TELEGRAM_PROXY_URL`), а не окружением процесса.

`trust_env=False` у httpx отключает не только прокси из окружения, но и чтение
SSL_CERT_FILE / SSL_CERT_DIR и ~/.netrc: клиенты используют встроенный набор сертификатов
certifi. Если когда-нибудь понадобится корпоративный корневой сертификат, его придётся
передавать явно, а не через окружение.
"""

from __future__ import annotations

import httpx

from squeezelab import __version__

_MAX_CONNECT_TIMEOUT_S = 5.0


def build_bingx_http_client(
    base_url: str,
    *,
    timeout: float = 10.0,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """HTTP-клиент для BingX: без прокси из окружения, только прямое соединение."""
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(timeout, connect=min(timeout, _MAX_CONNECT_TIMEOUT_S)),
        headers={"User-Agent": f"squeezelab/{__version__}"},
        transport=transport,
        trust_env=False,
    )


def build_telegram_http_client(
    *,
    timeout: float = 10.0,
    transport: httpx.BaseTransport | None = None,
    proxy: str | None = None,
) -> httpx.Client:
    """HTTP-клиент для Telegram: прокси только явный. Если задан `proxy`, httpx использует его
    вместо `transport` (в тестах с mock-транспортом прокси не задают)."""
    return httpx.Client(timeout=timeout, transport=transport, proxy=proxy, trust_env=False)
