"""REST-клиент BingX. Здесь только GET-запросы: ретраи безопасны (повтор ничего не меняет).

Публичные данные (свечи) не требуют ключей. Торговых эндпоинтов и ордеров в проекте пока нет.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

from squeezelab import __version__
from squeezelab.data.models import Candle

LIVE_BASE_URL = "https://open-api.bingx.com"
VST_BASE_URL = "https://open-api-vst.bingx.com"
KLINES_PATH = "/openApi/swap/v3/quote/klines"

# Интервалы из документации BingX (месяц «1M» не используем).
VALID_INTERVALS = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_DELAY_S = 30.0
_MAX_RETRY_AFTER_S = 60.0
_REQUIRED_KLINE_FIELDS = ("time", "open", "high", "low", "close", "volume")


class BingXError(Exception):
    """Базовая ошибка клиента BingX. Тексты не содержат секретов."""


class BingXRequestError(BingXError):
    """Сеть недоступна или попытки исчерпаны."""


class BingXHTTPError(BingXError):
    def __init__(self, status: int, excerpt: str) -> None:
        super().__init__(f"HTTP {status}: {excerpt}")
        self.status = status


class BingXAPIError(BingXError):
    def __init__(self, code: object, msg: str) -> None:
        super().__init__(f"ошибка API BingX: code={code}, msg={msg!r}")
        self.code = code
        self.msg = msg


class BingXParseError(BingXError):
    """Ответ получен, но не соответствует ожидаемому формату."""


@dataclass(frozen=True, slots=True)
class KlinesResponse:
    candles: list[Candle]  # в том порядке, в котором их вернула биржа
    raw_text: str
    limit_headers: dict[str, str]


def _number(value: object, field: str, index: int) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise BingXParseError(f"свеча #{index}: поле {field!r} не число: {value!r}") from None
    if not math.isfinite(number):
        raise BingXParseError(f"свеча #{index}: поле {field!r} не конечное число: {value!r}")
    return number


def parse_klines_data(data: object) -> list[Candle]:
    """Разбирает `data` ответа klines: список объектов с полями time/open/high/low/close/volume.

    Порядок сохраняется как в ответе. Формат подтверждён только документацией (см.
    docs/bingx_api_notes.md); на неизвестную форму парсер падает, а не угадывает.
    """
    if not isinstance(data, list):
        raise BingXParseError(f"data должен быть списком, получено {type(data).__name__}")
    candles: list[Candle] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise BingXParseError(
                f"свеча #{index}: ожидался объект, получено {type(item).__name__}"
            )
        missing = [name for name in _REQUIRED_KLINE_FIELDS if name not in item]
        if missing:
            raise BingXParseError(f"свеча #{index}: нет полей {missing}")
        ts = _number(item["time"], "time", index)
        if ts < 0 or ts != int(ts):
            raise BingXParseError(
                f"свеча #{index}: time должен быть целым числом мс: {item['time']!r}"
            )
        candles.append(
            Candle(
                ts_open=int(ts),
                o=_number(item["open"], "open", index),
                h=_number(item["high"], "high", index),
                l=_number(item["low"], "low", index),
                c=_number(item["close"], "close", index),
                v=_number(item["volume"], "volume", index),
            )
        )
    return candles


def unwrap_envelope(payload: object) -> Any:
    """Проверяет конверт {"code": 0, "msg": "...", "data": ...} и возвращает data."""
    if not isinstance(payload, dict) or "code" not in payload:
        raise BingXParseError("ответ не содержит поля code: это не конверт BingX")
    if payload["code"] != 0:
        raise BingXAPIError(payload["code"], str(payload.get("msg", "")))
    if "data" not in payload:
        raise BingXParseError("в ответе code=0, но нет поля data")
    return payload["data"]


def _excerpt(text: str, limit: int = 200) -> str:
    one_line = " ".join(text.split())
    return one_line[:limit] + ("…" if len(one_line) > limit else "")


def _limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if "limit" in k.lower() or "retry" in k.lower()}


class BingXPublicClient:
    def __init__(
        self,
        base_url: str = LIVE_BASE_URL,
        *,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries не может быть отрицательным")
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 5.0)),
            headers={"User-Agent": f"squeezelab/{__version__}"},
            transport=transport,
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

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after", "")
            if retry_after.isdigit():
                return min(float(retry_after), _MAX_RETRY_AFTER_S)
        return min(self._backoff_base * (2.0**attempt), _MAX_DELAY_S)

    def get(self, path: str, params: Mapping[str, str | int]) -> httpx.Response:
        """GET с ретраями (сеть, 429, 5xx) и нарастающей паузой. Не-GET методов нет намеренно."""
        last_error: str = ""
        for attempt in range(self._max_retries + 1):
            is_last = attempt == self._max_retries
            try:
                response = self._http.get(path, params=dict(params))
            except httpx.TransportError as exc:
                last_error = type(exc).__name__
                if is_last:
                    break
                self._sleep(self._delay(attempt, None))
                continue
            if response.status_code in RETRY_STATUSES and not is_last:
                self._sleep(self._delay(attempt, response))
                continue
            return response
        raise BingXRequestError(
            f"сеть недоступна: {self._max_retries + 1} попыток не удались ({last_error}); "
            "проверь интернет, DNS и доступность open-api.bingx.com"
        )

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 20,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> KlinesResponse:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"интервал {interval!r} не из списка {sorted(VALID_INTERVALS)}")
        if limit < 1:
            raise ValueError("limit должен быть не меньше 1")
        params: dict[str, str | int] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        response = self.get(KLINES_PATH, params)
        if response.status_code != 200:
            raise BingXHTTPError(response.status_code, _excerpt(response.text))
        try:
            payload = json.loads(response.text)
        except ValueError:
            raise BingXParseError(f"ответ не JSON: {_excerpt(response.text)}") from None
        candles = parse_klines_data(unwrap_envelope(payload))
        return KlinesResponse(candles, response.text, _limit_headers(response.headers))

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 20,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Candle]:
        return self.fetch_klines(
            symbol, interval, limit, start_time=start_time, end_time=end_time
        ).candles
