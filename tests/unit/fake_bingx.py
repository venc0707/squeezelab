"""Фальшивый сервер klines BingX для тестов (без сети).

По умолчанию воспроизводит поведение, НАБЛЮДАВШЕЕСЯ на настоящем BingX в пробе 2026-11-20
(BNB-USDT, 15m и 1m; `bingx-probe-klines`):

- окно startTime+endTime: свеча по времени открытия, `start` включительно, `end` исключительно;
- в ответ попадает не больше 1000 свечей (при `limit` до 1440), а если в окне их больше,
  остаются САМЫЕ НОВЫЕ (ранние теряются);
- только `startTime`: самые ранние свечи от start; только `endTime`: самые новые, end включительно;
- без времён: самые новые свечи, включая формирующуюся;
- порядок в ответе: новые первыми;
- `startTime` далеко до листинга (2019 год): пустой ответ.

Остальные варианты (`Behaviour`) нужны, чтобы проверить, что код не ломается молча, если биржа
изменит поведение.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx

from squeezelab.data.models import tf_to_ms


@dataclass(frozen=True)
class Behaviour:
    cap: int = 1000
    start_inclusive: bool = True
    end_exclusive_with_start: bool = True
    end_inclusive_alone: bool = True
    start_only: Literal["earliest", "latest"] = "earliest"
    windowed_truncate: Literal["latest", "earliest"] = "latest"
    order: Literal["desc", "asc"] = "desc"
    history_start: int = 0  # самая ранняя свеча биржи (ts_open, мс); 0 = с начала эпохи
    holes: frozenset[int] = frozenset()  # свечи, которых у биржи нет (ts_open)
    offgrid: bool = False  # добавлять в ответ свечу не по сетке (аномалия)
    start_before_history_is_empty: bool = True  # start до листинга даёт пустой ответ


def bar_values(ts_open: int, tf_ms: int) -> dict[str, object]:
    """Детерминированная свеча: значения зависят только от времени (для сверки после записи)."""
    index = ts_open // tf_ms
    base = 100.0 + (index % 97) / 10
    return {
        "open": f"{base:.2f}",
        "close": f"{base + 0.5:.2f}",
        "high": f"{base + 1.0:.2f}",
        "low": f"{base - 1.0:.2f}",
        "volume": f"{10 + index % 13}.5",
        "time": ts_open,
    }


@dataclass
class FakeBingX:
    behaviour: Behaviour = field(default_factory=Behaviour)
    now_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    fail_after: int | None = None  # после стольких успешных запросов отвечать ошибкой биржи
    requests: list[dict[str, int | str]] = field(default_factory=list)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def data_requests(self) -> list[dict[str, int | str]]:
        """Запросы с обоими временами (то есть сами загрузочные окна)."""
        return [r for r in self.requests if "startTime" in r and "endTime" in r]

    def forming_open(self, tf: str) -> int:
        tf_ms = tf_to_ms(tf)
        return self.now_ms - self.now_ms % tf_ms

    # ------------------------------------------------------------------------------
    def _handle(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.requests.append(
            {k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in params.items()}
        )
        if self.fail_after is not None and len(self.requests) > self.fail_after:
            return httpx.Response(200, json={"code": 100500, "msg": "internal error (fake)"})
        tf_ms = tf_to_ms(params["interval"])
        limit = int(params["limit"])
        if limit > 1440:
            return httpx.Response(
                200,
                json={
                    "code": 109400,
                    "msg": "limit: This field must be less than or equal to 1440.",
                },
            )
        start = int(params["startTime"]) if "startTime" in params else None
        end = int(params["endTime"]) if "endTime" in params else None
        opens = self._select(tf_ms, min(limit, self.behaviour.cap), start, end)
        if self.behaviour.order == "desc":
            opens.reverse()
        data = [bar_values(ts, tf_ms) for ts in opens]
        if self.behaviour.offgrid and data:
            odd = dict(data[0])
            odd["time"] = int(odd["time"]) + 1234  # type: ignore[call-overload]
            data.append(odd)
        return httpx.Response(200, text=json.dumps({"code": 0, "msg": "", "data": data}))

    def _exists(self, ts: int, newest: int) -> bool:
        return self.behaviour.history_start <= ts <= newest and ts not in self.behaviour.holes

    def _select(self, tf_ms: int, n: int, start: int | None, end: int | None) -> list[int]:
        b = self.behaviour
        newest = self.now_ms - self.now_ms % tf_ms
        if start is not None and start < b.history_start and b.start_before_history_is_empty:
            return []
        first = b.history_start
        if start is not None:
            first = max(first, -(-start // tf_ms) * tf_ms)
            if not b.start_inclusive and first == start:
                first += tf_ms
        last = newest
        if end is not None:
            exclusive = (
                b.end_exclusive_with_start if start is not None else not b.end_inclusive_alone
            )
            last = min(newest, end - end % tf_ms)
            if exclusive and end % tf_ms == 0:
                last -= tf_ms
        if start is not None and end is None:
            keep = b.start_only
        elif start is not None:
            keep = b.windowed_truncate
        else:
            keep = "latest"

        picked: list[int] = []
        if keep == "earliest":
            ts = first
            while ts <= last and len(picked) < n:
                if self._exists(ts, newest):
                    picked.append(ts)
                ts += tf_ms
        else:
            ts = last
            while ts >= first and len(picked) < n:
                if self._exists(ts, newest):
                    picked.append(ts)
                ts -= tf_ms
            picked.reverse()
        return picked
