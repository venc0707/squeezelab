"""Поиск начала истории пары на BingX (P01-T02, переиспользуется в P01-T06).

Наблюдение пробы 2026-11-20/21 (docs/bingx_api_notes.md, раздел 4): запрос только с `startTime`
отдаёт самые ранние свечи от start, но если start раньше начала истории (или раньше окна, которое
биржа хранит), ответ ПУСТОЙ, а не «самые ранние свечи». Поэтому начало истории ищется бисекцией
по `startTime`, выровненному по сетке ТФ: пусто = раньше начала, есть свеча = не раньше.

Предположение бисекции: «есть свеча» монотонно по start. Если оно нарушится (случайные пустые
ответы), результат всё равно проверяемый: возвращаются все шаги `(start, есть ли свеча)`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from squeezelab.data.bingx_rest import KlinesResponse
from squeezelab.data.models import tf_to_ms

Reason = Literal["found", "deeper_than_low", "no_data"]


class KlinesFetcher(Protocol):
    """Всё, что нужно поиску от клиента: `BingXPublicClient` или обёртка с повторами."""

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = ...,
        *,
        start_time: int | None = ...,
        end_time: int | None = ...,
    ) -> KlinesResponse: ...


@dataclass(frozen=True, slots=True)
class HistoryStart:
    interval: str
    first_ts: int | None  # ts_open самой ранней свечи, которую удалось получить
    reason: Reason  # found; deeper_than_low = свеча есть уже при low; no_data = нет и при high
    steps: tuple[tuple[int, bool], ...]  # (start, пришла ли свеча) в порядке запросов

    @property
    def requests(self) -> int:
        return len(self.steps)


def _align_down(value_ms: int, step_ms: int) -> int:
    return value_ms - value_ms % step_ms


def find_history_start(
    client: KlinesFetcher,
    symbol: str,
    interval: str,
    low_ms: int,
    high_ms: int,
    *,
    wait: Callable[[], None] = lambda: None,
) -> HistoryStart:
    """Самая ранняя свеча в `[low_ms, high_ms]` (границы выравниваются вниз по сетке ТФ).

    Ожидается, что при `low_ms` ответ пуст, а при `high_ms` в нём есть свеча; иначе результат
    помечается `deeper_than_low` или `no_data`. Ошибки API не перехватываются.
    """
    tf_ms = tf_to_ms(interval)
    lo = _align_down(low_ms, tf_ms)
    hi = _align_down(high_ms, tf_ms)
    steps: list[tuple[int, bool]] = []

    def first_from(start: int) -> int | None:
        wait()
        candles = client.fetch_klines(symbol, interval, 1, start_time=start).candles
        found = min((c.ts_open for c in candles), default=None)
        steps.append((start, found is not None))
        return found

    if hi < lo:
        raise ValueError("high_ms раньше low_ms")
    at_hi = first_from(hi)
    if at_hi is None:
        return HistoryStart(interval, None, "no_data", tuple(steps))
    at_lo = first_from(lo)
    if at_lo is not None:
        return HistoryStart(interval, at_lo, "deeper_than_low", tuple(steps))
    best = at_hi
    while hi - lo > tf_ms:
        mid = lo + ((hi - lo) // tf_ms // 2) * tf_ms
        found = first_from(mid)
        if found is None:
            lo = mid
        else:
            hi = mid
            best = found
    return HistoryStart(interval, best, "found", tuple(steps))
