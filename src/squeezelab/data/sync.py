"""Докачка пропусков и пометка «подтверждённо отсутствуют» (P01-T03).

Пропуск = слот сетки ТФ, в котором в хранилище нет свечи, хотя до и после неё свечи есть (или
слот лежит до последней закрытой свечи). Докачка просит именно недостающие слоты узкими окнами
`[первый, последний + ТФ)` по `SYNC_CHUNK_BARS` баров (далеко меньше наблюдаемого предела ответа
биржи 1000, поэтому окно не усекается; конец окна исключён, см. docs/bingx_api_notes.md).

Если после `SYNC_ATTEMPTS` попыток свеча так и не пришла, а она старше `CONFIRM_MIN_AGE_MS`, слот
помечается в реестре хранилища как «подтверждённо отсутствует» (у биржи его нет), и дальше его не
запрашивают бесконечно: проверка качества показывает такие слоты предупреждением, а не ошибкой.
Свежие слоты не подтверждаются: биржа могла ещё не дозаполнить свечу.

Допущение: окно в `SYNC_CHUNK_BARS` баров не усекается (предел ответа не меньше 100 свечей).
Если когда-нибудь окажется иначе, свечи из «хвоста» окна ошибочно попадут в реестр; поэтому
реестр хранится рядом с данными и снимается автоматически, когда свеча появляется.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from squeezelab.data.history import KlinesFetcher
from squeezelab.data.models import Candle, format_ts, is_candle_closed, tf_to_ms
from squeezelab.data.store import CandleStore, ConfirmedMissing, WriteStats

SYNC_ATTEMPTS = 3
SYNC_CHUNK_BARS = 100
CONFIRM_MIN_AGE_MS = 6 * 3_600_000
SYNC_RETRY_PAUSE_S = 1.0
_FLUSH_EVERY_RUNS = 20


class Waiter(Protocol):
    def wait(self) -> None: ...


@dataclass
class SyncReport:
    symbol: str
    tf: str
    start_ms: int
    end_ms: int  # исключительно
    gaps_found: int = 0  # недостающих слотов, которые пробовали докачать
    filled: int = 0
    confirmed_new: int = 0  # слотов, подтверждённых как отсутствующие в этом запуске
    already_confirmed: int = 0  # слотов диапазона, помеченных раньше (не запрашивались)
    pending: list[int] = field(default_factory=list)  # не докачаны и ещё не подтверждены
    requests: int = 0
    retries: int = 0
    last_retry_error: str = ""
    stats: WriteStats = field(default_factory=WriteStats)


def missing_slots(have: set[int], skip: set[int], start: int, end: int, tf_ms: int) -> list[int]:
    """Слоты сетки в `[start, end)`, которых нет ни в хранилище, ни среди `skip`."""
    return [ts for ts in range(start, end, tf_ms) if ts not in have and ts not in skip]


def split_runs(slots: list[int], tf_ms: int, max_bars: int) -> list[tuple[int, int]]:
    """Группы подряд идущих слотов `(первый, последний)`, не длиннее `max_bars` баров."""
    runs: list[tuple[int, int]] = []
    for ts in slots:
        if runs and ts - runs[-1][1] == tf_ms and (ts - runs[-1][0]) // tf_ms < max_bars:
            runs[-1] = (runs[-1][0], ts)
        else:
            runs.append((ts, ts))
    return runs


def _align_down(value: int, step: int) -> int:
    return value - value % step


def _align_up(value: int, step: int) -> int:
    return -(-value // step) * step


def sync_gaps(
    client: KlinesFetcher,
    store: CandleStore,
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
    *,
    now_ms: int,
    limiter: Waiter,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] | None = None,
) -> SyncReport:
    """Докачивает пропуски в `[start_ms, end_ms)` и подтверждает те, которых у биржи нет."""
    tf_ms = tf_to_ms(tf)
    start = _align_up(start_ms, tf_ms)
    end = min(_align_down(end_ms, tf_ms), _align_down(now_ms, tf_ms))
    report = SyncReport(symbol, tf, start, max(start, end))
    if end <= start:
        return report

    have = store.timestamps(symbol, tf)
    confirmed = set(store.confirmed_missing(symbol, tf))
    report.already_confirmed = sum(1 for ts in confirmed if start <= ts < end and ts not in have)
    missing = missing_slots(have, confirmed, start, end, tf_ms)
    report.gaps_found = len(missing)
    if not missing:
        return report

    found: dict[int, Candle] = {}
    pending_write: list[Candle] = []

    def flush() -> None:
        if pending_write:
            report.stats += store.write(symbol, tf, pending_write)
            pending_write.clear()

    runs = split_runs(missing, tf_ms, SYNC_CHUNK_BARS)
    try:
        for index, (first, last) in enumerate(runs, start=1):
            expected = set(range(first, last + tf_ms, tf_ms))
            got: dict[int, Candle] = {}
            for attempt in range(SYNC_ATTEMPTS):
                if attempt:
                    sleep(SYNC_RETRY_PAUSE_S)
                limiter.wait()
                response = client.fetch_klines(
                    symbol, tf, SYNC_CHUNK_BARS, start_time=first, end_time=last + tf_ms
                )
                report.requests += 1
                for candle in response.candles:
                    if candle.ts_open in expected and is_candle_closed(candle, tf_ms, now_ms):
                        got[candle.ts_open] = candle
                if expected <= got.keys():
                    break
            found.update(got)
            pending_write.extend(got.values())
            if index % _FLUSH_EVERY_RUNS == 0:
                flush()
                if progress is not None:
                    progress(f"  {symbol} {tf}: докачка {index} из {len(runs)} групп пропусков")
    finally:
        flush()

    report.filled = len(found)
    remaining = [ts for ts in missing if ts not in found]
    confirm = [ts for ts in remaining if now_ms - (ts + tf_ms) >= CONFIRM_MIN_AGE_MS]
    if confirm:
        report.confirmed_new = store.add_confirmed_missing(
            symbol, tf, (ConfirmedMissing(ts, SYNC_ATTEMPTS, now_ms) for ts in confirm)
        )
    report.pending = [ts for ts in remaining if ts not in set(confirm)]
    return report


def render_sync(report: SyncReport) -> list[str]:
    head = (
        f"{report.symbol} {report.tf}: докачка пропусков в "
        f"{format_ts(report.start_ms)} .. {format_ts(report.end_ms)} (конец исключён)"
    )
    if report.gaps_found == 0:
        lines = [head, "  пропусков нет"]
        if report.already_confirmed:
            lines.append(
                f"  ранее подтверждённо отсутствуют у биржи: {report.already_confirmed} "
                "(не запрашиваются)"
            )
        return lines
    lines = [
        head,
        f"  найдено пропусков: {report.gaps_found}; докачано: {report.filled}; подтверждённо "
        f"отсутствуют (новых): {report.confirmed_new}; ждут подтверждения: {len(report.pending)}; "
        f"запросов: {report.requests}",
    ]
    if report.already_confirmed:
        lines.append(
            f"  ранее подтверждённо отсутствуют: {report.already_confirmed} (не запрашивались)"
        )
    if report.pending:
        hours = CONFIRM_MIN_AGE_MS // 3_600_000
        lines.append(
            f"  ВНИМАНИЕ: {len(report.pending)} пропусков моложе {hours} ч, "
            "они не подтверждены и будут запрошены снова"
        )
    if report.retries:
        lines.append(
            f"  повторов запросов после временных ошибок: {report.retries} "
            f"(последняя ошибка: {report.last_retry_error})"
        )
    return lines
