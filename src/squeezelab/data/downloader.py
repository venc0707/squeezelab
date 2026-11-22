"""Загрузчик исторических свечей BingX в Parquet (P01-T02).

Пагинация по времени окнами `[start, end)`. Размер окна выбран по наблюдениям
`bingx-probe-klines` (docs/bingx_api_notes.md, раздел 4):

- при `startTime` и `endTime` свеча берётся, если `start <= ts_open < end` (конец исключён);
- окно шире 1000 баров биржа молча усекает до 1000 самых новых, хотя `limit` до 1440 принимает
  (для 1m предел проверяет второй набор пробы, `--set paging`; размер окна настраивается);
- порядок ответа «новые первыми»; формирующаяся свеча при запросе до «сейчас» попадает в выдачу.

Поэтому окно не больше `WINDOW_BARS = 1000` баров и не усекается, соседние окна
стыкуются без пропусков и пересечений, а полнота каждого окна проверяется по числу и меткам
времени (ничего не принимается на веру). Повторный запуск пропускает окна, которые уже целиком
лежат в хранилище, и докачивает остальные.

Если `start` раньше начала истории пары, биржа отдаёт на такой запрос пустой ответ и теряются
даже те свечи окна, которые у неё есть. Поэтому, когда на `start` и позже свечей нет, начало
истории ищется бисекцией (`data/history.py`), и период начинается с самой ранней свечи.

Закрытость свечи определяется дважды: по часам этой машины и по самой новой свече, которую
отдаёт биржа (она формирующаяся). Берётся более строгая граница, поэтому сбитые часы не дадут
записать недозакрытую свечу.

Временные сбои (сеть, ответ с ошибкой биржи, неразобранный ответ) не обрывают загрузку: каждый
запрос повторяется до трёх раз с паузами 2, 5 и 10 секунд, затем ошибка пробрасывается, а
накопленное остаётся на диске.

Пропуски, которых у биржи нет, здесь только фиксируются в отчёте после нескольких попыток.
Пометку «подтверждённо отсутствует» и команды проверки добавляет P01-T03.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from squeezelab.data.bingx_rest import BingXError, BingXPublicClient, KlinesResponse
from squeezelab.data.history import KlinesFetcher, find_history_start
from squeezelab.data.models import Candle, format_ts, is_candle_closed, tf_to_ms
from squeezelab.data.store import CandleStore, WriteStats

WINDOW_BARS = 1000  # не больше наблюдаемого предела ответа биржи
WINDOW_ATTEMPTS = 3  # попыток на окно, если в ответе не хватает свечей
FLUSH_EVERY_WINDOWS = 10  # как часто сбрасывать накопленное на диск
MAX_REQUESTS_PER_SECOND = 5.0  # требование задания; лимит биржи 500 за 10 с
DEFAULT_START = datetime(2026, 6, 25, tzinfo=UTC)  # прогрев 500 баров до 2026-07-01
RETRY_DELAYS_S = (2.0, 5.0, 10.0)  # паузы перед повторами запроса после ошибки
_MISSING_PREVIEW = 10


class RateLimiter:
    """Не пускает запросы чаще `max_per_second` (равномерные паузы)."""

    def __init__(
        self,
        max_per_second: float = MAX_REQUESTS_PER_SECOND,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second должен быть положительным")
        self._interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._next_at: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._next_at is None or now >= self._next_at:
            self._next_at = now + self._interval
        else:
            self._sleep(self._next_at - now)
            self._next_at += self._interval


@dataclass(slots=True)
class DownloadReport:
    symbol: str
    tf: str
    start_ms: int  # первая свеча периода (выровнена вверх по границе ТФ)
    end_ms: int  # конец периода, исключительно (граница ТФ, не позже формирующейся свечи)
    expected: int = 0  # сколько закрытых свечей должно быть в периоде
    windows: int = 0
    windows_skipped: int = 0  # окна, которые уже целиком были в хранилище
    requests: int = 0
    accepted: int = 0  # свечей принято из ответов (закрытые, внутри окна)
    dropped_unclosed: int = 0
    dropped_out_of_range: int = 0
    window_bars: int = WINDOW_BARS
    retries: int = 0  # повторов запросов после временных ошибок
    last_retry_error: str = ""
    requested_start_ms: int = 0  # начало, как его запросили (выровнено вверх по сетке ТФ)
    start_clamped: bool = False  # начало сдвинуто на начало истории пары
    history_conflict_ts: int | None = None  # бисекция дала это начало, но раньше свечи есть
    no_data: bool = False  # у биржи нет свечей в запрошенном периоде
    exchange_newest_ms: int | None = (
        None  # ts_open самой новой свечи в ответе биржи (формирующейся)
    )
    end_cut_by_exchange: bool = False  # конец периода урезан по свече биржи (часы машины спешат)
    stored: int = 0  # свечей периода в хранилище после загрузки
    stats: WriteStats = field(default_factory=WriteStats)
    missing: list[int] = field(default_factory=list)  # ts_open, которых нет нигде


def align_up(value_ms: int, step_ms: int) -> int:
    return -(-value_ms // step_ms) * step_ms


def align_down(value_ms: int, step_ms: int) -> int:
    return value_ms - value_ms % step_ms


def plan_windows(
    start_ms: int, end_ms: int, tf_ms: int, window_bars: int = WINDOW_BARS
) -> list[tuple[int, int]]:
    """Окна `[start, end)` по `window_bars` баров, без пропусков и пересечений.

    `start_ms` и `end_ms` должны лежать на границе ТФ; последнее окно может быть короче.
    """
    if window_bars < 1:
        raise ValueError("window_bars должен быть не меньше 1")
    if start_ms % tf_ms or end_ms % tf_ms:
        raise ValueError("границы периода должны лежать на границе таймфрейма")
    windows: list[tuple[int, int]] = []
    cursor = start_ms
    while cursor < end_ms:
        windows.append((cursor, min(cursor + window_bars * tf_ms, end_ms)))
        cursor = windows[-1][1]
    return windows


class _RetryingClient:
    """Повторяет `fetch_klines` после ошибок биржи и считает повторы в отчёте."""

    def __init__(
        self,
        client: BingXPublicClient,
        sleep: Callable[[float], None],
        report: DownloadReport,
    ) -> None:
        self._client = client
        self._sleep = sleep
        self._report = report

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 20,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> KlinesResponse:
        for delay in (*RETRY_DELAYS_S, None):
            try:
                return self._client.fetch_klines(
                    symbol, interval, limit, start_time=start_time, end_time=end_time
                )
            except BingXError as exc:
                if delay is None:
                    raise
                self._report.retries += 1
                self._report.last_retry_error = f"{type(exc).__name__}: {exc}"
                self._sleep(delay)
        raise AssertionError("недостижимо")  # pragma: no cover


def _fetch_window(
    client: KlinesFetcher,
    symbol: str,
    tf: str,
    tf_ms: int,
    window: tuple[int, int],
    now_ms: int,
    limiter: RateLimiter,
    report: DownloadReport,
    window_bars: int,
) -> dict[int, Candle]:
    start, end = window
    expected = set(range(start, end, tf_ms))
    got: dict[int, Candle] = {}
    for _ in range(WINDOW_ATTEMPTS):
        limiter.wait()
        response = client.fetch_klines(symbol, tf, window_bars, start_time=start, end_time=end)
        report.requests += 1
        for candle in response.candles:
            if not is_candle_closed(candle, tf_ms, now_ms):
                report.dropped_unclosed += 1
            elif candle.ts_open not in expected:
                report.dropped_out_of_range += 1
            else:
                got[candle.ts_open] = candle
        if expected <= got.keys():
            break
    return got


def download_tf(
    client: BingXPublicClient,
    store: CandleStore,
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
    *,
    now_ms: int,
    limiter: RateLimiter,
    progress: Callable[[str], None] | None = None,
    window_bars: int = WINDOW_BARS,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadReport:
    """Докачивает свечи `[start_ms, end_ms)` одной пары и одного ТФ.

    Период выравнивается: начало вверх, конец вниз по границе ТФ, и не заходит на формирующуюся
    свечу (ни по часам машины, ни по самой новой свече биржи). Накопленное пишется на диск
    каждые несколько окон и при любом выходе (в том числе по ошибке API), поэтому прогресс не
    теряется, а повторный запуск продолжает с места остановки.
    """
    tf_ms = tf_to_ms(tf)
    start = align_up(start_ms, tf_ms)
    end = min(align_down(end_ms, tf_ms), align_down(now_ms, tf_ms))
    report = DownloadReport(
        symbol, tf, start, max(start, end), window_bars=window_bars, requested_start_ms=start
    )
    if end <= start:
        return report
    fetcher = _RetryingClient(client, sleep, report)
    # Граница по бирже: самая новая свеча в выдаче формирующаяся, закрытые лежат строго раньше.
    limiter.wait()
    newest = fetcher.fetch_klines(symbol, tf, 1).candles
    report.requests += 1
    if newest:
        report.exchange_newest_ms = max(candle.ts_open for candle in newest)
        exchange_end = align_down(report.exchange_newest_ms, tf_ms)
        if exchange_end < end:
            end = exchange_end
            report.end_cut_by_exchange = True
            report.end_ms = max(start, end)
            if end <= start:
                return report
    existing = store.timestamps(symbol, tf)
    if start not in existing:
        # Есть ли у биржи свечи не раньше start? Если нет, start раньше начала истории.
        limiter.wait()
        probe = fetcher.fetch_klines(symbol, tf, 1, start_time=start).candles
        report.requests += 1
        if not probe:
            history = find_history_start(fetcher, symbol, tf, start, end - tf_ms, wait=limiter.wait)
            report.requests += history.requests
            if history.first_ts is None:
                report.no_data = True
                report.start_ms = report.end_ms = start
                return report
            if history.first_ts > start:
                # Проверка окном (его семантика однозначна): бисекция предполагает сплошной ряд,
                # а при пропусках в данных биржи могла остановиться посреди истории.
                limiter.wait()
                earlier = fetcher.fetch_klines(
                    symbol, tf, 1, start_time=start, end_time=history.first_ts
                ).candles
                report.requests += 1
                if earlier:
                    report.history_conflict_ts = history.first_ts
                else:
                    start = history.first_ts
                    report.start_ms = start
                    report.start_clamped = True
                    if end <= start:
                        report.end_ms = start
                        return report
    report.expected = (end - start) // tf_ms
    windows = plan_windows(start, end, tf_ms, window_bars)
    report.windows = len(windows)

    pending: list[Candle] = []

    def flush() -> None:
        if pending:
            report.stats += store.write(symbol, tf, pending)
            existing.update(candle.ts_open for candle in pending)
            pending.clear()

    since_flush = 0
    try:
        for index, window in enumerate(windows, start=1):
            expected = range(window[0], window[1], tf_ms)
            if all(ts in existing for ts in expected):
                report.windows_skipped += 1
                continue
            got = _fetch_window(
                fetcher, symbol, tf, tf_ms, window, now_ms, limiter, report, window_bars
            )
            pending.extend(got.values())
            report.accepted += len(got)
            report.missing.extend(ts for ts in expected if ts not in got and ts not in existing)
            since_flush += 1
            if since_flush >= FLUSH_EVERY_WINDOWS:
                flush()
                since_flush = 0
            if progress is not None and (index % FLUSH_EVERY_WINDOWS == 0 or index == len(windows)):
                progress(f"  {symbol} {tf}: окно {index} из {len(windows)}")
    finally:
        flush()
        report.stored = sum(1 for ts in existing if start <= ts < end)
    return report


def render_report(report: DownloadReport) -> list[str]:
    if report.no_data:
        return [
            f"{report.symbol} {report.tf}: у биржи нет свечей в периоде с "
            f"{format_ts(report.requested_start_ms)}: нечего загружать"
        ]
    if report.expected == 0:
        return [f"{report.symbol} {report.tf}: нечего загружать (период пуст или ещё не закрыт)"]
    lines = [
        f"{report.symbol} {report.tf}: {format_ts(report.start_ms)} .. {format_ts(report.end_ms)} "
        f"(конец исключён); ожидается свечей: {report.expected}",
        f"  окон: {report.windows} по {report.window_bars} баров (уже полных в хранилище: "
        f"{report.windows_skipped}), запросов: {report.requests}",
        f"  принято из ответов: {report.accepted}; отброшено формирующихся: "
        f"{report.dropped_unclosed}, вне окна: {report.dropped_out_of_range}",
        f"  запись: новых {report.stats.added}, обновлено {report.stats.updated}, "
        f"без изменений {report.stats.unchanged}, файлов записано {report.stats.files_written}",
        f"  в хранилище за период: {report.stored} из {report.expected}",
    ]
    if report.retries:
        lines.append(
            f"  повторов запросов после временных ошибок: {report.retries} "
            f"(последняя ошибка: {report.last_retry_error})"
        )
    if report.history_conflict_ts is not None:
        lines.append(
            "  ВНИМАНИЕ: поиск по startTime указал начало "
            f"{format_ts(report.history_conflict_ts)}, но раньше в окне свечи есть; "
            "период НЕ урезан, загрузка идёт с запрошенного начала"
        )
    if report.start_clamped:
        lines.append(
            f"  ВНИМАНИЕ: у биржи нет свечей раньше {format_ts(report.start_ms)}; "
            f"период начат с неё (запрошено с {format_ts(report.requested_start_ms)})"
        )
    if report.end_cut_by_exchange and report.exchange_newest_ms is not None:
        lines.append(
            "  ВНИМАНИЕ: конец периода урезан по свече биржи "
            f"({format_ts(report.exchange_newest_ms)}): часы этой машины спешат относительно биржи"
        )
    if report.missing:
        shown = ", ".join(format_ts(ts) for ts in report.missing[:_MISSING_PREVIEW])
        more = len(report.missing) - _MISSING_PREVIEW
        suffix = f" … и ещё {more}" if more > 0 else ""
        lines.append(f"  ПРОПУСКИ ({len(report.missing)}): {shown}{suffix}")
    else:
        lines.append("  пропусков нет")
    return lines
