"""Пробные запросы klines для P01-T02: что биржа делает с startTime, endTime и limit.

Загрузчик свечей строится на ответах на эти вопросы, а факты об API проекта подтверждаются
реальными запросами (правило 3 задания). Модуль делает ~14 небольших публичных запросов
(без ключей), печатает сырые наблюдения и выводы из них. Выводы относятся только к этим запросам.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise

from squeezelab.data.bingx_rest import BingXError, BingXPublicClient
from squeezelab.data.history import HistoryStart, find_history_start
from squeezelab.data.models import Candle, detect_order, format_ts, is_candle_closed, tf_to_ms

DEFAULT_ANCHOR = datetime(2026, 11, 1, tzinfo=UTC)  # граница любого ТФ до 1d включительно
DEFAULT_VOLUME_BAR = datetime(2026, 11, 13, 9, 30, tzinfo=UTC)  # volume=167.58 по заданию
FAR_PAST = datetime(2019, 1, 1, tzinfo=UTC)
PAUSE_S = 0.3  # пауза между пробными запросами (лимит биржи 500 за 10 с, мы далеко от него)
MAX_LIMIT = 1440

_PREVIEW_ALL = 8  # до стольких меток времени печатаем целиком, иначе три первые и три последние


@dataclass(frozen=True, slots=True)
class ProbeCase:
    key: str
    title: str
    interval: str
    limit: int
    start: int | None = None
    end: int | None = None
    pick_raw_ts: int | None = None  # напечатать сырой объект свечи с этим временем


@dataclass(frozen=True, slots=True)
class ProbeResult:
    case: ProbeCase
    candles: list[Candle] | None  # None = запрос не удался
    error: str | None = None
    raw_item: str | None = None


def to_ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def build_cases(anchor_ms: int, volume_bar_ms: int) -> list[ProbeCase]:
    """Набор пробных запросов. Ключи A…K используются в выводах."""
    tf = tf_to_ms("15m")
    tf1 = tf_to_ms("1m")
    t = anchor_ms
    return [
        ProbeCase("A", "окно из 5 баров по границам свечи", "15m", MAX_LIMIT, t, t + 4 * tf),
        ProbeCase("B", "start на 1 мс раньше свечи", "15m", MAX_LIMIT, t - 1, t + 4 * tf - 1),
        ProbeCase("C", "start на 1 мс позже начала свечи", "15m", MAX_LIMIT, t + 1, t + 4 * tf),
        ProbeCase("D", "только startTime, limit 5", "15m", 5, start=t),
        ProbeCase("E", "только endTime, limit 5", "15m", 5, end=t + 100 * tf),
        ProbeCase("F", "без времён, limit 3 (самые новые)", "15m", 3),
        ProbeCase(
            "G", "окно шире limit (1500 баров, limit 1440)", "15m", MAX_LIMIT, t, t + 1500 * tf
        ),
        ProbeCase("H", "1m: окно из 10 минут", "1m", MAX_LIMIT, t, t + 9 * tf1),
        ProbeCase("I1", "соседнее окно 1 (баров 0..99)", "15m", MAX_LIMIT, t, t + 99 * tf),
        ProbeCase(
            "I2", "соседнее окно 2 (баров 100..199)", "15m", MAX_LIMIT, t + 100 * tf, t + 199 * tf
        ),
        ProbeCase(
            "J",
            "свеча для сверки volume с TradingView",
            "15m",
            MAX_LIMIT,
            volume_bar_ms,
            volume_bar_ms + 2 * tf,
            pick_raw_ts=volume_bar_ms,
        ),
        ProbeCase("K", "start далеко в прошлом, limit 3", "15m", 3, start=to_ms(FAR_PAST)),
    ]


def _pick_raw(raw_text: str, ts: int | None) -> str | None:
    if ts is None:
        return None
    try:
        data = json.loads(raw_text)["data"]
        for item in data:
            if int(item["time"]) == ts:
                return json.dumps(item, ensure_ascii=False)
    except (ValueError, KeyError, TypeError):
        return None
    return None


def run_probe(
    client: BingXPublicClient,
    symbol: str,
    cases: list[ProbeCase],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for index, case in enumerate(cases):
        if index:
            sleep(PAUSE_S)
        try:
            response = client.fetch_klines(
                symbol, case.interval, case.limit, start_time=case.start, end_time=case.end
            )
        except BingXError as exc:
            results.append(ProbeResult(case, None, error=str(exc)))
            continue
        raw_item = _pick_raw(response.raw_text, case.pick_raw_ts)
        results.append(ProbeResult(case, response.candles, raw_item=raw_item))
    return results


# --------------------------------------------------------------------------------- вывод


def _short(ts_ms: int) -> str:
    """ГГГГ-ММ-ДД ЧЧ:ММ UTC (секунды не нужны: границы свечей лежат на целых минутах)."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _describe_params(case: ProbeCase) -> str:
    parts = [f"interval={case.interval}", f"limit={case.limit}"]
    if case.start is not None:
        parts.append(f"startTime={case.start} ({format_ts(case.start)})")
    if case.end is not None:
        parts.append(f"endTime={case.end} ({format_ts(case.end)})")
    return ", ".join(parts)


_ORDER_NAMES = {
    "empty": "нет свечей",
    "single": "одна свеча",
    "asc": "по возрастанию ts_open",
    "desc": "по убыванию ts_open",
    "mixed": "смешанный",
}


def _times_preview(candles: list[Candle]) -> str:
    stamps = [_short(c.ts_open) for c in candles]
    if len(stamps) <= _PREVIEW_ALL:
        return "; ".join(stamps)
    return "; ".join(stamps[:3]) + " … " + "; ".join(stamps[-3:])


def render_results(results: list[ProbeResult]) -> list[str]:
    lines: list[str] = []
    for result in results:
        case = result.case
        lines.append(f"[{case.key}] {case.title}")
        lines.append(f"    запрос: {_describe_params(case)}")
        if result.candles is None:
            lines.append(f"    ОШИБКА: {result.error}")
            continue
        candles = result.candles
        lines.append(
            f"    получено: {len(candles)}; порядок: {_ORDER_NAMES[detect_order(candles)]}"
        )
        if candles:
            lines.append(f"    времена в порядке ответа: {_times_preview(candles)}")
        if result.raw_item is not None:
            lines.append(f"    сырой объект свечи: {result.raw_item}")
    return lines


def _stamps(result: ProbeResult | None) -> set[int] | None:
    if result is None or result.candles is None:
        return None
    return {c.ts_open for c in result.candles}


def _yes_no(value: bool) -> str:
    return "ДА" if value else "НЕТ"


def derive_findings(
    results: list[ProbeResult], anchor_ms: int, now_ms: int, volume_bar_ms: int
) -> list[str]:
    """Выводы из пробных запросов. Каждый пункт называет факт, на котором основан."""
    by_key = {r.case.key: r for r in results}
    tf = tf_to_ms("15m")
    t = anchor_ms
    out: list[str] = []

    a = _stamps(by_key.get("A"))
    if a is not None:
        out.append(
            f"startTime включительно по времени открытия: {_yes_no(t in a)} "
            f"(A: свеча {_short(t)} {'вернулась' if t in a else 'не вернулась'})"
        )
        last = t + 4 * tf
        out.append(
            f"endTime включительно по времени открытия: {_yes_no(last in a)} "
            f"(A: свеча {_short(last)} {'вернулась' if last in a else 'не вернулась'}; "
            f"всего {len(a)} из 5)"
        )

    b = _stamps(by_key.get("B"))
    if b is not None:
        out.append(
            "start=T-1 мс: "
            + (
                "предыдущая свеча не попала, нужная вернулась"
                if (t - tf) not in b and t in b
                else f"неожиданный состав: {sorted(_short(x) for x in b)}"
            )
            + " (B)"
        )

    c = _stamps(by_key.get("C"))
    if c is not None:
        if t in c:
            out.append(
                "start=T+1 мс: вернулась свеча, В КОТОРУЮ ПОПАЛ start (T): биржа округляет "
                "startTime вниз до начала свечи (C)"
            )
        else:
            out.append("start=T+1 мс: свеча T не вернулась: отбор по времени открытия >= start (C)")

    d = by_key.get("D")
    if d is not None and d.candles:
        times = [x.ts_open for x in d.candles]
        if min(times) == t:
            out.append("только startTime: биржа отдаёт САМЫЕ РАННИЕ свечи от start (D)")
        elif max(times) >= now_ms - 2 * tf:
            out.append(
                "только startTime: биржа отдаёт САМЫЕ НОВЫЕ свечи, start лишь нижняя граница (D)"
            )
        else:
            out.append(f"только startTime: неожиданный состав: {_times_preview(d.candles)} (D)")

    e = by_key.get("E")
    if e is not None and e.candles:
        newest = max(x.ts_open for x in e.candles)
        if newest == t + 100 * tf:
            out.append(
                "только endTime: самые новые свечи, свеча с временем end ВХОДИТ (включительно) (E)"
            )
        elif newest == t + 99 * tf:
            out.append("только endTime: самые новые свечи, свеча с временем end НЕ входит (E)")
        else:
            out.append(f"только endTime: неожиданный состав: {_times_preview(e.candles)} (E)")

    f = by_key.get("F")
    if f is not None and f.candles:
        newest_candle = max(f.candles, key=lambda x: x.ts_open)
        closed = is_candle_closed(newest_candle, tf, now_ms)
        out.append(
            "без времён: самая новая свеча "
            + (
                "ЗАКРЫТА (формирующейся в ответе нет)"
                if closed
                else "ещё ФОРМИРУЕТСЯ (отбрасывать)"
            )
            + f" (F: {_short(newest_candle.ts_open)}, сейчас по часам машины {_short(now_ms)})"
        )

    g = by_key.get("G")
    if g is not None and g.candles is not None:
        n = len(g.candles)
        # Окно [T, T+1500*ТФ]: 1500 свечей, если конец исключён, и 1501, если включён.
        a_stamps = _stamps(by_key.get("A"))
        end_inclusive = a_stamps is not None and (t + 4 * tf) in a_stamps
        in_window = 1501 if end_inclusive else 1500
        if n == 0:
            out.append("широкое окно: пустой ответ (G)")
        else:
            lo = min(x.ts_open for x in g.candles)
            hi = max(x.ts_open for x in g.candles)
            window_last = t + (in_window - 1) * tf  # последняя свеча окна
            if n >= in_window:
                kept = "усечения нет"
            elif lo == t:
                kept = "остались САМЫЕ РАННИЕ, новые потеряны"
            elif hi in (window_last, t + 1500 * tf):
                kept = "остались САМЫЕ НОВЫЕ, ранние потеряны"
            else:
                kept = "состав неожиданный"
            cap_note = (
                f"; ответ короче limit={g.case.limit}: у биржи есть свой предел ответа"
                if n < min(g.case.limit, in_window)
                else ""
            )
            out.append(
                f"окно шире ответа: получено {n} из {in_window} в окне; {kept}{cap_note} "
                f"(G: от {_short(lo)} до {_short(hi)})"
            )

    h = by_key.get("H")
    if h is not None and h.candles is not None:
        out.append(
            f"1m окно 00:00..00:09: получено {len(h.candles)} "
            "(10, если оба края включительно; 9, если конец исключён) (H)"
        )

    i1, i2 = by_key.get("I1"), by_key.get("I2")
    if i1 and i2 and i1.candles is not None and i2.candles is not None:
        s1 = {x.ts_open for x in i1.candles}
        s2 = {x.ts_open for x in i2.candles}
        both = sorted(s1 | s2)
        gaps = sum(1 for p, q in pairwise(both) if q - p != tf)
        out.append(
            f"два соседних окна: {len(s1)} + {len(s2)} свечей, пересечений {len(s1 & s2)}, "
            f"пропусков на стыке и внутри {gaps} (I1, I2)"
        )

    j = by_key.get("J")
    if j is not None:
        out.append(
            f"свеча {_short(volume_bar_ms)} для сверки volume: "
            + ("сырой объект напечатан выше (J)" if j.raw_item else "в ответе не найдена (J)")
        )

    k = by_key.get("K")
    if k is not None and k.candles:
        out.append(
            f"start=2019-01-01, limit 3: вернулись {_times_preview(k.candles)} (K); если это самые "
            "ранние свечи пары, глубину истории можно определять таким запросом (сверить с D)"
        )

    orders = {
        detect_order(r.candles) for r in results if r.candles is not None and len(r.candles) > 1
    }
    out.append(
        "порядок свечей во всех ответах: " + ", ".join(sorted(_ORDER_NAMES[o] for o in orders))
    )
    errors = [r.case.key for r in results if r.candles is None]
    if errors:
        out.append("запросы с ошибкой: " + ", ".join(errors))
    return out


# ----------------------------------------------------------- набор paging: пределы и глубина

DEPTH_YEARS = (2021, 2022, 2023, 2024, 2025)


def build_paging_cases(anchor_ms: int) -> list[ProbeCase]:
    """Второй набор: скрытый предел ответа, границы endTime, 1m и поиск начала истории."""
    tf = tf_to_ms("15m")
    tf1 = tf_to_ms("1m")
    t = anchor_ms
    cases = [
        ProbeCase("L1", "15m без времён, limit 1440", "15m", MAX_LIMIT),
        ProbeCase("L2", "15m только startTime, limit 1440", "15m", MAX_LIMIT, start=t),
        ProbeCase("L3", "15m только endTime, limit 1440", "15m", MAX_LIMIT, end=t + 1700 * tf),
        ProbeCase("W1000", "окно ровно в 1000 баров, limit 1000", "15m", 1000, t, t + 1000 * tf),
        ProbeCase("W1001", "окно в 1001 бар, limit 1440", "15m", MAX_LIMIT, t, t + 1001 * tf),
        ProbeCase("W1440", "окно в 1440 баров, limit 1440", "15m", MAX_LIMIT, t, t + 1440 * tf),
        ProbeCase("W1500", "окно в 1500 баров, limit 1000", "15m", 1000, t, t + 1500 * tf),
        ProbeCase(
            "E2", "только endTime на 1 мс раньше границы, limit 3", "15m", 3, end=t + 100 * tf - 1
        ),
        ProbeCase(
            "E3", "только endTime на 1 мс позже границы, limit 3", "15m", 3, end=t + 100 * tf + 1
        ),
        ProbeCase("X1", "окно с end на 1 мс позже границы", "15m", MAX_LIMIT, t, t + 4 * tf + 1),
        ProbeCase("M1", "1m: окно в 1440 баров, limit 1440", "1m", MAX_LIMIT, t, t + 1440 * tf1),
        ProbeCase("M2", "1m: без времён, limit 1440", "1m", MAX_LIMIT),
    ]
    for year in DEPTH_YEARS:
        cases.append(
            ProbeCase(
                f"R{year}",
                f"15m: только startTime {year}-01-01, limit 1",
                "15m",
                1,
                start=to_ms(datetime(year, 1, 1, tzinfo=UTC)),
            )
        )
    return cases


def derive_paging_findings(results: list[ProbeResult], anchor_ms: int, now_ms: int) -> list[str]:
    by_key = {r.case.key: r for r in results}
    tf = tf_to_ms("15m")
    t = anchor_ms
    out: list[str] = []

    def count(key: str) -> int | None:
        r = by_key.get(key)
        return None if r is None or r.candles is None else len(r.candles)

    def span(key: str) -> str:
        r = by_key.get(key)
        if r is None or not r.candles:
            return "нет свечей"
        lo = min(x.ts_open for x in r.candles)
        hi = max(x.ts_open for x in r.candles)
        return f"от {_short(lo)} до {_short(hi)}"

    for key, label in (
        ("L1", "15m без времён, limit 1440"),
        ("L2", "15m только startTime, limit 1440"),
        ("L3", "15m только endTime, limit 1440"),
        ("W1000", "окно 1000 баров, limit 1000"),
        ("W1001", "окно 1001 бар, limit 1440"),
        ("W1440", "окно 1440 баров, limit 1440"),
        ("W1500", "окно 1500 баров, limit 1000"),
        ("M1", "1m окно 1440 баров, limit 1440"),
        ("M2", "1m без времён, limit 1440"),
    ):
        n = count(key)
        if n is not None:
            out.append(f"{label}: получено {n}, {span(key)} ({key})")

    caps = []
    for interval, keys in (("15m", ("L1", "L2", "L3", "W1001", "W1440")), ("1m", ("M1", "M2"))):
        known = [c for c in (count(k) for k in keys) if c is not None]
        if known:
            caps.append(f"{interval}: {max(known)} свечей")
    if caps:
        out.append("наибольший ответ при limit=1440: " + "; ".join(caps))

    w1001 = by_key.get("W1001")
    if w1001 is not None and w1001.candles:
        lo = min(x.ts_open for x in w1001.candles)
        if len(w1001.candles) < 1001:
            side = "ранние" if lo > t else "новые"
            out.append(f"окно в 1001 бар не поместилось в ответ: потеряны {side} свечи (W1001)")

    for key, end_ms, text in (
        ("E2", t + 100 * tf - 1, "endTime на 1 мс раньше границы"),
        ("E3", t + 100 * tf + 1, "endTime на 1 мс позже границы"),
    ):
        r = by_key.get(key)
        if r is not None and r.candles:
            newest = max(x.ts_open for x in r.candles)
            if newest == -(-end_ms // tf) * tf:
                verdict = "endTime округляется ВВЕРХ до границы свечи, свеча на границе входит"
            elif newest == end_ms - end_ms % tf:
                verdict = "endTime округляется вниз до границы свечи, свеча на границе входит"
            else:
                verdict = "неожиданный состав"
            out.append(
                f"только endTime, {text}: самая новая свеча {_short(newest)} ({key}): {verdict}"
            )

    x1 = by_key.get("X1")
    if x1 is not None and x1.candles is not None:
        has_boundary = (t + 4 * tf) in {c.ts_open for c in x1.candles}
        out.append(
            f"окно с end на 1 мс позже границы: свеча на границе "
            f"{'ВХОДИТ' if has_boundary else 'не входит'}; получено {len(x1.candles)} (X1)"
        )

    depth: list[str] = []
    for year in DEPTH_YEARS:
        r = by_key.get(f"R{year}")
        if r is None:
            continue
        if r.candles is None:
            depth.append(f"{year}: ошибка")
        elif not r.candles:
            depth.append(f"{year}: пусто")
        else:
            depth.append(f"{year}: {_short(r.candles[0].ts_open)}")
    all_empty = all(
        r.candles is not None and not r.candles for r in results if r.case.key.startswith("R")
    )
    if depth and all_empty:
        out.append(
            "при startTime раньше начала истории биржа отдаёт ПУСТОЙ ответ, а не самые ранние "
            "свечи: начало истории ищется бисекцией (`--set depth`)"
        )
    if depth:
        out.append(
            "первая свеча при startTime=1 января года (пусто = биржа ничего не вернула): "
            + "; ".join(depth)
        )

    errors = [r.case.key for r in results if r.candles is None]
    if errors:
        out.append("запросы с ошибкой: " + ", ".join(errors))
    return out


# ------------------------------------------------- набор depth: начало истории по ТФ

DEPTH_INTERVALS = ("15m", "1m", "1h", "4h", "1d")
TARGET_15M_DAYS = 365 + 200  # цель мастера: 12 месяцев плюс 200 суток прогрева (поправка A)
_DAY_MS = 86_400_000


@dataclass(frozen=True, slots=True)
class DepthResult:
    interval: str
    history: HistoryStart | None
    newest_count: int | None = None  # сколько свечей пришло на запрос без времён, limit=1440
    error: str | None = None


def run_depth_probe(
    client: BingXPublicClient,
    symbol: str,
    high_ms: int,
    *,
    intervals: tuple[str, ...] = DEPTH_INTERVALS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[DepthResult]:
    """Для каждого ТФ ищет самую раннюю свечу бисекцией по startTime и меряет предел ответа."""
    results: list[DepthResult] = []
    low_ms = to_ms(FAR_PAST)

    def pause() -> None:
        sleep(PAUSE_S)

    for interval in intervals:
        try:
            history = find_history_start(client, symbol, interval, low_ms, high_ms, wait=pause)
            pause()
            newest = client.fetch_klines(symbol, interval, MAX_LIMIT).candles
        except BingXError as exc:
            results.append(DepthResult(interval, None, error=str(exc)))
            continue
        results.append(DepthResult(interval, history, newest_count=len(newest)))
    return results


def render_depth(results: list[DepthResult]) -> list[str]:
    lines: list[str] = []
    for result in results:
        lines.append(f"[{result.interval}] начало истории (бисекция по startTime, limit=1)")
        if result.history is None:
            lines.append(f"    ОШИБКА: {result.error}")
            continue
        history = result.history
        tail = ", ".join(
            f"{_short(start)} {'есть' if found else 'пусто'}" for start, found in history.steps[-4:]
        )
        lines.append(f"    запросов: {history.requests}; последние шаги: {tail}")
        if history.first_ts is None:
            lines.append("    свечей нет даже при самом позднем start")
        else:
            lines.append(f"    самая ранняя свеча: {_short(history.first_ts)} ({history.first_ts})")
        lines.append(f"    без времён, limit=1440: получено {result.newest_count}")
    return lines


def derive_depth_findings(results: list[DepthResult], now_ms: int) -> list[str]:
    out: list[str] = []
    firsts: dict[str, int] = {}
    for r in results:
        if r.history is None:
            out.append(f"{r.interval}: ошибка запроса")
            continue
        h = r.history
        if h.first_ts is None:
            out.append(f"{r.interval}: свечей нет")
            continue
        firsts[r.interval] = h.first_ts
        days = (now_ms - h.first_ts) / _DAY_MS
        note = (
            " (история глубже 2019-01-01, граница поиска)" if h.reason == "deeper_than_low" else ""
        )
        out.append(
            f"{r.interval}: самая ранняя свеча {_short(h.first_ts)}, это {days:.0f} суток "
            f"назад ({days / 30.44:.1f} мес.), запросов на поиск {h.requests}{note}"
        )
    if "15m" in firsts:
        days = (now_ms - firsts["15m"]) / _DAY_MS
        verdict = (
            "ДОСТАТОЧНО" if days >= TARGET_15M_DAYS else f"НЕДОСТАТОЧНО: доступно {days:.0f} суток"
        )
        out.append(
            f"цель по глубине 15m ({TARGET_15M_DAYS} суток: 12 мес. + 200 суток прогрева): "
            f"{verdict}"
        )
    if "15m" in firsts and "1m" in firsts:
        shift = (firsts["1m"] - firsts["15m"]) / _DAY_MS
        out.append(f"начало 1m относительно 15m: {shift:+.1f} суток (0 = совпадает)")
    native = [i for i in ("1h", "4h", "1d") if i in firsts]
    if native and "15m" in firsts:
        parts = [f"{i}: {(firsts[i] - firsts['15m']) / _DAY_MS:+.1f}" for i in native]
        out.append("начало нативных ТФ относительно 15m, суток: " + "; ".join(parts))
    if firsts:
        out.append(
            "ОГОВОРКА: бисекция считает ряд свечей сплошным; если у биржи есть пропуски, начало "
            "могло быть найдено неверно (завышено). Сверьте с `--set coverage`"
        )
    caps = [f"{r.interval}: {r.newest_count}" for r in results if r.newest_count is not None]
    if caps:
        out.append(
            "без времён, limit=1440 (предел ответа, если истории хватает): " + "; ".join(caps)
        )
    return out


# -------------------------------------------- набор coverage: карта наличия данных по месяцам

COVERAGE_FIRST_MONTH = (2022, 1)
# Какие дни месяца пробуем (по одному суточному окну на день): для 15m и 1m два, для остальных один.
COVERAGE_DAYS: dict[str, tuple[int, ...]] = {
    "15m": (1, 15),
    "1m": (1, 15),
    "1h": (1,),
    "4h": (1,),
    "1d": (1,),
}
EXTRA_COVERAGE_DAYS = (datetime(2026, 6, 25, tzinfo=UTC),)  # начало периода по умолчанию
_FULL_SHARE = 0.95  # доля свечей в суточном окне, с которой день считается покрытым
COVERAGE_ATTEMPTS = 2  # одна повторная попытка после ошибки
COVERAGE_RETRY_PAUSE_S = 2.0


@dataclass(frozen=True, slots=True)
class CoverageCell:
    interval: str
    day_ms: int  # начало суточного окна, UTC
    got: int | None  # None = запрос не удался
    expected: int
    error: str | None = None  # причина, если запрос не удался даже после повтора


def expected_bars(interval: str) -> int:
    return _DAY_MS // tf_to_ms(interval)


def coverage_samples(now_ms: int) -> list[tuple[str, int]]:
    """(ТФ, начало суток) для всех месяцев с 2022-01 до последних полных суток."""
    samples: list[tuple[str, int]] = []
    year, month = COVERAGE_FIRST_MONTH
    last_day = now_ms - now_ms % _DAY_MS  # начало текущих суток: они ещё не закончились
    while True:
        for interval, days in COVERAGE_DAYS.items():
            for day in days:
                start = to_ms(datetime(year, month, day, tzinfo=UTC))
                if start + _DAY_MS <= last_day:
                    samples.append((interval, start))
        month += 1
        if month > 12:
            year, month = year + 1, 1
        if to_ms(datetime(year, month, 1, tzinfo=UTC)) >= last_day:
            break
    for interval in ("15m", "1m"):
        for moment in EXTRA_COVERAGE_DAYS:
            if to_ms(moment) + _DAY_MS <= last_day:
                samples.append((interval, to_ms(moment)))
    return samples


def run_coverage_probe(
    client: BingXPublicClient,
    symbol: str,
    now_ms: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> list[CoverageCell]:
    """Для каждого дня-образца просит суточное окно `[день, день+сутки)` и считает свечи."""
    cells: list[CoverageCell] = []
    for index, (interval, day) in enumerate(coverage_samples(now_ms)):
        if index:
            sleep(PAUSE_S)
        expected = expected_bars(interval)
        candles: list[Candle] | None = None
        error = ""
        for attempt in range(COVERAGE_ATTEMPTS):
            if attempt:
                sleep(COVERAGE_RETRY_PAUSE_S)
            try:
                candles = client.fetch_klines(
                    symbol,
                    interval,
                    min(expected, MAX_LIMIT),
                    start_time=day,
                    end_time=day + _DAY_MS,
                ).candles
                break
            except BingXError as exc:
                error = f"{type(exc).__name__}: {exc}"
        if candles is None:
            cells.append(CoverageCell(interval, day, None, expected, error))
            continue
        inside = sum(1 for c in candles if day <= c.ts_open < day + _DAY_MS)
        cells.append(CoverageCell(interval, day, inside, expected))
    return cells


def _cell_text(cell: CoverageCell | None) -> str:
    if cell is None:
        return "-"
    return "ошибка" if cell.got is None else f"{cell.got}/{cell.expected}"


def render_coverage(cells: list[CoverageCell]) -> list[str]:
    columns = [(i, d) for i, days in COVERAGE_DAYS.items() for d in days]
    by_key = {(c.interval, c.day_ms): c for c in cells}
    months = sorted(
        {datetime.fromtimestamp(c.day_ms / 1000, tz=UTC).strftime("%Y-%m") for c in cells}
    )
    header = ["месяц".ljust(8)] + [f"{i} д{d}".rjust(9) for i, d in columns]
    lines = [" ".join(header)]
    for month in months:
        year, mon = (int(x) for x in month.split("-"))
        row = [month.ljust(8)]
        for interval, day in columns:
            key = (interval, to_ms(datetime(year, mon, day, tzinfo=UTC)))
            row.append(_cell_text(by_key.get(key)).rjust(9))
        lines.append(" ".join(row))
    extra = [c for c in cells if any(c.day_ms == to_ms(m) for m in EXTRA_COVERAGE_DAYS)]
    for cell in sorted(extra, key=lambda c: (c.interval, c.day_ms)):
        lines.append(
            f"{_short(cell.day_ms)[:10]} (начало периода по умолчанию) "
            f"{cell.interval}: {_cell_text(cell)}"
        )
    return lines


def derive_coverage_findings(cells: list[CoverageCell], now_ms: int) -> list[str]:
    out: list[str] = []
    for interval in COVERAGE_DAYS:
        mine = sorted((c for c in cells if c.interval == interval), key=lambda c: c.day_ms)
        regular = [c for c in mine if c.got is not None]
        if not regular:
            out.append(f"{interval}: нет данных пробы")
            continue
        covered = [c for c in regular if c.got is not None and c.got >= _FULL_SHARE * c.expected]
        with_data = [c for c in regular if c.got]
        partial = [c for c in with_data if c not in covered]
        if not with_data:
            out.append(f"{interval}: свечей нет ни в одном из {len(regular)} дней-образцов")
            continue
        first_any = with_data[0]
        # «Сплошное» покрытие: первый день, начиная с которого все следующие дни покрыты.
        stable_from = None
        for index, cell in enumerate(regular):
            if all(
                c.got is not None and c.got >= _FULL_SHARE * c.expected for c in regular[index:]
            ):
                stable_from = cell
                break
        parts = [f"первые свечи в образцах {_short(first_any.day_ms)[:10]}"]
        if stable_from is not None:
            days = (now_ms - stable_from.day_ms) / _DAY_MS
            parts.append(
                f"сплошное покрытие с {_short(stable_from.day_ms)[:10]} ({days:.0f} суток назад)"
            )
        else:
            parts.append("сплошного покрытия до конца нет")
        parts.append(f"дней-образцов с данными {len(with_data)} из {len(regular)}")
        if partial:
            shown = ", ".join(f"{_short(c.day_ms)[:10]} {c.got}/{c.expected}" for c in partial[:6])
            parts.append(f"частичных {len(partial)}: {shown}" + (" …" if len(partial) > 6 else ""))
        gaps = [c for c in regular if not c.got and c.day_ms > first_any.day_ms]
        if gaps:
            shown = ", ".join(_short(c.day_ms)[:10] for c in gaps[:6])
            parts.append(
                f"пустых ПОСЛЕ первых данных {len(gaps)}: {shown}" + (" …" if len(gaps) > 6 else "")
            )
        out.append(f"{interval}: " + "; ".join(parts))
    errors = [c for c in cells if c.got is None]
    if errors:
        kinds: dict[str, int] = {}
        for cell in errors:
            kinds[cell.error or "причина не сохранена"] = (
                kinds.get(cell.error or "причина не сохранена", 0) + 1
            )
        details = "; ".join(f"{text} ×{n}" for text, n in sorted(kinds.items()))
        out.append(f"запросов с ошибкой после повтора: {len(errors)} ({details})")
    return out
