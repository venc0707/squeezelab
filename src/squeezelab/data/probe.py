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
        out.append(
            "только endTime: "
            + (
                "самые новые свечи не позже end (E)"
                if newest in (t + 100 * tf, t + 99 * tf)
                else f"неожиданный состав: {_times_preview(e.candles)} (E)"
            )
        )

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
        if n == 0:
            out.append("широкое окно: пустой ответ (G)")
        else:
            lo = min(x.ts_open for x in g.candles)
            hi = max(x.ts_open for x in g.candles)
            if n < 1500:
                kept = (
                    "оставлены самые ранние"
                    if lo == t
                    else "оставлены самые поздние"
                    if hi == t + 1500 * tf
                    else "состав неожиданный"
                )
            else:
                kept = "усечения нет"
            out.append(
                f"окно шире limit: получено {n} при 1500 в окне; {kept} "
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
