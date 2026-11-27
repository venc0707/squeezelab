"""Ресемпл крупного ТФ из мелкого и сверка с нативным (P01-T03, поправка B).

Нативный 15m BingX хранится около 100 суток, а 1m около года. Историю 15m до этого периода
собираем из 1m BingX: корзины по границам :00/:15/:30/:45 UTC, только полные (все 15 минуток на
месте). Результат лежит в отдельном слое хранилища (`candles_resampled/`) и не смешивается с
нативным: читающий API (`data/reader.py`) отдаёт происхождение свечи, нативный предпочитается.

Корзины, которые собрать нельзя (в 1m есть дыра), заносятся в реестр слоя как «не собирается»
(`attempts=0`): `data check` показывает их предупреждением, а когда дыру закроет `data sync`,
следующий ресемпл соберёт корзину, и пометка снимется сама.

Допустимость ресемпла решает `compare_layers`: OHLC нативного 15m и ресемпла на общем периоде
должны совпасть на всех барах; исключения допустимы, пока их меньше 0,1 % и они перечислены в
отчёте; иначе вердикт «не допустим» (остановиться и написать мастеру). Объём сверяется и
перечисляется, но вердикт определяют цены, как требует поправка B.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from squeezelab.data.models import Candle, format_ts, tf_to_ms
from squeezelab.data.quality import OHLC_EXCEPTION_SHARE, PRICE_TOL, VOLUME_TOL
from squeezelab.data.store import CandleStore, ConfirmedMissing, WriteStats

EXCEPTION_SHARE = (
    OHLC_EXCEPTION_SHARE  # доля баров с расхождением OHLC, до которой ресемпл допустим
)
MAX_DETAILS = 5000  # сколько расхождений держим в памяти для отчёта
Verdict = Literal["acceptable", "acceptable_with_exceptions", "not_acceptable", "no_overlap"]
_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class ResampleReport:
    symbol: str
    fine_tf: str
    coarse_tf: str
    start_ms: int = 0
    end_ms: int = 0  # исключительно
    buckets: int = 0  # корзин в диапазоне
    built: int = 0  # собрано полных корзин
    incomplete: int = 0  # корзин с неполным набором мелких свечей
    stats: WriteStats = field(default_factory=WriteStats)
    registered: int = 0  # неполных корзин, занесённых в реестр слоя


def _align_down(value: int, step: int) -> int:
    return value - value % step


def _align_up(value: int, step: int) -> int:
    return -(-value // step) * step


def aggregate(parts: list[Candle], ts_open: int) -> Candle:
    """Крупная свеча из упорядоченных мелких: open первой, close последней, max high, min low."""
    return Candle(
        ts_open,
        parts[0].o,
        max(c.h for c in parts),
        min(c.l for c in parts),
        parts[-1].c,
        sum(c.v for c in parts),
    )


def resample_range(
    native: CandleStore,
    resampled: CandleStore,
    symbol: str,
    fine_tf: str,
    coarse_tf: str,
    *,
    now_ms: int,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> ResampleReport:
    """Собирает `coarse_tf` из нативного `fine_tf` и пишет в слой ресемпла.

    Границы корзин кратны `coarse_tf` от начала эпохи (UTC). Диапазон по умолчанию: от первой
    корзины, целиком покрытой диапазоном мелких свечей, до последней такой корзины.
    """
    coarse_ms, fine_ms = tf_to_ms(coarse_tf), tf_to_ms(fine_tf)
    if coarse_ms <= fine_ms or coarse_ms % fine_ms:
        raise ValueError(f"{coarse_tf} не делится нацело на {fine_tf}")
    if native.layer != "candles" or resampled.layer != "candles_resampled":
        raise ValueError("ресемпл читает нативный слой и пишет в слой ресемпла")
    ratio = coarse_ms // fine_ms
    report = ResampleReport(symbol, fine_tf, coarse_tf)
    fine = native.read(symbol, fine_tf, start_ms, end_ms)
    if not fine:
        return report
    first = _align_up(fine[0].ts_open, coarse_ms)
    last_end = _align_down(fine[-1].ts_open + fine_ms, coarse_ms)  # исключительно
    if start_ms is not None:
        first = max(first, _align_up(start_ms, coarse_ms))
    if end_ms is not None:
        last_end = min(last_end, _align_down(end_ms, coarse_ms))
    report.start_ms, report.end_ms = first, max(first, last_end)
    if last_end <= first:
        return report

    buckets: defaultdict[int, list[Candle]] = defaultdict(list)
    for candle in fine:
        bucket = candle.ts_open - candle.ts_open % coarse_ms
        if first <= bucket < last_end:
            buckets[bucket].append(candle)

    built: list[Candle] = []
    incomplete: list[int] = []
    for bucket in range(first, last_end, coarse_ms):
        parts = buckets.get(bucket, [])
        if len(parts) == ratio:
            built.append(aggregate(sorted(parts, key=lambda c: c.ts_open), bucket))
        else:
            incomplete.append(bucket)
    report.buckets = (last_end - first) // coarse_ms
    report.built = len(built)
    report.incomplete = len(incomplete)
    report.stats = resampled.write(symbol, coarse_tf, built)
    if incomplete:
        report.registered = resampled.add_confirmed_missing(
            symbol, coarse_tf, (ConfirmedMissing(ts, 0, now_ms) for ts in incomplete)
        )
    return report


def render_resample(report: ResampleReport) -> list[str]:
    head = f"{report.symbol}: ресемпл {report.coarse_tf} из {report.fine_tf}"
    if report.buckets == 0:
        return [f"{head}: нечего собирать (нет полных корзин в сохранённых {report.fine_tf})"]
    lines = [
        f"{head}: {format_ts(report.start_ms)} .. {format_ts(report.end_ms)} (конец исключён)",
        f"  корзин: {report.buckets}; собрано полных: {report.built}; "
        f"неполных (в {report.fine_tf} есть дыры): {report.incomplete}",
        f"  запись: новых {report.stats.added}, обновлено {report.stats.updated}, "
        f"без изменений {report.stats.unchanged}, файлов записано {report.stats.files_written}",
    ]
    if report.incomplete:
        lines.append(
            f"  ВНИМАНИЕ: {report.incomplete} корзин не собраны из-за дыр в {report.fine_tf}; "
            "после `data sync` повторите ресемпл"
        )
    return lines


# ------------------------------------------------------------- сравнение с нативным


@dataclass(frozen=True, slots=True)
class Mismatch:
    ts_open: int
    fields: tuple[tuple[str, float, float], ...]  # (поле, нативный, ресемпл)


@dataclass
class LayerComparison:
    symbol: str
    tf: str
    compared: int = 0  # баров, которые есть и в нативном, и в ресемпле
    native_only: int = 0  # нативные бары общего периода без ресемпла (дыры в мелком ТФ)
    resampled_only: int = 0  # бары ресемпла общего периода без нативного
    first_ts: int | None = None
    last_ts: int | None = None
    mismatched_ohlc: int = 0
    mismatched_volume: int = 0
    by_field: dict[str, int] = field(default_factory=dict)
    max_abs: dict[str, float] = field(default_factory=dict)
    details: list[Mismatch] = field(default_factory=list)  # не больше MAX_DETAILS

    @property
    def ohlc_share(self) -> float:
        return self.mismatched_ohlc / self.compared if self.compared else 0.0

    @property
    def verdict(self) -> Verdict:
        if self.compared == 0:
            return "no_overlap"
        if self.mismatched_ohlc == 0:
            return "acceptable"
        if self.ohlc_share < EXCEPTION_SHARE:
            return "acceptable_with_exceptions"
        return "not_acceptable"


def compare_layers(
    native: CandleStore,
    resampled: CandleStore,
    symbol: str,
    tf: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> LayerComparison:
    """Нативный `tf` против ресемпла `tf` на общем периоде (OHLC и объём)."""
    result = LayerComparison(symbol, tf)
    nat = {c.ts_open: c for c in native.read(symbol, tf, start_ms, end_ms)}
    res = {c.ts_open: c for c in resampled.read(symbol, tf, start_ms, end_ms)}
    common = sorted(nat.keys() & res.keys())
    if not common:
        return result
    first, last = common[0], common[-1]
    result.first_ts, result.last_ts = first, last
    result.native_only = sum(1 for ts in nat if first <= ts <= last and ts not in res)
    result.resampled_only = sum(1 for ts in res if first <= ts <= last and ts not in nat)
    for ts in common:
        a, b = nat[ts], res[ts]
        pairs = {
            "open": (a.o, b.o, PRICE_TOL),
            "high": (a.h, b.h, PRICE_TOL),
            "low": (a.l, b.l, PRICE_TOL),
            "close": (a.c, b.c, PRICE_TOL),
            "volume": (a.v, b.v, VOLUME_TOL),
        }
        diffs: list[tuple[str, float, float]] = []
        for name, (x, y, tol) in pairs.items():
            if abs(x - y) > tol + 1e-9 * max(abs(x), abs(y)):
                diffs.append((name, x, y))
                result.by_field[name] = result.by_field.get(name, 0) + 1
                result.max_abs[name] = max(result.max_abs.get(name, 0.0), abs(x - y))
        result.compared += 1
        if diffs:
            if any(name != "volume" for name, _, _ in diffs):
                result.mismatched_ohlc += 1
            if any(name == "volume" for name, _, _ in diffs):
                result.mismatched_volume += 1
            if len(result.details) < MAX_DETAILS:
                result.details.append(Mismatch(ts, tuple(diffs)))
    return result


_VERDICT_TEXT = {
    "acceptable": "ресемпл допустим: OHLC совпадают на всех барах",
    "acceptable_with_exceptions": "ресемпл допустим с исключениями (меньше 0,1 % баров)",
    "not_acceptable": "РЕСЕМПЛ НЕ ДОПУСТИМ: OHLC расходится больше чем на 0,1 % баров",
    "no_overlap": "проверить нечего: нет общего периода нативного 15m и ресемпла",
}


def verdict_text(comparison: LayerComparison) -> str:
    text = _VERDICT_TEXT[comparison.verdict]
    if comparison.verdict == "not_acceptable":
        text += ": остановиться и написать мастеру"
    return text


def render_comparison(comparison: LayerComparison) -> list[str]:
    c = comparison
    head = f"{c.symbol}: нативный {c.tf} против ресемпла"
    if c.compared == 0:
        return [f"{head}: общего периода нет", f"ВЕРДИКТ: {verdict_text(c)}"]
    period = f"{format_ts(c.first_ts or 0)} .. {format_ts(c.last_ts or 0)}"
    lines = [
        f"{head}: общий период {period}, сверено {c.compared} баров",
        f"  OHLC расходится в {c.mismatched_ohlc} барах ({c.ohlc_share * 100:.3f} %), "
        f"объём в {c.mismatched_volume} барах; только в нативном {c.native_only}, "
        f"только в ресемпле {c.resampled_only}",
    ]
    if c.by_field:
        lines.append(
            "  по полям: " + ", ".join(f"{k} ×{c.by_field[k]}" for k in _FIELDS if k in c.by_field)
        )
    lines.append(f"ВЕРДИКТ: {verdict_text(c)}")
    return lines


def render_resample_markdown(
    reports: list[ResampleReport],
    comparisons: list[LayerComparison],
    *,
    generated_ms: int,
    version: str,
) -> str:
    lines = [
        "# Отчёт о ресемпле SqueezeLab",
        "",
        f"- Сформирован: {format_ts(generated_ms)}",
        f"- Версия squeezelab: {version}",
        "- Правило: корзины по границам крупного ТФ в UTC, только полные; свечи пишутся в слой "
        "`candles_resampled`, нативные не затрагиваются",
        "",
    ]
    for r in reports:
        lines += [f"## {r.symbol}: {r.coarse_tf} из {r.fine_tf}", ""]
        lines += [f"- {line.strip()}" for line in render_resample(r)[1:]] or ["- нечего собирать"]
        lines.append("")
    for c in comparisons:
        lines += [f"## {c.symbol}: нативный {c.tf} против ресемпла", ""]
        lines += [f"- **Вердикт:** {verdict_text(c)}"]
        if c.compared:
            period = f"{format_ts(c.first_ts or 0)} .. {format_ts(c.last_ts or 0)}"
            lines += [
                f"- Общий период: {period}; сверено баров: {c.compared}",
                f"- Расхождение OHLC: {c.mismatched_ohlc} баров ({c.ohlc_share * 100:.3f} %); "
                f"допуск исключений: меньше {EXCEPTION_SHARE * 100:.1f} %",
                f"- Расхождение объёма: {c.mismatched_volume} баров (на вердикт не влияет)",
                f"- Только в нативном: {c.native_only}; только в ресемпле: {c.resampled_only}",
                f"- Допуск: цена {PRICE_TOL:g}, объём {VOLUME_TOL:g}",
            ]
        if c.by_field:
            lines += [
                "",
                "| Поле | Баров с расхождением | Наибольшее отклонение |",
                "|---|---|---|",
            ]
            for name in _FIELDS:
                if name in c.by_field:
                    lines.append(f"| {name} | {c.by_field[name]} | {c.max_abs[name]:g} |")
        if c.details:
            lines += ["", f"### Список расхождений (первые {len(c.details)})", ""]
            lines += ["| Время (UTC) | Поля: нативный → ресемпл |", "|---|---|"]
            for item in c.details:
                when = datetime.fromtimestamp(item.ts_open / 1000, tz=UTC).strftime(
                    "%Y-%m-%d %H:%M"
                )
                what = "; ".join(f"{n}: {x:g} → {y:g}" for n, x, y in item.fields)
                lines.append(f"| {when} | {what} |")
        lines.append("")
    return "\n".join(lines) + "\n"
