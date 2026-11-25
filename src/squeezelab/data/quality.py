"""Проверки качества свечей, кросс-проверка ТФ и отчёт (P01-T03).

`check_series` читает файлы хранилища «как лежат» (без слияния) и всегда возвращает полный
набор проверок: каждая либо пройдена (`count == 0`), либо нет. Ошибка (`error`) означает дефект
данных, предупреждение (`warning`) известное и объяснимое состояние (свечи, которых у биржи
нет, помечены «подтверждённо отсутствуют» докачкой, `data/sync.py`).

`cross_check` сверяет крупный ТФ с агрегатом из мелкого (15m из 1m, 1h из 15m и так далее):
open первой, close последней, максимум high, минимум low, сумма объёма. Расхождение цен
допускается не больше 1e-9, объёма не больше 1e-6 (суммы float дают шум порядка 1e-13).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Literal

from squeezelab.data.models import Candle, format_ts, tf_to_ms
from squeezelab.data.store import CandleStore, StoreError, month_of

Severity = Literal["error", "warning"]
MAX_EXAMPLES = 8
PRICE_TOL = 1e-9
VOLUME_TOL = 1e-6
_DAY_MS = 86_400_000


@dataclass(frozen=True, slots=True)
class CheckResult:
    code: str
    title: str
    severity: Severity  # как трактуется нарушение
    count: int = 0  # сколько нарушений; 0 = проверка пройдена
    examples: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.count == 0


@dataclass
class SeriesReport:
    symbol: str
    tf: str
    src: str = "native"  # слой хранилища: native или resampled
    files: int = 0
    rows: int = 0  # строк во всех файлах (дубли считаются)
    unique: int = 0
    first_ts: int | None = None
    last_ts: int | None = None
    expected: int = 0  # слотов сетки от первой до последней свечи
    lag_ms: int | None = None  # на сколько последняя свеча отстаёт от «сейчас» после закрытия
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "error" and not c.passed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "warning" and not c.passed]

    @property
    def ok(self) -> bool:
        return not self.errors


def _utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M")


def _bad_reason(c: Candle) -> str | None:
    values = (c.o, c.h, c.l, c.c, c.v)
    if not all(isfinite(x) for x in values):
        return "значение не число (NaN или бесконечность)"
    if min(c.o, c.h, c.l, c.c) <= 0:
        return "цена не больше нуля"
    if c.h < max(c.o, c.c):
        return "high ниже open или close"
    if c.l > min(c.o, c.c):
        return "low выше open или close"
    if c.h < c.l:
        return "high ниже low"
    if c.v < 0:
        return "объём отрицателен"
    return None


def _missing_runs(missing: list[int], tf_ms: int) -> list[tuple[int, int]]:
    """(первый слот, число слотов) для подряд идущих пропусков."""
    runs: list[tuple[int, int]] = []
    for ts in missing:
        if runs and ts - (runs[-1][0] + (runs[-1][1] - 1) * tf_ms) == tf_ms:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((ts, 1))
    return runs


def _runs_text(runs: list[tuple[int, int]]) -> tuple[str, ...]:
    shown = tuple(f"{_utc(first)} ({n} шт.)" for first, n in runs[:MAX_EXAMPLES])
    extra = len(runs) - MAX_EXAMPLES
    return shown + ((f"… и ещё {extra} групп",) if extra > 0 else ())


def check_series(
    store: CandleStore,
    symbol: str,
    tf: str,
    *,
    now_ms: int,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> SeriesReport:
    """Все проверки одного ряда. `start_ms` включительно, `end_ms` исключительно."""
    tf_ms = tf_to_ms(tf)
    report = SeriesReport(symbol, tf, src=store.src)
    try:
        raw = store.read_raw(symbol, tf)
    except StoreError as exc:
        report.checks.append(
            CheckResult("readable", "Файлы читаются, схема верна", "error", 1, (str(exc),))
        )
        return report
    report.checks.append(CheckResult("readable", "Файлы читаются, схема верна", "error"))
    report.files = len(raw)
    if not raw:
        report.checks.append(
            CheckResult("empty", "В хранилище есть свечи", "error", 1, ("нет ни одного файла",))
        )
        return report

    def in_range(ts: int) -> bool:
        return (start_ms is None or ts >= start_ms) and (end_ms is None or ts < end_ms)

    wrong_month: list[str] = []
    disorder: list[str] = []
    seen: Counter[int] = Counter()
    off_grid: list[str] = []
    bad: defaultdict[str, list[str]] = defaultdict(list)
    bad_total = 0
    rows = 0
    for file in raw:
        previous: int | None = None
        for candle in file.rows:
            if not in_range(candle.ts_open):
                previous = candle.ts_open
                continue
            rows += 1
            seen[candle.ts_open] += 1
            if month_of(candle.ts_open) != file.month:
                wrong_month.append(f"{file.path.name}: {_utc(candle.ts_open)}")
            if previous is not None and candle.ts_open < previous:
                disorder.append(f"{file.path.name}: {_utc(candle.ts_open)} после {_utc(previous)}")
            previous = candle.ts_open
            if tf_ms <= _DAY_MS and candle.ts_open % tf_ms:
                off_grid.append(_utc(candle.ts_open))
            reason = _bad_reason(candle)
            if reason is not None:
                bad_total += 1
                bad[reason].append(
                    f"{_utc(candle.ts_open)}: o={candle.o} h={candle.h} l={candle.l} "
                    f"c={candle.c} v={candle.v}"
                )
    report.rows = rows
    report.unique = len(seen)

    duplicates = sorted(ts for ts, n in seen.items() if n > 1)
    dup_count = sum(seen[ts] - 1 for ts in duplicates)
    bad_examples = tuple(f"{why}: {items[0]}" for why, items in sorted(bad.items()))

    on_grid = sorted(ts for ts in seen if tf_ms > _DAY_MS or ts % tf_ms == 0)
    gap_check = CheckResult("gaps", "Нет пропусков между первой и последней свечой", "error")
    confirmed_check = CheckResult(
        "confirmed", "Свечи, подтверждённо отсутствующие у биржи", "warning"
    )
    if on_grid and tf_ms <= _DAY_MS:
        report.first_ts, report.last_ts = on_grid[0], on_grid[-1]
        report.expected = (on_grid[-1] - on_grid[0]) // tf_ms + 1
        have = set(on_grid)
        try:
            registry = set(store.confirmed_missing(symbol, tf))
        except StoreError as exc:
            registry = set()
            report.checks.append(
                CheckResult(
                    "registry", "Реестр подтверждённых пропусков читается", "error", 1, (str(exc),)
                )
            )
        absent = [ts for ts in range(on_grid[0], on_grid[-1] + 1, tf_ms) if ts not in have]
        unconfirmed = [ts for ts in absent if ts not in registry]
        confirmed = [ts for ts in absent if ts in registry]
        if unconfirmed:
            gap_check = CheckResult(
                "gaps",
                gap_check.title,
                "error",
                len(unconfirmed),
                _runs_text(_missing_runs(unconfirmed, tf_ms)),
            )
        if confirmed:
            confirmed_check = CheckResult(
                "confirmed",
                confirmed_check.title,
                "warning",
                len(confirmed),
                _runs_text(_missing_runs(confirmed, tf_ms)),
            )
    elif on_grid:
        report.first_ts, report.last_ts = on_grid[0], on_grid[-1]

    unclosed = CheckResult("closed", "Последняя свеча закрыта", "error")
    if report.last_ts is not None:
        report.lag_ms = now_ms - (report.last_ts + tf_ms)
        if report.lag_ms < 0:
            unclosed = CheckResult(
                "closed",
                unclosed.title,
                "error",
                1,
                (f"{_utc(report.last_ts)} ещё формируется или лежит в будущем",),
            )

    report.checks += [
        CheckResult(
            "month",
            "Свеча лежит в файле своего месяца",
            "error",
            len(wrong_month),
            tuple(wrong_month[:MAX_EXAMPLES]),
        ),
        CheckResult(
            "order",
            "Время растёт внутри файла",
            "error",
            len(disorder),
            tuple(disorder[:MAX_EXAMPLES]),
        ),
        CheckResult(
            "duplicates",
            "Нет повторов времени открытия",
            "error",
            dup_count,
            tuple(_utc(ts) for ts in duplicates[:MAX_EXAMPLES]),
        ),
        CheckResult(
            "grid",
            "Время открытия на сетке ТФ",
            "error",
            len(off_grid),
            tuple(off_grid[:MAX_EXAMPLES]),
        ),
        CheckResult(
            "values",
            "Значения корректны (high не ниже open и close, low не выше, объём >= 0, нет NaN)",
            "error",
            bad_total,
            bad_examples[:MAX_EXAMPLES],
        ),
        gap_check,
        confirmed_check,
        unclosed,
    ]
    return report


# ------------------------------------------------------------------- кросс-проверка


@dataclass
class CrossCheckReport:
    symbol: str
    coarse_tf: str
    fine_tf: str
    compared: int = 0  # крупных свечей, у которых есть все мелкие
    skipped_incomplete: int = 0  # не хватает части мелких свечей
    skipped_no_fine: int = 0  # мелких свечей нет совсем (вне покрытия мелкого ТФ)
    first_ts: int | None = None
    last_ts: int | None = None
    mismatched: int = 0  # крупных свечей с расхождением хотя бы в одном поле
    by_field: dict[str, int] = field(default_factory=dict)
    max_abs: dict[str, float] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.compared > 0 and self.mismatched == 0


_FIELDS = ("open", "high", "low", "close", "volume")


def cross_check(
    store: CandleStore,
    symbol: str,
    coarse_tf: str,
    fine_tf: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> CrossCheckReport:
    """Сверяет `coarse_tf` с агрегатом из `fine_tf` там, где мелкие свечи есть."""
    coarse_ms, fine_ms = tf_to_ms(coarse_tf), tf_to_ms(fine_tf)
    if coarse_ms <= fine_ms or coarse_ms % fine_ms:
        raise ValueError(f"{coarse_tf} не делится нацело на {fine_tf}")
    ratio = coarse_ms // fine_ms
    report = CrossCheckReport(symbol, coarse_tf, fine_tf)
    coarse = store.read(symbol, coarse_tf, start_ms, end_ms)
    if not coarse:
        return report
    fine = store.read(symbol, fine_tf, coarse[0].ts_open, coarse[-1].ts_open + coarse_ms)
    buckets: defaultdict[int, list[Candle]] = defaultdict(list)
    for candle in fine:
        buckets[candle.ts_open - candle.ts_open % coarse_ms].append(candle)

    for big in coarse:
        parts = buckets.get(big.ts_open, [])
        if not parts:
            report.skipped_no_fine += 1
            continue
        if len(parts) < ratio:
            report.skipped_incomplete += 1
            continue
        report.compared += 1
        report.first_ts = big.ts_open if report.first_ts is None else report.first_ts
        report.last_ts = big.ts_open
        parts.sort(key=lambda c: c.ts_open)
        aggregate = {
            "open": (big.o, parts[0].o, PRICE_TOL),
            "high": (big.h, max(c.h for c in parts), PRICE_TOL),
            "low": (big.l, min(c.l for c in parts), PRICE_TOL),
            "close": (big.c, parts[-1].c, PRICE_TOL),
            "volume": (big.v, sum(c.v for c in parts), VOLUME_TOL),
        }
        diffs: list[str] = []
        for name, (native, built, tol) in aggregate.items():
            delta = abs(native - built)
            if delta > tol + 1e-9 * max(abs(native), abs(built)):
                report.by_field[name] = report.by_field.get(name, 0) + 1
                report.max_abs[name] = max(report.max_abs.get(name, 0.0), delta)
                diffs.append(f"{name}: {fine_tf}-агрегат {built:g}, {coarse_tf} {native:g}")
        if diffs:
            report.mismatched += 1
            if len(report.examples) < MAX_EXAMPLES:
                report.examples.append(f"{_utc(big.ts_open)}: " + "; ".join(diffs))
    return report


# ----------------------------------------------------------------------------- отчёт


def _label(report: SeriesReport) -> str:
    return report.tf if report.src == "native" else f"{report.tf} [ресемпл]"


def series_summary_line(report: SeriesReport) -> str:
    head = f"{report.symbol} {_label(report)}: {report.unique} свечей"
    if report.first_ts is not None and report.last_ts is not None:
        head += f", {_utc(report.first_ts)} .. {_utc(report.last_ts)}"
    if report.ok and not report.warnings:
        return head + ": OK"
    parts = [f"{c.title.split(' (')[0]} ×{c.count}" for c in report.errors]
    parts += [f"предупреждение: {c.title.lower()} ×{c.count}" for c in report.warnings]
    status = "ОШИБКИ" if not report.ok else "OK с предупреждениями"
    return f"{head}: {status}: " + "; ".join(parts)


def cross_summary_line(report: CrossCheckReport) -> str:
    head = f"{report.symbol} {report.coarse_tf} против агрегата из {report.fine_tf}"
    if report.compared == 0:
        return f"{head}: сравнивать нечего (нет общих периодов, пропущено {report.skipped_no_fine})"
    base = (
        f"{head}: сверено {report.compared} свечей, "
        f"пропущено без мелких {report.skipped_no_fine}, неполных {report.skipped_incomplete}"
    )
    if report.ok:
        return base + ": расхождений нет"
    fields = ", ".join(f"{k} ×{v}" for k, v in report.by_field.items())
    return base + f": РАСХОЖДЕНИЯ в {report.mismatched} свечах ({fields})"


def render_markdown(
    series: list[SeriesReport],
    crosses: list[CrossCheckReport],
    *,
    generated_ms: int,
    data_dir: Path,
    version: str,
) -> str:
    errors = sum(len(s.errors) for s in series) + sum(1 for c in crosses if c.mismatched)
    warnings = sum(len(s.warnings) for s in series)
    lines = [
        "# Отчёт о качестве данных SqueezeLab",
        "",
        f"- Сформирован: {format_ts(generated_ms)}",
        f"- Каталог данных: `{data_dir}`",
        f"- Версия squeezelab: {version}",
        "- Итог: "
        + ("**ошибок нет**" if errors == 0 else f"**найдено проблем: {errors}**")
        + (f"; предупреждений: {warnings}" if warnings else ""),
        "",
        "## Сводка",
        "",
        "| Пара | ТФ | Свечей | Период (UTC) | Слотов | Ошибок | Предупреждений |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in series:
        period = (
            f"{_utc(s.first_ts)} .. {_utc(s.last_ts)}"
            if s.first_ts is not None and s.last_ts is not None
            else "-"
        )
        lines.append(
            f"| {s.symbol} | {_label(s)} | {s.unique} | {period} | {s.expected} | "
            f"{len(s.errors)} | {len(s.warnings)} |"
        )
    for s in series:
        lines += ["", f"## {s.symbol} {_label(s)}", ""]
        lines.append(f"Файлов: {s.files}; строк: {s.rows}; уникальных свечей: {s.unique}.")
        if s.lag_ms is not None:
            hours = s.lag_ms / 3_600_000
            lines.append(f"Последняя свеча закрылась {hours:.1f} ч назад (информация, не дефект).")
        lines += ["", "| Проверка | Результат | Нарушений | Примеры |", "|---|---|---|---|"]
        for c in s.checks:
            if c.passed:
                status = "пройдена"
            else:
                status = "ОШИБКА" if c.severity == "error" else "предупреждение"
            examples = "<br>".join(c.examples) if c.examples else ""
            lines.append(f"| {c.title} | {status} | {c.count} | {examples} |")
    if crosses:
        lines += ["", "## Кросс-проверка ТФ", ""]
    for x in crosses:
        lines += [f"### {x.symbol}: {x.coarse_tf} против агрегата из {x.fine_tf}", ""]
        if x.compared == 0:
            lines.append("Сравнивать нечего: нет свечей, у которых есть все мелкие.")
            continue
        period = f"{_utc(x.first_ts)} .. {_utc(x.last_ts)}" if x.first_ts and x.last_ts else "-"
        lines.append(f"- Сверено: {x.compared} свечей ({period})")
        lines.append(
            f"- Пропущено: {x.skipped_no_fine} без мелких свечей, "
            f"{x.skipped_incomplete} с неполным набором мелких"
        )
        lines.append(
            "- Правило: open первой, close последней, max high, min low, сумма объёма; допуск цены "
            f"{PRICE_TOL:g}, объёма {VOLUME_TOL:g}"
        )
        if x.ok:
            lines.append("- Результат: **расхождений нет**")
            continue
        lines.append(f"- Результат: **расхождения в {x.mismatched} свечах**")
        lines += ["", "| Поле | Свечей с расхождением | Наибольшее отклонение |", "|---|---|---|"]
        for name in _FIELDS:
            if name in x.by_field:
                lines.append(f"| {name} | {x.by_field[name]} | {x.max_abs[name]:g} |")
        lines += ["", "Примеры:", ""] + [f"- {e}" for e in x.examples]
    return "\n".join(lines) + "\n"
