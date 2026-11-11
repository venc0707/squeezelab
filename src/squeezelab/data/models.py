"""Модели данных слоя данных: свеча, таймфреймы, порядок и закрытость свечей.

Время везде UTC в миллисекундах (целые числа), начало свечи — `ts_open`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Literal

_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}

Order = Literal["empty", "single", "asc", "desc", "mixed"]


@dataclass(frozen=True, slots=True)
class Candle:
    ts_open: int  # UTC, миллисекунды, начало свечи
    o: float
    h: float
    l: float  # noqa: E741  (имя из контракта данных проекта)
    c: float
    v: float


def tf_to_ms(tf: str) -> int:
    """'15m' -> 900000. Поддерживаются m, h, d, w (месяц '1M' в проекте не используется)."""
    if len(tf) < 2 or not tf[:-1].isdigit() or tf[-1] not in _UNIT_MS or int(tf[:-1]) <= 0:
        raise ValueError(f"неизвестный таймфрейм {tf!r}: ожидается вид 15m, 1h, 1d, 1w")
    return int(tf[:-1]) * _UNIT_MS[tf[-1]]


def format_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def sort_candles(candles: Sequence[Candle]) -> list[Candle]:
    """Новый список по возрастанию ts_open (порядок ответа биржи не гарантирован)."""
    return sorted(candles, key=lambda candle: candle.ts_open)


def detect_order(candles: Sequence[Candle]) -> Order:
    """Порядок свечей в ответе: по возрастанию, по убыванию, смешанный."""
    if not candles:
        return "empty"
    if len(candles) == 1:
        return "single"
    stamps = [candle.ts_open for candle in candles]
    pairs = list(pairwise(stamps))
    if all(b > a for a, b in pairs):
        return "asc"
    if all(b < a for a, b in pairs):
        return "desc"
    return "mixed"


def is_candle_closed(candle: Candle, tf_ms: int, now_ms: int) -> bool:
    """Свеча закрыта, если её конец (ts_open + длина ТФ) не позже текущего момента."""
    return candle.ts_open + tf_ms <= now_ms


def is_contiguous(candles: Sequence[Candle], tf_ms: int) -> bool:
    """Нет пропусков и дублей: соседние (по возрастанию) свечи отстоят ровно на один ТФ."""
    stamps = sorted(candle.ts_open for candle in candles)
    return all(b - a == tf_ms for a, b in pairwise(stamps))
