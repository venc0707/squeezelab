"""Читающий API свечей с явным происхождением (P01-T03, поправка B).

Нативные свечи BingX и свечи, собранные нами ресемплом, лежат в разных слоях хранилища и никогда
не смешиваются молча: каждая свеча отдаётся вместе с `src`. По умолчанию (`src="auto"`)
предпочитается нативная: ресемпл подставляется только в те слоты, где нативной нет. Так история
15m до начала нативного окна (около 100 суток) заполняется ресемплом из 1m, а там, где есть
настоящие свечи биржи, используются они.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from squeezelab.data.models import Candle
from squeezelab.data.store import NATIVE_LAYER, RESAMPLED_LAYER, CandleStore

Source = Literal["native", "resampled"]
Prefer = Literal["auto", "native", "resampled"]


@dataclass(frozen=True, slots=True)
class SourcedCandle:
    candle: Candle
    src: Source

    @property
    def ts_open(self) -> int:
        return self.candle.ts_open


class CandleReader:
    def __init__(self, data_dir: Path | str) -> None:
        self.native = CandleStore(data_dir, layer=NATIVE_LAYER)
        self.resampled = CandleStore(data_dir, layer=RESAMPLED_LAYER)

    def read(
        self,
        symbol: str,
        tf: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        *,
        src: Prefer = "auto",
    ) -> list[SourcedCandle]:
        """Свечи `[start_ms, end_ms)` по возрастанию времени, у каждой источник.

        `auto`: нативные, а в слотах без нативной свечи ресемпл. `native` и `resampled`: только
        свой слой.
        """
        if src not in ("auto", "native", "resampled"):
            raise ValueError(f"src {src!r}: ожидается auto, native или resampled")
        native: list[SourcedCandle] = []
        resampled: list[SourcedCandle] = []
        if src in ("auto", "native"):
            native = [
                SourcedCandle(c, "native") for c in self.native.read(symbol, tf, start_ms, end_ms)
            ]
        if src in ("auto", "resampled"):
            resampled = [
                SourcedCandle(c, "resampled")
                for c in self.resampled.read(symbol, tf, start_ms, end_ms)
            ]
        if src != "auto":
            return native or resampled
        have = {item.ts_open for item in native}
        merged = native + [item for item in resampled if item.ts_open not in have]
        merged.sort(key=lambda item: item.ts_open)
        return merged


def count_by_source(items: list[SourcedCandle]) -> dict[str, int]:
    """Сколько свечей каждого происхождения: чтобы смешение было видно, а не молчаливо."""
    counts = Counter(item.src for item in items)
    return {src: counts[src] for src in ("native", "resampled") if counts[src]}
