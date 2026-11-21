"""Тесты P01-T02: поиск начала истории пары бисекцией по startTime (фальшивая биржа)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fake_bingx import Behaviour, FakeBingX

from squeezelab.data.bingx_rest import BingXError, BingXPublicClient
from squeezelab.data.history import find_history_start
from squeezelab.data.models import tf_to_ms

SYMBOL = "BNB-USDT"


def ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


LOW = ms(2019, 1, 1)
HIGH = ms(2026, 11, 1)
LISTING = ms(2025, 3, 9)  # граница суток: выровнена под любой ТФ до 1d


def search(
    exchange: FakeBingX, interval: str, low: int = LOW, high: int = HIGH
) -> tuple[object, int]:
    waits = [0]

    def wait() -> None:
        waits[0] += 1

    with BingXPublicClient(transport=exchange.transport, sleep=lambda _s: None) as client:
        result = find_history_start(client, SYMBOL, interval, low, high, wait=wait)
    assert waits[0] == result.requests  # перед каждым запросом вызван wait
    return result, waits[0]


@pytest.mark.parametrize("interval", ["15m", "1m", "1h", "4h", "1d"])
def test_finds_the_first_candle_on_the_grid(interval: str) -> None:
    exchange = FakeBingX(Behaviour(history_start=LISTING))
    result, _ = search(exchange, interval)
    assert result.reason == "found"  # type: ignore[attr-defined]
    assert result.first_ts == LISTING  # type: ignore[attr-defined]
    steps = result.steps  # type: ignore[attr-defined]
    assert steps[0] == (HIGH - HIGH % tf_to_ms(interval), True)  # первым проверяется верх
    assert all(found == (start >= LISTING) for start, found in steps)  # пусто строго до начала
    assert len(steps) <= 28  # бисекция, а не перебор: 1m за 7,8 лет это около 4 млн баров


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("15m", ms(2025, 3, 9, 7, 30)),
        ("1m", ms(2025, 3, 9, 7, 30)),
        ("1h", ms(2025, 3, 9, 8)),  # свеча 07:00 началась раньше начала истории
        ("4h", ms(2025, 3, 9, 8)),
        ("1d", ms(2025, 3, 10)),
    ],
)
def test_history_that_starts_inside_a_candle_gives_the_first_complete_candle(
    interval: str, expected: int
) -> None:
    exchange = FakeBingX(Behaviour(history_start=ms(2025, 3, 9, 7, 30)))
    result, _ = search(exchange, interval)
    assert result.reason == "found"  # type: ignore[attr-defined]
    assert result.first_ts == expected  # type: ignore[attr-defined]


def test_history_deeper_than_the_lower_bound_is_flagged() -> None:
    exchange = FakeBingX(Behaviour(history_start=ms(2018, 6, 1)))
    result, _ = search(exchange, "1d")
    assert result.reason == "deeper_than_low"  # type: ignore[attr-defined]
    assert result.first_ts == LOW  # type: ignore[attr-defined]
    assert result.requests == 2  # type: ignore[attr-defined]


def test_no_data_when_nothing_exists_even_at_the_high_bound() -> None:
    exchange = FakeBingX(Behaviour(history_start=ms(2030, 1, 1)))
    result, _ = search(exchange, "15m")
    assert result.reason == "no_data"  # type: ignore[attr-defined]
    assert result.first_ts is None  # type: ignore[attr-defined]
    assert result.requests == 1  # type: ignore[attr-defined]


def test_bounds_are_aligned_down_to_the_grid() -> None:
    exchange = FakeBingX(Behaviour(history_start=LISTING))
    result, _ = search(exchange, "1h", LOW + 12_345, HIGH + 54_321)
    assert result.first_ts == LISTING  # type: ignore[attr-defined]
    assert all(start % 3_600_000 == 0 for start, _ in result.steps)  # type: ignore[attr-defined]


def test_inverted_bounds_are_rejected() -> None:
    exchange = FakeBingX(Behaviour(history_start=LISTING))
    with pytest.raises(ValueError, match="раньше"):
        search(exchange, "15m", HIGH, LOW)


def test_api_error_is_not_swallowed() -> None:
    exchange = FakeBingX(Behaviour(history_start=LISTING), fail_after=3)
    with pytest.raises(BingXError):
        search(exchange, "15m")
