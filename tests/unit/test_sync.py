"""Тесты P01-T03: докачка пропусков и пометка «подтверждённо отсутствуют»."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fake_bingx import Behaviour, FakeBingX, bar_values

from squeezelab.data.bingx_rest import BingXError, BingXPublicClient
from squeezelab.data.downloader import RateLimiter, RetryingClient, RetryTally
from squeezelab.data.models import Candle
from squeezelab.data.store import CandleStore
from squeezelab.data.sync import (
    CONFIRM_MIN_AGE_MS,
    SYNC_ATTEMPTS,
    SYNC_CHUNK_BARS,
    SyncReport,
    missing_slots,
    render_sync,
    split_runs,
    sync_gaps,
)

SYMBOL = "BNB-USDT"
TF = 900_000


def ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


NOW = ms(2026, 11, 20, 8, 33)
START = ms(2026, 11, 17)


def candle(ts: int) -> Candle:
    values = bar_values(ts, TF)
    return Candle(
        ts,
        float(str(values["open"])),
        float(str(values["high"])),
        float(str(values["low"])),
        float(str(values["close"])),
        float(str(values["volume"])),
    )


@pytest.fixture
def store(tmp_path: object) -> CandleStore:
    return CandleStore(tmp_path / "data")  # type: ignore[operator]


def fast() -> RateLimiter:
    return RateLimiter(clock=lambda: 0.0, sleep=lambda _s: None)


def seed(store: CandleStore, slots: list[int]) -> None:
    store.write(SYMBOL, "15m", [candle(ts) for ts in slots])


def run_sync(
    exchange: FakeBingX,
    store: CandleStore,
    start: int = START,
    end: int | None = None,
    sleeps: list[float] | None = None,
    progress: list[str] | None = None,
) -> SyncReport:
    with BingXPublicClient(transport=exchange.transport, sleep=lambda _s: None) as client:
        return sync_gaps(
            client,
            store,
            SYMBOL,
            "15m",
            start,
            end if end is not None else exchange.now_ms,
            now_ms=exchange.now_ms,
            limiter=fast(),
            sleep=sleeps.append if sleeps is not None else (lambda _s: None),
            progress=progress.append if progress is not None else None,
        )


# ------------------------------------------------------------------- вспомогательные


def test_missing_slots_skips_stored_and_confirmed() -> None:
    have = {0, TF, 3 * TF}
    assert missing_slots(have, {2 * TF}, 0, 6 * TF, TF) == [4 * TF, 5 * TF]
    assert missing_slots(have, set(), 0, 4 * TF, TF) == [2 * TF]
    assert missing_slots(have, set(), 0, 0, TF) == []


def test_split_runs_groups_consecutive_slots_and_limits_length() -> None:
    slots = [0, TF, 2 * TF, 5 * TF, 6 * TF, 9 * TF]
    assert split_runs(slots, TF, 100) == [(0, 2 * TF), (5 * TF, 6 * TF), (9 * TF, 9 * TF)]
    long_run = [i * TF for i in range(250)]
    assert split_runs(long_run, TF, 100) == [
        (0, 99 * TF),
        (100 * TF, 199 * TF),
        (200 * TF, 249 * TF),
    ]
    assert split_runs([], TF, 100) == []


# ----------------------------------------------------------------------- докачка


def test_sync_fills_gaps_the_exchange_has(store: CandleStore) -> None:
    all_slots = [START + i * TF for i in range(200)]
    gaps = {START + 10 * TF, START + 11 * TF, START + 150 * TF}
    seed(store, [ts for ts in all_slots if ts not in gaps])
    exchange = FakeBingX(now_ms=NOW)
    report = run_sync(exchange, store, end=START + 200 * TF)
    assert report.gaps_found == 3
    assert report.filled == 3
    assert report.confirmed_new == 0
    assert report.pending == []
    assert report.requests == 2  # две группы пропусков, каждая нашлась с первой попытки
    assert store.timestamps(SYMBOL, "15m") >= set(all_slots)
    assert store.read(SYMBOL, "15m", START + 10 * TF, START + 12 * TF) == [
        candle(START + 10 * TF),
        candle(START + 11 * TF),
    ]


def test_sync_requests_gaps_as_narrow_end_exclusive_windows(store: CandleStore) -> None:
    all_slots = [START + i * TF for i in range(400)]
    gap = set(range(START + 20 * TF, START + 270 * TF, TF))  # 250 подряд
    seed(store, [ts for ts in all_slots if ts not in gap])
    exchange = FakeBingX(now_ms=NOW)
    report = run_sync(exchange, store, end=START + 400 * TF)
    assert report.filled == 250
    windows = [(int(r["startTime"]), int(r["endTime"])) for r in exchange.data_requests()]
    assert windows == [
        (START + 20 * TF, START + 120 * TF),
        (START + 120 * TF, START + 220 * TF),
        (START + 220 * TF, START + 270 * TF),
    ]
    assert all(int(r["limit"]) == SYNC_CHUNK_BARS for r in exchange.data_requests())


def test_sync_confirms_what_the_exchange_does_not_have(store: CandleStore) -> None:
    holes = {START + 5 * TF, START + 6 * TF}
    slots = [START + i * TF for i in range(40) if START + i * TF not in holes]
    seed(store, slots)
    exchange = FakeBingX(Behaviour(holes=frozenset(holes)), now_ms=NOW)
    report = run_sync(exchange, store, end=START + 40 * TF)
    assert (report.gaps_found, report.filled, report.confirmed_new) == (2, 0, 2)
    assert report.pending == []
    assert report.requests == SYNC_ATTEMPTS  # одна группа, три попытки
    registry = store.confirmed_missing(SYMBOL, "15m")
    assert set(registry) == holes
    assert all(
        item.attempts == SYNC_ATTEMPTS and item.confirmed_at_ms == NOW for item in registry.values()
    )


def test_confirmed_gaps_are_not_requested_again(store: CandleStore) -> None:
    holes = {START + 5 * TF}
    seed(store, [START + i * TF for i in range(40) if START + i * TF not in holes])
    run_sync(FakeBingX(Behaviour(holes=frozenset(holes)), now_ms=NOW), store, end=START + 40 * TF)
    again_exchange = FakeBingX(Behaviour(holes=frozenset(holes)), now_ms=NOW)
    again = run_sync(again_exchange, store, end=START + 40 * TF)
    assert again.gaps_found == 0
    assert again.already_confirmed == 1
    assert again.requests == 0
    assert again_exchange.requests == []
    assert render_sync(again)[-1].startswith("  ранее подтверждённо отсутствуют у биржи: 1")


def test_young_gaps_are_not_confirmed(store: CandleStore) -> None:
    young = NOW - NOW % TF - 2 * 3_600_000  # два часа назад: моложе порога
    assert NOW - (young + TF) < CONFIRM_MIN_AGE_MS
    old = young - 10 * 3_600_000
    slots = [ts for ts in range(old - 5 * TF, young + 5 * TF, TF) if ts not in {young, old}]
    seed(store, slots)
    exchange = FakeBingX(Behaviour(holes=frozenset({young, old})), now_ms=NOW)
    end = young + 5 * TF
    report = run_sync(exchange, store, start=old - 5 * TF, end=end)
    assert report.confirmed_new == 1
    assert set(store.confirmed_missing(SYMBOL, "15m")) == {old}
    assert report.pending == [young]
    lines = "\n".join(render_sync(report))
    assert "ждут подтверждения: 1" in lines
    assert "ВНИМАНИЕ: 1 пропусков моложе 6 ч" in lines
    # на следующем запуске молодой пропуск запрашивается снова, старый нет
    again = run_sync(
        FakeBingX(Behaviour(holes=frozenset({young, old})), now_ms=NOW), store, old - 5 * TF, end
    )
    assert again.gaps_found == 1
    assert again.already_confirmed == 1


def test_candle_that_appears_later_clears_its_confirmed_mark(store: CandleStore) -> None:
    hole = START + 5 * TF
    seed(store, [START + i * TF for i in range(40) if START + i * TF != hole])
    run_sync(FakeBingX(Behaviour(holes=frozenset({hole})), now_ms=NOW), store, end=START + 40 * TF)
    assert set(store.confirmed_missing(SYMBOL, "15m")) == {hole}
    store.write(SYMBOL, "15m", [candle(hole)])  # биржа свечу всё же выдала
    assert store.confirmed_missing(SYMBOL, "15m") == {}


def test_sync_never_touches_the_forming_candle(store: CandleStore) -> None:
    forming = NOW - NOW % TF
    seed(store, [forming - 5 * TF + i * TF for i in range(4)])  # хвост до forming-1 не заполнен
    exchange = FakeBingX(now_ms=NOW)
    report = run_sync(exchange, store, start=forming - 5 * TF, end=forming + 10 * TF)
    assert report.end_ms == forming
    assert forming not in store.timestamps(SYMBOL, "15m")
    assert all(int(r["endTime"]) <= forming for r in exchange.data_requests())


def test_empty_range_does_nothing(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW)
    report = run_sync(exchange, store, start=START, end=START)
    assert report.gaps_found == 0
    assert exchange.requests == []
    assert render_sync(report)[1] == "  пропусков нет"


def test_no_gaps_means_no_requests(store: CandleStore) -> None:
    seed(store, [START + i * TF for i in range(50)])
    exchange = FakeBingX(now_ms=NOW)
    report = run_sync(exchange, store, end=START + 50 * TF)
    assert report.gaps_found == 0
    assert exchange.requests == []


# ------------------------------------------------------------------ сбои и повторы


def test_api_error_propagates_and_keeps_what_was_filled(store: CandleStore) -> None:
    slots = [START + i * TF for i in range(60)]
    gaps = {START + 10 * TF, START + 30 * TF, START + 50 * TF}
    seed(store, [ts for ts in slots if ts not in gaps])
    exchange = FakeBingX(now_ms=NOW, fail_after=1)  # первая группа проходит, вторая падает
    with pytest.raises(BingXError):
        run_sync(exchange, store, end=START + 60 * TF)
    stored = store.timestamps(SYMBOL, "15m")
    assert START + 10 * TF in stored
    assert START + 30 * TF not in stored
    assert store.confirmed_missing(SYMBOL, "15m") == {}  # при сбое ничего не подтверждается


def test_retrying_client_survives_a_transient_error(store: CandleStore) -> None:
    seed(store, [START + i * TF for i in range(40) if i != 5])
    exchange = FakeBingX(now_ms=NOW, fail_requests=frozenset({1}))
    tally = RetryTally()
    sleeps: list[float] = []
    with BingXPublicClient(transport=exchange.transport, sleep=lambda _s: None) as client:
        report = sync_gaps(
            RetryingClient(client, sleeps.append, tally),
            store,
            SYMBOL,
            "15m",
            START,
            START + 40 * TF,
            now_ms=NOW,
            limiter=fast(),
            sleep=lambda _s: None,
        )
    assert report.filled == 1
    assert tally.retries == 1
    assert sleeps == [2.0]


def test_progress_is_reported_every_twenty_groups(store: CandleStore) -> None:
    slots = [START + i * TF for i in range(100)]
    gaps = {ts for i, ts in enumerate(slots) if i % 2 == 1 and i < 99}  # 49 одиночных пропусков
    seed(store, [ts for ts in slots if ts not in gaps])
    messages: list[str] = []
    report = run_sync(FakeBingX(now_ms=NOW), store, end=START + 100 * TF, progress=messages)
    assert report.filled == 49
    assert messages == [
        f"  {SYMBOL} 15m: докачка 20 из 49 групп пропусков",
        f"  {SYMBOL} 15m: докачка 40 из 49 групп пропусков",
    ]


def test_render_sync_lists_counts() -> None:
    report = SyncReport(
        SYMBOL, "15m", START, START + 10 * TF, gaps_found=5, filled=2, confirmed_new=2, requests=9
    )
    report.pending = [START]
    report.retries = 1
    report.last_retry_error = "BingXAPIError: x"
    text = "\n".join(render_sync(report))
    assert "найдено пропусков: 5; докачано: 2; подтверждённо отсутствуют (новых): 2" in text
    assert "ждут подтверждения: 1; запросов: 9" in text
    assert (
        "повторов запросов после временных ошибок: 1 (последняя ошибка: BingXAPIError: x)" in text
    )
