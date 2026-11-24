"""Тесты P01-T02: загрузчик свечей и команда `data download` на фальшивой бирже."""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pytest
from fake_bingx import Behaviour, FakeBingX, bar_values

from squeezelab.app.cli import main
from squeezelab.data.bingx_rest import BingXError, BingXPublicClient
from squeezelab.data.downloader import (
    WINDOW_BARS,
    DownloadReport,
    RateLimiter,
    align_down,
    align_up,
    download_tf,
    plan_windows,
    render_report,
)
from squeezelab.data.models import Candle, is_contiguous, tf_to_ms
from squeezelab.data.store import CandleStore

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
SYMBOL = "BNB-USDT"


def ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


NOW = ms(2026, 11, 20, 8, 33)  # формирующаяся 15м свеча открылась в 08:30
FORMING_15M = ms(2026, 11, 20, 8, 30)
START = ms(2026, 11, 17)
LONG_START = ms(2026, 10, 1)  # от неё до «сейчас» больше 1000 свечей 15m


def fast_limiter() -> RateLimiter:
    return RateLimiter(clock=lambda: 0.0, sleep=lambda _s: None)


@pytest.fixture
def store(tmp_path: Path) -> CandleStore:
    return CandleStore(tmp_path / "data")


def run_download(
    exchange: FakeBingX,
    store: CandleStore,
    tf: str = "15m",
    start: int = START,
    end: int | None = None,
    *,
    now_ms: int | None = None,
    window_bars: int = WINDOW_BARS,
    sleeps: list[float] | None = None,
) -> DownloadReport:
    with BingXPublicClient(transport=exchange.transport, sleep=lambda _s: None) as client:
        return download_tf(
            client,
            store,
            SYMBOL,
            tf,
            start,
            end if end is not None else exchange.now_ms,
            now_ms=now_ms if now_ms is not None else exchange.now_ms,
            limiter=fast_limiter(),
            window_bars=window_bars,
            sleep=sleeps.append if sleeps is not None else (lambda _s: None),
        )


def assert_matches_exchange(store: CandleStore, tf: str, start: int, end: int) -> None:
    tf_ms = tf_to_ms(tf)
    got = store.read(SYMBOL, tf, start, end)
    assert [c.ts_open for c in got] == list(range(start, end, tf_ms))
    for candle in got:
        expected = bar_values(candle.ts_open, tf_ms)
        assert candle == Candle(
            candle.ts_open,
            float(str(expected["open"])),
            float(str(expected["high"])),
            float(str(expected["low"])),
            float(str(expected["close"])),
            float(str(expected["volume"])),
        )


# ------------------------------------------------------------------- вспомогательное


def test_alignment_helpers() -> None:
    assert align_down(1_000_001, 900_000) == 900_000
    assert align_down(900_000, 900_000) == 900_000
    assert align_up(1, 900_000) == 900_000
    assert align_up(900_000, 900_000) == 900_000


def test_plan_windows_are_adjacent_and_cover_the_period_exactly() -> None:
    tf = 900_000
    windows = plan_windows(0, 2500 * tf, tf, 1000)
    assert windows == [(0, 1000 * tf), (1000 * tf, 2000 * tf), (2000 * tf, 2500 * tf)]
    assert all(a[1] == b[0] for a, b in pairwise(windows))
    assert plan_windows(0, 0, tf) == []
    assert plan_windows(0, tf, tf) == [(0, tf)]


def test_plan_windows_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="границе"):
        plan_windows(1, 900_000, 900_000)
    with pytest.raises(ValueError, match="window_bars"):
        plan_windows(0, 900_000, 900_000, 0)


def test_rate_limiter_spaces_requests_and_does_not_delay_idle_start() -> None:
    clock = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    limiter = RateLimiter(5, clock=lambda: clock[0], sleep=sleep)
    for _ in range(6):
        limiter.wait()
    assert math.isclose(sum(slept), 5 * 0.2)  # пять промежутков между шестью запросами
    slept.clear()
    clock[0] += 10  # долгий простой не копит «кредит»
    limiter.wait()
    limiter.wait()
    assert len(slept) == 1
    assert math.isclose(slept[0], 0.2)


def test_rate_limiter_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError, match="положительным"):
        RateLimiter(0)


# --------------------------------------------------------------------------- загрузка


def test_full_15m_download_matches_exchange_and_skips_forming_candle(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW)
    report = run_download(exchange, store)
    assert report.end_ms == FORMING_15M
    assert report.expected == (FORMING_15M - START) // 900_000
    assert report.stored == report.expected
    assert report.missing == []
    assert report.exchange_newest_ms == FORMING_15M
    assert not report.end_cut_by_exchange
    assert_matches_exchange(store, "15m", START, FORMING_15M)
    assert store.timestamps(SYMBOL, "15m").isdisjoint({FORMING_15M})
    # запросов: граница по бирже, проверка начала + по одному на окно (без повторов)
    assert report.requests == 2 + report.windows == len(exchange.requests)


def test_pagination_across_exchange_cap_gives_complete_contiguous_data(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW)
    start = ms(2026, 11, 17)
    report = run_download(exchange, store, "1m", start, ms(2026, 11, 19, 12))
    end = ms(2026, 11, 19, 12)
    assert report.expected == (end - start) // 60_000 == 3600
    assert report.windows == 4  # 1000 + 1000 + 1000 + 600
    assert report.missing == []
    assert_matches_exchange(store, "1m", start, end)
    assert is_contiguous(store.read(SYMBOL, "1m", start, end), 60_000)
    # окна запрашиваются встык: конец одного равен началу следующего
    windows = [(int(r["startTime"]), int(r["endTime"])) for r in exchange.data_requests()]
    assert windows == plan_windows(start, end, 60_000)
    assert all(int(r["limit"]) == WINDOW_BARS for r in exchange.data_requests())


def test_months_are_split_into_files(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW)
    start = ms(2026, 10, 31, 22)
    run_download(exchange, store, "15m", start, ms(2026, 11, 1, 2))
    assert [p.name for p in store.month_files(SYMBOL, "15m")] == [
        "2026-10.parquet",
        "2026-11.parquet",
    ]
    assert_matches_exchange(store, "15m", start, ms(2026, 11, 1, 2))


def test_unaligned_period_is_rounded_inward(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW)
    report = run_download(exchange, store, "15m", START + 1, START + 3 * 900_000 + 5)
    assert report.start_ms == START + 900_000
    assert report.end_ms == START + 3 * 900_000
    assert report.expected == 2


@pytest.mark.parametrize(
    ("start", "end"),
    [(START, START), (START + 900_000, START), (NOW, NOW + 10_000_000)],
)
def test_empty_or_future_period_downloads_nothing(store: CandleStore, start: int, end: int) -> None:
    exchange = FakeBingX(now_ms=NOW)
    report = run_download(exchange, store, "15m", start, end)
    assert report.expected == 0
    assert report.missing == []
    assert store.month_files(SYMBOL, "15m") == []
    assert "нечего загружать" in render_report(report)[0]


def test_period_end_in_the_future_is_clipped_to_last_closed_candle(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW)
    report = run_download(exchange, store, "15m", START, NOW + 10 * 86_400_000)
    assert report.end_ms == FORMING_15M
    assert max(store.timestamps(SYMBOL, "15m")) == FORMING_15M - 900_000


# ----------------------------------------------------------------- повторный запуск


def test_rerun_requests_only_the_exchange_bound(store: CandleStore) -> None:
    run_download(FakeBingX(now_ms=NOW), store)
    again = FakeBingX(now_ms=NOW)
    report = run_download(again, store)
    assert report.windows_skipped == report.windows
    assert report.requests == 1  # только проверка границы по бирже
    assert report.stats.added == 0
    assert report.stats.files_written == 0
    assert report.missing == []
    assert again.data_requests() == []


def test_rerun_after_partial_download_fetches_only_missing_windows(store: CandleStore) -> None:
    first = FakeBingX(now_ms=NOW)
    cut = LONG_START + 1000 * 900_000
    run_download(first, store, "15m", LONG_START, cut)
    second = FakeBingX(now_ms=NOW)
    report = run_download(second, store, "15m", LONG_START)
    assert report.windows_skipped == 1
    requested = [int(r["startTime"]) for r in second.data_requests()]
    assert requested[0] == cut  # первое окно уже было целиком
    assert_matches_exchange(store, "15m", LONG_START, FORMING_15M)


def test_missing_candle_inside_a_window_is_refetched(store: CandleStore) -> None:
    run_download(FakeBingX(now_ms=NOW), store)
    victim = START + 5 * 900_000
    kept = [c for c in store.read(SYMBOL, "15m") if c.ts_open != victim]
    for path in store.month_files(SYMBOL, "15m"):
        path.unlink()
    store.write(SYMBOL, "15m", kept)
    assert victim not in store.timestamps(SYMBOL, "15m")
    report = run_download(FakeBingX(now_ms=NOW), store)
    assert report.stats.added == 1
    assert report.missing == []
    assert victim in store.timestamps(SYMBOL, "15m")


# ------------------------------------------------------------------ закрытость свечи


def test_fast_local_clock_cannot_leak_the_forming_candle(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW)
    fast_clock = NOW + 20 * 60_000  # часы машины спешат на 20 минут
    report = run_download(exchange, store, now_ms=fast_clock, end=fast_clock)
    assert report.end_cut_by_exchange
    assert report.end_ms == FORMING_15M
    assert max(store.timestamps(SYMBOL, "15m")) == FORMING_15M - 900_000
    assert any("часы этой машины спешат" in line for line in render_report(report))


def test_slow_local_clock_is_respected(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW)
    slow_clock = NOW - 40 * 60_000
    report = run_download(exchange, store, now_ms=slow_clock, end=NOW)
    assert report.end_ms == align_down(slow_clock, 900_000)
    assert not report.end_cut_by_exchange


# ---------------------------------------------------------------- странности биржи


def test_exchange_holes_are_confirmed_missing_and_never_requested_again(
    store: CandleStore,
) -> None:
    holes = frozenset({START + 3 * 900_000, START + 7 * 900_000})
    exchange = FakeBingX(Behaviour(holes=holes), now_ms=NOW)
    report = run_download(exchange, store)
    assert report.missing == []  # неподтверждённых пропусков нет
    assert report.confirmed_missing == sorted(holes)
    assert report.stored == report.expected - 2
    assert report.sync is not None
    assert (report.sync.gaps_found, report.sync.filled, report.sync.confirmed_new) == (2, 0, 2)
    assert set(store.confirmed_missing(SYMBOL, "15m")) == holes
    # окно с дырами запрошено 3 раза (попытки), дальше две группы пропусков по 3 попытки докачки
    attempts = [int(r["startTime"]) for r in exchange.data_requests()]
    assert attempts.count(START) == 3
    assert attempts.count(START + 3 * 900_000) == 3
    assert attempts.count(START + 7 * 900_000) == 3
    text = "\n".join(render_report(report))
    assert "ПОДТВЕРЖДЁННО ОТСУТСТВУЮТ у биржи (2)" in text
    assert "докачка пропусков: найдено 2, докачано 0, подтверждённо отсутствуют 2" in text
    assert "пропусков нет" not in text

    again_exchange = FakeBingX(Behaviour(holes=holes), now_ms=NOW)
    again = run_download(again_exchange, store)
    assert again.windows_skipped == again.windows  # окно с подтверждёнными дырами считается полным
    assert again.requests == 1  # только граница по бирже
    assert again.confirmed_missing == sorted(holes)
    assert again.missing == []
    assert again_exchange.data_requests() == []


def test_offgrid_candles_are_dropped_and_counted(store: CandleStore) -> None:
    exchange = FakeBingX(Behaviour(offgrid=True), now_ms=NOW)
    report = run_download(exchange, store, "15m", START, START + 4 * 900_000)
    assert report.dropped_out_of_range >= 1
    assert store.timestamps(SYMBOL, "15m") == {START + i * 900_000 for i in range(4)}


def test_gap_sync_recovers_what_a_smaller_cap_truncated(store: CandleStore) -> None:
    """Окно в 1000 при пределе 300 усекается, но докачка узкими окнами (100) добирает остальное."""
    exchange = FakeBingX(Behaviour(cap=300), now_ms=NOW)
    report = run_download(exchange, store, "15m", LONG_START, LONG_START + 1000 * 900_000)
    assert report.missing == []
    assert report.confirmed_missing == []
    assert report.stored == report.expected
    assert report.sync is not None
    assert report.sync.filled == 700
    assert_matches_exchange(store, "15m", LONG_START, LONG_START + 1000 * 900_000)


def test_window_size_can_follow_a_smaller_cap(store: CandleStore) -> None:
    exchange = FakeBingX(Behaviour(cap=300), now_ms=NOW)
    end = LONG_START + 1000 * 900_000
    report = run_download(exchange, store, "15m", LONG_START, end, window_bars=300)
    assert report.missing == []
    assert report.windows == 4
    assert_matches_exchange(store, "15m", LONG_START, end)


@pytest.mark.parametrize("order", ["desc", "asc"])
def test_response_order_does_not_matter(store: CandleStore, order: str) -> None:
    exchange = FakeBingX(Behaviour(order=order), now_ms=NOW)  # type: ignore[arg-type]
    report = run_download(exchange, store)
    assert report.missing == []
    assert_matches_exchange(store, "15m", START, FORMING_15M)


# ------------------------------------------------------------- начало истории пары


def test_start_before_history_is_clamped_and_nothing_is_lost(store: CandleStore) -> None:
    listing = START + 50 * 900_000
    exchange = FakeBingX(Behaviour(history_start=listing), now_ms=NOW)
    report = run_download(exchange, store)
    assert report.start_clamped
    assert report.requested_start_ms == START
    assert report.start_ms == listing
    assert report.expected == (FORMING_15M - listing) // 900_000
    assert report.stored == report.expected
    assert report.missing == []
    assert min(store.timestamps(SYMBOL, "15m")) == listing
    assert_matches_exchange(store, "15m", listing, FORMING_15M)
    text = "\n".join(render_report(report))
    assert "ВНИМАНИЕ: у биржи нет свечей раньше 2026-11-17 12:30:00 UTC" in text


def test_history_starting_inside_the_first_window_does_not_lose_that_window(
    store: CandleStore,
) -> None:
    listing = LONG_START + 500 * 900_000  # первое окно в 1000 баров начинается до листинга
    exchange = FakeBingX(Behaviour(history_start=listing), now_ms=NOW)
    report = run_download(exchange, store, "15m", LONG_START)
    assert report.start_clamped
    assert report.missing == []
    assert report.stored == report.expected
    assert_matches_exchange(store, "15m", listing, FORMING_15M)


def test_no_data_at_all_is_reported_without_writing(store: CandleStore) -> None:
    exchange = FakeBingX(Behaviour(history_start=NOW + 86_400_000), now_ms=NOW)
    report = run_download(exchange, store)
    assert report.no_data
    assert report.expected == 0
    assert store.month_files(SYMBOL, "15m") == []
    assert "у биржи нет свечей в периоде" in render_report(report)[0]


def test_start_probe_is_skipped_when_the_start_candle_is_already_stored(
    store: CandleStore,
) -> None:
    run_download(FakeBingX(now_ms=NOW), store)
    again = FakeBingX(now_ms=NOW)
    report = run_download(again, store)
    assert report.requests == 1  # только граница по бирже
    assert not report.start_clamped


def test_rerun_with_early_start_clamps_again_without_refetching_data(
    store: CandleStore,
) -> None:
    listing = START + 50 * 900_000
    run_download(FakeBingX(Behaviour(history_start=listing), now_ms=NOW), store)
    second = FakeBingX(Behaviour(history_start=listing), now_ms=NOW)
    report = run_download(second, store)  # снова с START, раньше начала истории
    assert report.start_clamped
    assert report.stats.added == 0
    assert report.windows_skipped == report.windows
    # данные окон не запрашивались; был лишь проверочный запрос окна `[start, начало истории)`
    assert [r for r in second.data_requests() if int(r["limit"]) > 1] == []


def test_clamp_is_refused_when_window_check_finds_earlier_candles(
    store: CandleStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Бисекция могла ошибиться из-за пропусков в данных биржи: период тогда не урезается."""
    from squeezelab.data.history import HistoryStart

    # Гипотеза «только startTime = окно из limit слотов»: дыра в слоте START даёт пустой ответ.
    exchange = FakeBingX(Behaviour(start_only_window=True, holes=frozenset({START})), now_ms=NOW)
    wrong = HistoryStart("15m", START + 10 * 900_000, "found", ((START, False),))
    monkeypatch.setattr("squeezelab.data.downloader.find_history_start", lambda *_a, **_k: wrong)
    report = run_download(exchange, store)
    assert report.history_conflict_ts == START + 10 * 900_000
    assert not report.start_clamped
    assert report.start_ms == START
    assert report.missing == []
    assert report.confirmed_missing == [START]  # единственная настоящая дыра
    assert report.stored == report.expected - 1
    assert any("период НЕ урезан" in line for line in render_report(report))


# ----------------------------------------------------------------- временные ошибки


def test_transient_error_in_a_window_is_retried(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW, fail_requests=frozenset({3}))  # 1 граница, 2 старт, 3 окно
    sleeps: list[float] = []
    report = run_download(exchange, store, sleeps=sleeps)
    assert report.retries == 1
    assert "rate limit" in report.last_retry_error
    assert "BingXAPIError" in report.last_retry_error
    assert sleeps == [2.0]
    assert report.missing == []
    assert_matches_exchange(store, "15m", START, FORMING_15M)
    assert any("повторов запросов после временных ошибок: 1" in x for x in render_report(report))


def test_transient_errors_in_service_requests_are_retried_too(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW, fail_requests=frozenset({1, 3}))  # граница и проверка начала
    sleeps: list[float] = []
    report = run_download(exchange, store, sleeps=sleeps)
    assert report.retries == 2
    assert report.missing == []


def test_retry_delays_grow_and_then_the_error_is_raised(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW, fail_after=0)
    sleeps: list[float] = []
    with pytest.raises(BingXError):
        run_download(exchange, store, sleeps=sleeps)
    assert sleeps == [2.0, 5.0, 10.0]
    assert len(exchange.requests) == 4  # первый запрос и три повтора


def test_progress_survives_when_retries_are_exhausted(store: CandleStore) -> None:
    start, end = ms(2026, 11, 17), ms(2026, 11, 19, 12)  # 1m, четыре окна
    exchange = FakeBingX(now_ms=NOW, fail_after=4)
    sleeps: list[float] = []
    with pytest.raises(BingXError):
        run_download(exchange, store, "1m", start, end, sleeps=sleeps)
    assert sleeps == [2.0, 5.0, 10.0]
    assert store.timestamps(SYMBOL, "1m") == set(range(start, start + 2000 * 60_000, 60_000))


def test_cli_retries_transient_errors_and_reports_them(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli(monkeypatch, FakeBingX(fail_requests=frozenset({3})), tmp_path)
    assert main(ARGS) == 0
    out = capsys.readouterr().out
    assert "повторов запросов после временных ошибок: 1" in out
    assert "пропусков нет" in out


# ----------------------------------------------------------------------- сбой API


def test_api_error_keeps_progress_and_rerun_finishes(store: CandleStore) -> None:
    start, end = ms(2026, 11, 17), ms(2026, 11, 19, 12)  # 1m, четыре окна
    # граница, проверка начала и два окна проходят, третье окно падает
    exchange = FakeBingX(now_ms=NOW, fail_after=4)
    with pytest.raises(BingXError):
        run_download(exchange, store, "1m", start, end)
    saved = store.timestamps(SYMBOL, "1m")
    assert saved == set(range(start, start + 2000 * 60_000, 60_000))
    report = run_download(FakeBingX(now_ms=NOW), store, "1m", start, end)
    assert report.windows_skipped == 2
    assert report.missing == []
    assert_matches_exchange(store, "1m", start, end)


def test_failure_of_the_bound_request_stops_before_any_write(store: CandleStore) -> None:
    exchange = FakeBingX(now_ms=NOW, fail_after=0)
    with pytest.raises(BingXError):
        run_download(exchange, store)
    assert store.month_files(SYMBOL, "15m") == []


# ----------------------------------------------------------------------- отчёт


def test_render_report_for_a_clean_run(store: CandleStore) -> None:
    report = run_download(FakeBingX(now_ms=NOW), store, "15m", START, START + 96 * 900_000)
    text = "\n".join(render_report(report))
    assert "ожидается свечей: 96" in text
    assert "в хранилище за период: 96 из 96" in text
    assert "новых 96" in text
    assert "пропусков нет" in text


# ----------------------------------------------------------------------------- CLI


def _patch_cli(monkeypatch: pytest.MonkeyPatch, exchange: FakeBingX, tmp_path: Path) -> None:
    original = BingXPublicClient.__init__

    def patched(self: BingXPublicClient, base_url: str, **_kwargs: object) -> None:
        original(self, base_url, transport=exchange.transport, sleep=lambda _s: None)

    monkeypatch.setattr(BingXPublicClient, "__init__", patched)
    monkeypatch.setattr("squeezelab.app.cli.RateLimiter", fast_limiter)
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # паузы повторов после ошибок
    monkeypatch.setenv("SQUEEZELAB_DATA_DIR", str(tmp_path / "data"))


ARGS = [
    "data",
    "download",
    "--config-dir",
    str(REPO_CONFIG),
    "--tf",
    "15m",
    "--start",
    "2026-11-17",
    "--end",
    "2026-11-18",
]


def test_cli_download_and_rerun(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exchange = FakeBingX()  # время «сейчас» берётся из часов, период целиком в прошлом
    _patch_cli(monkeypatch, exchange, tmp_path)
    assert main(ARGS) == 0
    out = capsys.readouterr().out
    assert f"Каталог данных: {tmp_path / 'data'} (источник: env)" in out
    assert "BNB-USDT 15m: 2026-11-17 00:00:00 UTC .. 2026-11-18 00:00:00 UTC" in out
    assert "ожидается свечей: 96" in out
    assert "новых 96" in out
    assert "пропусков нет" in out
    store = CandleStore(tmp_path / "data")
    assert len(store.read(SYMBOL, "15m")) == 96

    assert main(ARGS) == 0
    again = capsys.readouterr().out
    assert "новых 0" in again
    assert "уже полных в хранилище: 1" in again


def test_cli_default_timeframes_are_15m_and_1m(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli(monkeypatch, FakeBingX(), tmp_path)
    args = [a for a in ARGS if a not in ("--tf", "15m")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "BNB-USDT 15m:" in out
    assert "BNB-USDT 1m:" in out
    assert "ожидается свечей: 1440" in out
    store = CandleStore(tmp_path / "data")
    assert len(store.read(SYMBOL, "1m")) == 1440


def test_cli_reports_gaps_and_still_exits_zero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hole = ms(2026, 11, 17, 3)
    _patch_cli(monkeypatch, FakeBingX(Behaviour(holes=frozenset({hole}))), tmp_path)
    assert main(ARGS) == 0
    out = capsys.readouterr().out
    assert "ПОДТВЕРЖДЁННО ОТСУТСТВУЮТ у биржи (1): 2026-11-17 03:00:00 UTC" in out
    assert "подтверждённо отсутствуют у биржи свечей всего: 1" in out
    assert "ПРЕДУПРЕЖДЕНИЕ: пропусков всего" not in out


def test_cli_reports_clamped_start(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    listing = ms(2026, 11, 17, 6)
    _patch_cli(monkeypatch, FakeBingX(Behaviour(history_start=listing)), tmp_path)
    assert main(ARGS) == 0
    out = capsys.readouterr().out
    assert "ВНИМАНИЕ: у биржи нет свечей раньше 2026-11-17 06:00:00 UTC" in out
    assert "ожидается свечей: 72" in out
    assert "пропусков нет" in out


def test_cli_api_failure_exits_one_and_keeps_what_was_saved(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli(monkeypatch, FakeBingX(fail_after=0), tmp_path)
    assert main(ARGS) == 1
    err = capsys.readouterr().err
    assert "BNB-USDT 15m: ОШИБКА" in err
    assert "повторный запуск продолжит" in err


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        (["--tf", "7m"], "7m"),
        (["--symbol", "bnb"], "символ"),
        (["--window-bars", "0"], "window-bars"),
        (["--window-bars", "1441"], "window-bars"),
        (["--start", "не-дата"], "не-дата"),
    ],
)
def test_cli_rejects_bad_arguments(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: list[str],
    fragment: str,
) -> None:
    _patch_cli(monkeypatch, FakeBingX(), tmp_path)
    assert main([*ARGS, *extra]) == 2
    assert fragment in capsys.readouterr().err
    assert not (tmp_path / "data").exists()
