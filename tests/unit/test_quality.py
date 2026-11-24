"""Тесты P01-T03: проверки качества, кросс-проверка ТФ, отчёт, команды `data check` и `sync`."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fake_bingx import Behaviour, FakeBingX, bar_values

from squeezelab.app.cli import main
from squeezelab.data.bingx_rest import BingXPublicClient
from squeezelab.data.downloader import RateLimiter
from squeezelab.data.models import Candle
from squeezelab.data.quality import (
    PRICE_TOL,
    check_series,
    cross_check,
    cross_summary_line,
    render_markdown,
    series_summary_line,
)
from squeezelab.data.store import SCHEMA, CandleStore, ConfirmedMissing

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
SYMBOL = "BNB-USDT"
TF = 900_000
MIN = 60_000
FUTURE = datetime(2030, 1, 1, tzinfo=UTC)


def ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


START = ms(2026, 11, 1)
NOW = int(FUTURE.timestamp() * 1000)  # все свечи тестов закрыты


def good(ts: int, tf_ms: int = TF) -> Candle:
    base = 100.0 + (ts // tf_ms) % 50
    return Candle(ts, base, base + 2, base - 1, base + 1, 10.5)


@pytest.fixture
def store(tmp_path: Path) -> CandleStore:
    return CandleStore(tmp_path / "data")


def put(store: CandleStore, rows: list[Candle], tf: str = "15m") -> None:
    store.write(SYMBOL, tf, rows)


def put_raw(store: CandleStore, rows: list[Candle], year: int = 2026, month: int = 11) -> None:
    """Файл месяца мимо слияния: дубли, беспорядок, чужие месяцы, NaN."""
    path = store.month_path(SYMBOL, "15m", year, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "ts_open": pa.array([r.ts_open for r in rows], type=pa.int64()),
            "o": pa.array([r.o for r in rows], type=pa.float64()),
            "h": pa.array([r.h for r in rows], type=pa.float64()),
            "l": pa.array([r.l for r in rows], type=pa.float64()),
            "c": pa.array([r.c for r in rows], type=pa.float64()),
            "v": pa.array([r.v for r in rows], type=pa.float64()),
        },
        schema=SCHEMA,
    )
    pq.write_table(table, path)


def series(n: int = 100) -> list[Candle]:
    return [good(START + i * TF) for i in range(n)]


def result(report, code: str):  # type: ignore[no-untyped-def]
    return next(c for c in report.checks if c.code == code)


# ------------------------------------------------------------------- чистый ряд


def test_clean_series_passes_every_check(store: CandleStore) -> None:
    put(store, series(100))
    report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    assert report.ok
    assert report.errors == []
    assert report.warnings == []
    assert all(c.passed for c in report.checks)
    assert [c.code for c in report.checks] == [
        "readable",
        "month",
        "order",
        "duplicates",
        "grid",
        "values",
        "gaps",
        "confirmed",
        "closed",
    ]
    assert (report.rows, report.unique, report.expected) == (100, 100, 100)
    assert (report.first_ts, report.last_ts) == (START, START + 99 * TF)
    assert report.lag_ms is not None
    assert report.lag_ms > 0
    assert series_summary_line(report).endswith(": OK")


def test_empty_store_is_an_error(store: CandleStore) -> None:
    report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    assert not report.ok
    assert [c.code for c in report.errors] == ["empty"]


# --------------------------------------------------------------------- пропуски


def test_gaps_are_found_with_run_lengths(store: CandleStore) -> None:
    rows = [
        r
        for r in series(100)
        if r.ts_open not in {START + 10 * TF, START + 11 * TF, START + 50 * TF}
    ]
    put(store, rows)
    report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    gaps = result(report, "gaps")
    assert gaps.count == 3
    assert gaps.examples == ("2026-11-01 02:30 (2 шт.)", "2026-11-01 12:30 (1 шт.)")
    assert not report.ok
    assert report.expected == 100
    assert "Нет пропусков между первой и последней свечой ×3" in series_summary_line(report)


def test_confirmed_missing_is_a_warning_not_an_error(store: CandleStore) -> None:
    hole = START + 10 * TF
    put(store, [r for r in series(100) if r.ts_open != hole])
    store.add_confirmed_missing(SYMBOL, "15m", [ConfirmedMissing(hole, 3, NOW)])
    report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    assert report.ok
    assert result(report, "gaps").passed
    assert result(report, "confirmed").count == 1
    assert [c.code for c in report.warnings] == ["confirmed"]
    assert "OK с предупреждениями" in series_summary_line(report)


# ------------------------------------------------- дефекты, которых нет через API хранилища


def test_duplicates_are_found(store: CandleStore) -> None:
    rows = series(10)
    put_raw(store, [*rows, rows[3], rows[3]])
    report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    dup = result(report, "duplicates")
    assert dup.count == 2
    assert dup.examples == ("2026-11-01 00:45",)
    assert (report.rows, report.unique) == (12, 10)
    assert not report.ok


def test_disorder_inside_a_file_is_found(store: CandleStore) -> None:
    rows = series(10)
    rows[4], rows[5] = rows[5], rows[4]
    put_raw(store, rows)
    order = result(check_series(store, SYMBOL, "15m", now_ms=NOW), "order")
    assert order.count == 1
    assert "после" in order.examples[0]


def test_candle_in_the_wrong_month_file_is_found(store: CandleStore) -> None:
    put_raw(store, [good(ms(2026, 10, 31, 23, 45)), good(START)], year=2026, month=11)
    month = result(check_series(store, SYMBOL, "15m", now_ms=NOW), "month")
    assert month.count == 1
    assert "2026-11.parquet: 2026-10-31 23:45" in month.examples[0]


def test_off_grid_candle_is_found(store: CandleStore) -> None:
    rows = series(5)
    rows.append(Candle(START + 5 * TF + 1000, 100, 102, 99, 101, 1))
    put_raw(store, rows)
    report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    assert result(report, "grid").count == 1
    assert report.unique == 6
    assert report.expected == 5  # слоты считаются только по свечам на сетке


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (Candle(START, float("nan"), 102, 99, 101, 1), "значение не число"),
        (Candle(START, float("inf"), 102, 99, 101, 1), "значение не число"),
        (Candle(START, 0.0, 102, 0.0, 101, 1), "цена не больше нуля"),
        (Candle(START, 100, 99, 98, 99.5, 1), "high ниже open или close"),
        (Candle(START, 100, 102, 101, 101.5, 1), "low выше open или close"),
        (Candle(START, 100, 100.5, 101, 100.2, 1), "low выше open или close"),
        (Candle(START, 100, 102, 99, 101, -1.0), "объём отрицателен"),
    ],
)
def test_bad_values_are_found(store: CandleStore, bad: Candle, reason: str) -> None:
    put_raw(store, [bad, *[good(START + i * TF) for i in range(1, 5)]])
    values = result(check_series(store, SYMBOL, "15m", now_ms=NOW), "values")
    assert values.count == 1
    assert reason in values.examples[0]


def test_valid_extreme_candles_are_not_flagged(store: CandleStore) -> None:
    flat = Candle(START, 100, 100, 100, 100, 0.0)  # цена не меняется, объёма нет
    doji = Candle(START + TF, 100, 105, 95, 100, 3.0)
    put(store, [flat, doji])
    assert result(check_series(store, SYMBOL, "15m", now_ms=NOW), "values").passed


# ----------------------------------------------------------------- закрытость и чтение


def test_unclosed_last_candle_is_an_error(store: CandleStore) -> None:
    put(store, series(10))
    last_close = START + 10 * TF
    closed = check_series(store, SYMBOL, "15m", now_ms=last_close)
    assert result(closed, "closed").passed
    unclosed = check_series(store, SYMBOL, "15m", now_ms=last_close - 1)
    assert result(unclosed, "closed").count == 1
    assert not unclosed.ok


def test_unreadable_file_is_reported_without_crashing(store: CandleStore) -> None:
    put(store, series(10))
    store.month_files(SYMBOL, "15m")[0].write_bytes("мусор".encode())
    report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    assert [c.code for c in report.errors] == ["readable"]
    assert not report.ok


def test_range_limits_what_is_checked(store: CandleStore) -> None:
    rows = series(100)
    rows[80] = Candle(rows[80].ts_open, 100, 90, 99, 101, 1)  # дефект после проверяемого периода
    put_raw(store, rows)
    all_report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    assert not all_report.ok
    part = check_series(store, SYMBOL, "15m", now_ms=NOW, end_ms=START + 50 * TF)
    assert part.ok
    assert part.unique == 50


# ------------------------------------------------------------------------ отчёт


def test_markdown_report_lists_checks_and_examples(store: CandleStore) -> None:
    put(store, [r for r in series(100) if r.ts_open != START + 10 * TF])
    report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    text = render_markdown([report], [], generated_ms=NOW, data_dir=Path("/srv/x"), version="9.9")
    assert text.startswith("# Отчёт о качестве данных SqueezeLab")
    assert "- Каталог данных: `/srv/x`" in text
    assert "найдено проблем: 1" in text
    assert "| BNB-USDT | 15m | 99 |" in text
    assert (
        "| Нет пропусков между первой и последней свечой | ОШИБКА | 1 | 2026-11-01 02:30 (1 шт.) |"
        in text
    )
    assert "| Файлы читаются, схема верна | пройдена | 0 |" in text


def test_markdown_report_of_a_clean_series(store: CandleStore) -> None:
    put(store, series(20))
    report = check_series(store, SYMBOL, "15m", now_ms=NOW)
    text = render_markdown([report], [], generated_ms=NOW, data_dir=Path("/d"), version="1")
    assert "**ошибок нет**" in text
    assert "ОШИБКА" not in text


# ---------------------------------------------------------------- кросс-проверка


def one_minute(start: int, minutes: int) -> list[Candle]:
    return [
        Candle(
            start + i * MIN,
            100.0 + i * 0.01,
            100.0 + i * 0.01 + 0.5 + (i % 7) * 0.01,
            100.0 + i * 0.01 - 0.4 - (i % 5) * 0.01,
            100.0 + i * 0.01 + 0.1,
            1.5 + (i % 3),
        )
        for i in range(minutes)
    ]


def aggregate(fine: list[Candle], start: int, ratio: int, tf_ms: int) -> list[Candle]:
    out: list[Candle] = []
    for i in range(len(fine) // ratio):
        part = fine[i * ratio : (i + 1) * ratio]
        out.append(
            Candle(
                start + i * tf_ms,
                part[0].o,
                max(c.h for c in part),
                min(c.l for c in part),
                part[-1].c,
                sum(c.v for c in part),
            )
        )
    return out


def seed_pair(store: CandleStore, minutes: int = 240) -> tuple[list[Candle], list[Candle]]:
    fine = one_minute(START, minutes)
    coarse = aggregate(fine, START, 15, TF)
    put(store, fine, "1m")
    put(store, coarse, "15m")
    return fine, coarse


def test_cross_check_of_consistent_data_is_clean(store: CandleStore) -> None:
    seed_pair(store)
    report = cross_check(store, SYMBOL, "15m", "1m")
    assert report.ok
    assert (report.compared, report.mismatched, report.skipped_incomplete) == (16, 0, 0)
    assert report.first_ts == START
    assert report.last_ts == START + 15 * TF
    assert cross_summary_line(report).endswith("расхождений нет")


def test_cross_check_finds_volume_and_price_mismatches(store: CandleStore) -> None:
    _fine, coarse = seed_pair(store)
    broken = list(coarse)
    broken[2] = Candle(
        broken[2].ts_open, broken[2].o, broken[2].h, broken[2].l, broken[2].c, broken[2].v + 1.0
    )
    broken[5] = Candle(
        broken[5].ts_open,
        broken[5].o + 0.5,
        broken[5].h + 0.5,
        broken[5].l,
        broken[5].c,
        broken[5].v,
    )
    put(store, broken)  # новые значения для тех же времён побеждают
    report = cross_check(store, SYMBOL, "15m", "1m")
    assert not report.ok
    assert report.mismatched == 2
    assert report.by_field == {"volume": 1, "open": 1, "high": 1}
    assert report.max_abs["volume"] == pytest.approx(1.0)
    assert report.max_abs["open"] == pytest.approx(0.5)
    assert any("volume:" in e and "2026-11-01 00:30" in e for e in report.examples)
    assert "РАСХОЖДЕНИЯ в 2 свечах (volume ×1, open ×1, high ×1)" in cross_summary_line(report)


def test_cross_check_tolerates_float_noise(store: CandleStore) -> None:
    _fine, coarse = seed_pair(store)
    noisy = [Candle(c.ts_open, c.o, c.h, c.l, c.c, c.v + 1e-12) for c in coarse]
    put(store, noisy)
    assert cross_check(store, SYMBOL, "15m", "1m").ok
    assert PRICE_TOL == 1e-9


def test_cross_check_skips_bars_without_complete_fine_data(store: CandleStore) -> None:
    fine, _coarse = seed_pair(store)
    for path in store.month_files(SYMBOL, "1m"):
        path.unlink()
    thin = [c for c in fine if c.ts_open != START + 20 * MIN]  # нет одной минуты в бакете 2
    put(store, thin, "1m")
    report = cross_check(store, SYMBOL, "15m", "1m")
    assert report.skipped_incomplete == 1
    assert report.compared == 15
    assert report.ok


def test_cross_check_counts_coarse_bars_outside_fine_coverage(store: CandleStore) -> None:
    all_fine = one_minute(START, 240)
    coarse = aggregate(all_fine, START, 15, TF)
    put(store, all_fine[45:165], "1m")  # мелкие только для крупных свечей 3..10 (8 штук)
    put(store, coarse, "15m")
    report = cross_check(store, SYMBOL, "15m", "1m")
    assert (report.compared, report.skipped_no_fine, report.skipped_incomplete) == (8, 8, 0)
    assert report.ok
    assert report.first_ts == START + 3 * TF


def test_cross_check_without_data_compares_nothing(store: CandleStore) -> None:
    report = cross_check(store, SYMBOL, "15m", "1m")
    assert report.compared == 0
    assert not report.ok
    assert "сравнивать нечего" in cross_summary_line(report)


@pytest.mark.parametrize(("coarse", "fine"), [("1m", "15m"), ("15m", "15m"), ("1h", "7m")])
def test_cross_check_rejects_non_divisible_pairs(
    store: CandleStore, coarse: str, fine: str
) -> None:
    with pytest.raises(ValueError, match="не делится"):
        cross_check(store, SYMBOL, coarse, fine)


def test_cross_check_1h_from_15m(store: CandleStore) -> None:
    fine = [good(START + i * TF) for i in range(24)]
    put(store, fine)
    put(store, aggregate(fine, START, 4, 3_600_000), "1h")
    report = cross_check(store, SYMBOL, "1h", "15m")
    assert report.ok
    assert report.compared == 6


def test_markdown_contains_cross_check_section(store: CandleStore) -> None:
    _fine, coarse = seed_pair(store)
    bad = list(coarse)
    bad[0] = Candle(bad[0].ts_open, bad[0].o, bad[0].h, bad[0].l, bad[0].c, bad[0].v + 3)
    put(store, bad)
    text = render_markdown(
        [check_series(store, SYMBOL, "15m", now_ms=NOW)],
        [cross_check(store, SYMBOL, "15m", "1m")],
        generated_ms=NOW,
        data_dir=Path("/d"),
        version="1",
    )
    assert "## Кросс-проверка ТФ" in text
    assert "### BNB-USDT: 15m против агрегата из 1m" in text
    assert "**расхождения в 1 свечах**" in text
    assert "| volume | 1 |" in text
    assert "найдено проблем: 1" in text


# ------------------------------------------------------------------------------ CLI


@pytest.fixture
def data_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SQUEEZELAB_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def run_cli(*args: str) -> int:
    return main([*args, "--config-dir", str(REPO_CONFIG)])


def test_cli_check_clean_data_writes_report(
    capsys: pytest.CaptureFixture[str], data_env: Path, tmp_path: Path
) -> None:
    seed_pair(CandleStore(data_env))
    report_dir = tmp_path / "reports"
    assert run_cli("data", "check", "--report-dir", str(report_dir)) == 0
    out = capsys.readouterr().out
    assert "BNB-USDT 1m: 240 свечей" in out
    assert ": OK" in out
    assert "BNB-USDT 15m: 16 свечей" in out
    assert "BNB-USDT 15m против агрегата из 1m: сверено 16 свечей" in out
    assert "расхождений нет" in out
    assert "ИТОГ: ошибок нет" in out
    (report,) = report_dir.glob("quality_*.md")
    assert f"Отчёт: {report}" in out
    text = report.read_text(encoding="utf-8")
    assert "**ошибок нет**" in text
    assert "## Кросс-проверка ТФ" in text


def test_cli_check_default_report_dir_is_inside_data_dir(
    capsys: pytest.CaptureFixture[str], data_env: Path
) -> None:
    seed_pair(CandleStore(data_env))
    assert run_cli("data", "check") == 0
    assert list((data_env / "reports").glob("quality_*.md"))


def test_cli_check_exits_one_on_defects(capsys: pytest.CaptureFixture[str], data_env: Path) -> None:
    store = CandleStore(data_env)
    _fine, coarse = seed_pair(store)
    for path in store.month_files(SYMBOL, "15m"):
        path.unlink()
    put(store, [c for c in coarse if c.ts_open != START + 3 * TF])  # дыра в 15m
    assert run_cli("data", "check") == 1
    out = capsys.readouterr().out
    assert "ОШИБКИ: Нет пропусков между первой и последней свечой ×1" in out
    assert "ИТОГ: найдено проблем: 1" in out


def test_cli_check_exits_one_on_cross_mismatch(
    capsys: pytest.CaptureFixture[str], data_env: Path
) -> None:
    store = CandleStore(data_env)
    _fine, coarse = seed_pair(store)
    put(
        store,
        [Candle(coarse[0].ts_open, coarse[0].o, coarse[0].h, coarse[0].l, coarse[0].c, 999.0)],
    )
    assert run_cli("data", "check") == 1
    out = capsys.readouterr().out
    assert "РАСХОЖДЕНИЯ в 1 свечах (volume ×1)" in out


def test_cli_check_options(capsys: pytest.CaptureFixture[str], data_env: Path) -> None:
    seed_pair(CandleStore(data_env))
    assert run_cli("data", "check", "--no-cross", "--tf", "15m") == 0
    out = capsys.readouterr().out
    assert "против агрегата" not in out
    assert "BNB-USDT 1m" not in out
    assert run_cli("data", "check", "--cross", "15m:1m", "--tf", "15m", "--tf", "1m") == 0
    assert "сверено 16 свечей" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        (["--cross", "кривой"], "КРУПНЫЙ:МЕЛКИЙ"),
        (["--cross", "1m:15m"], "должен делиться"),
        (["--start", "не-дата"], "не-дата"),
    ],
)
def test_cli_check_rejects_bad_arguments(
    capsys: pytest.CaptureFixture[str], data_env: Path, extra: list[str], fragment: str
) -> None:
    seed_pair(CandleStore(data_env))
    assert run_cli("data", "check", *extra) == 2
    assert fragment in capsys.readouterr().err


def test_cli_check_without_data_is_a_configuration_error(
    capsys: pytest.CaptureFixture[str], data_env: Path
) -> None:
    assert run_cli("data", "check") == 2
    assert "нет сохранённых свечей" in capsys.readouterr().err
    assert not data_env.exists()


def test_cli_check_reports_confirmed_missing_as_warning(
    capsys: pytest.CaptureFixture[str], data_env: Path
) -> None:
    store = CandleStore(data_env)
    _fine, coarse = seed_pair(store)
    hole = START + 3 * TF
    for path in store.month_files(SYMBOL, "15m"):
        path.unlink()
    put(store, [c for c in coarse if c.ts_open != hole])
    store.add_confirmed_missing(SYMBOL, "15m", [ConfirmedMissing(hole, 3, NOW)])
    assert run_cli("data", "check", "--no-cross") == 0
    out = capsys.readouterr().out
    assert "OK с предупреждениями" in out
    assert "ИТОГ: ошибок нет; предупреждений: 1" in out


# ------------------------------------------------------------------- data sync (CLI)


def patch_exchange(monkeypatch: pytest.MonkeyPatch, exchange: FakeBingX) -> None:
    original = BingXPublicClient.__init__

    def patched(self: BingXPublicClient, base_url: str, **_kwargs: object) -> None:
        original(self, base_url, transport=exchange.transport, sleep=lambda _s: None)

    monkeypatch.setattr(BingXPublicClient, "__init__", patched)
    monkeypatch.setattr(
        "squeezelab.app.cli.RateLimiter",
        lambda: RateLimiter(clock=lambda: 0.0, sleep=lambda _s: None),
    )
    monkeypatch.setattr(time, "sleep", lambda _s: None)


def exchange_candles(start: int, count: int, tf_ms: int = TF) -> list[Candle]:
    out = []
    for i in range(count):
        v = bar_values(start + i * tf_ms, tf_ms)
        out.append(
            Candle(
                start + i * tf_ms,
                float(str(v["open"])),
                float(str(v["high"])),
                float(str(v["low"])),
                float(str(v["close"])),
                float(str(v["volume"])),
            )
        )
    return out


def test_cli_sync_fills_and_confirms(
    capsys: pytest.CaptureFixture[str], data_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CandleStore(data_env)
    rows = exchange_candles(START, 60)
    hole = START + 30 * TF
    absent_at_exchange = {hole}
    put(store, [r for r in rows if r.ts_open not in {START + 10 * TF, hole}])
    patch_exchange(monkeypatch, FakeBingX(Behaviour(holes=frozenset(absent_at_exchange))))
    assert run_cli("data", "sync", "--end", "2026-11-01T15:00") == 0
    out = capsys.readouterr().out
    assert "BNB-USDT 15m: докачка пропусков" in out
    assert "найдено пропусков: 2; докачано: 1; подтверждённо отсутствуют (новых): 1" in out
    assert START + 10 * TF in store.timestamps(SYMBOL, "15m")
    assert set(store.confirmed_missing(SYMBOL, "15m")) == {hole}

    assert run_cli("data", "sync", "--end", "2026-11-01T15:00") == 0
    again = capsys.readouterr().out
    assert "пропусков нет" in again
    assert "ранее подтверждённо отсутствуют у биржи: 1" in again


def test_cli_sync_api_failure_exits_one(
    capsys: pytest.CaptureFixture[str], data_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CandleStore(data_env)
    put(store, [r for r in exchange_candles(START, 30) if r.ts_open != START + 5 * TF])
    patch_exchange(monkeypatch, FakeBingX(fail_after=0))
    assert run_cli("data", "sync", "--end", "2026-11-01T08:00") == 1
    assert "ОШИБКА" in capsys.readouterr().err


def test_cli_sync_without_data_is_a_configuration_error(
    capsys: pytest.CaptureFixture[str], data_env: Path
) -> None:
    assert run_cli("data", "sync") == 2
    assert "нет сохранённых свечей" in capsys.readouterr().err


def test_cli_sync_rejects_bad_timeframe(capsys: pytest.CaptureFixture[str], data_env: Path) -> None:
    put(CandleStore(data_env), exchange_candles(START, 5))
    assert run_cli("data", "sync", "--tf", "7x") == 2
    assert "таймфрейм" in capsys.readouterr().err
