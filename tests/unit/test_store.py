"""Тесты P01-T02: хранилище свечей в Parquet (временные каталоги, сеть не нужна)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from squeezelab.data.models import Candle, is_contiguous
from squeezelab.data.store import (
    CONFIRMED_FILE,
    SCHEMA,
    CandleStore,
    ConfirmedMissing,
    StoreError,
    WriteStats,
)

TF = 900_000
GARBAGE = "это не parquet".encode()


def ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def candle(ts: int, bump: float = 0.0) -> Candle:
    base = 100.0 + (ts // TF) % 50 + bump
    return Candle(ts, base, base + 2, base - 1, base + 1, 10.5 + bump)


def run(start: int, count: int, step: int = TF, bump: float = 0.0) -> list[Candle]:
    return [candle(start + i * step, bump) for i in range(count)]


@pytest.fixture
def store(tmp_path: Path) -> CandleStore:
    return CandleStore(tmp_path / "data")


# ------------------------------------------------------------------ запись и чтение


def test_roundtrip_returns_candles_in_ascending_order(store: CandleStore) -> None:
    candles = run(ms(2026, 11, 1), 10)
    store.write("BNB-USDT", "15m", list(reversed(candles)))
    assert store.read("BNB-USDT", "15m") == candles


def test_values_survive_exactly(store: CandleStore) -> None:
    original = Candle(ms(2026, 11, 1), 635.83, 637.41, 635.65, 637.06, 167.58)
    store.write("BNB-USDT", "15m", [original])
    assert store.read("BNB-USDT", "15m") == [original]


def test_layout_and_schema(store: CandleStore) -> None:
    store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 3))
    path = store.root / "candles" / "BNB-USDT" / "15m" / "2026-11.parquet"
    assert path.is_file()
    table = pq.read_table(path)
    assert table.schema.equals(SCHEMA)
    assert [str(f.type) for f in table.schema] == [
        "int64",
        "double",
        "double",
        "double",
        "double",
        "double",
    ]
    assert table.column("ts_open").to_pylist() == sorted(table.column("ts_open").to_pylist())


def test_months_split_on_utc_boundary(store: CandleStore) -> None:
    last_of_october = ms(2026, 10, 31, 23, 45)
    first_of_november = ms(2026, 11, 1)
    store.write("BNB-USDT", "15m", [candle(first_of_november), candle(last_of_october)])
    names = [p.name for p in store.month_files("BNB-USDT", "15m")]
    assert names == ["2026-10.parquet", "2026-11.parquet"]
    assert [c.ts_open for c in store.read("BNB-USDT", "15m")] == [
        last_of_october,
        first_of_november,
    ]


def test_year_boundary(store: CandleStore) -> None:
    store.write("BNB-USDT", "1m", [candle(ms(2026, 12, 31, 23, 59), 0), candle(ms(2027, 1, 1))])
    names = [p.name for p in store.month_files("BNB-USDT", "1m")]
    assert names == ["2026-12.parquet", "2027-01.parquet"]


def test_read_range_is_start_inclusive_end_exclusive(store: CandleStore) -> None:
    candles = run(ms(2026, 11, 1), 8)
    store.write("BNB-USDT", "15m", candles)
    got = store.read("BNB-USDT", "15m", candles[2].ts_open, candles[5].ts_open)
    assert [c.ts_open for c in got] == [c.ts_open for c in candles[2:5]]
    assert store.read("BNB-USDT", "15m", end_ms=candles[0].ts_open) == []


def test_missing_data_reads_as_empty(store: CandleStore) -> None:
    assert store.read("BNB-USDT", "15m") == []
    assert store.timestamps("BNB-USDT", "15m") == set()
    assert store.month_files("BNB-USDT", "15m") == []


def test_symbols_and_timeframes_are_isolated(store: CandleStore) -> None:
    store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 3))
    store.write("BNB-USDT", "1m", run(ms(2026, 11, 1), 5, 60_000))
    store.write("ETH-USDT", "15m", run(ms(2026, 11, 1), 2))
    assert len(store.read("BNB-USDT", "15m")) == 3
    assert len(store.read("BNB-USDT", "1m")) == 5
    assert len(store.read("ETH-USDT", "15m")) == 2


def test_timestamps_collects_all_months(store: CandleStore) -> None:
    candles = [candle(ms(2026, 10, 31, 23, 45)), candle(ms(2026, 11, 1))]
    store.write("BNB-USDT", "15m", candles)
    assert store.timestamps("BNB-USDT", "15m") == {c.ts_open for c in candles}


# ---------------------------------------------------------------------- идемпотентность


def test_second_identical_write_changes_nothing(store: CandleStore) -> None:
    candles = run(ms(2026, 11, 1), 20)
    first = store.write("BNB-USDT", "15m", candles)
    path = store.month_files("BNB-USDT", "15m")[0]
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    second = store.write("BNB-USDT", "15m", candles)
    assert first == WriteStats(added=20, files_written=1)
    assert second == WriteStats(unchanged=20)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_overlapping_write_adds_only_new_candles(store: CandleStore) -> None:
    store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 10))
    stats = store.write("BNB-USDT", "15m", run(ms(2026, 11, 1) + 5 * TF, 10))
    assert (stats.added, stats.unchanged, stats.updated) == (5, 5, 0)
    got = store.read("BNB-USDT", "15m")
    assert len(got) == 15
    assert is_contiguous(got, TF)


def test_duplicates_inside_one_write_are_counted_once(store: CandleStore) -> None:
    c = candle(ms(2026, 11, 1))
    stats = store.write("BNB-USDT", "15m", [c, c, c])
    assert stats.added == 1
    assert len(store.read("BNB-USDT", "15m")) == 1


def test_changed_values_replace_old_and_are_reported(store: CandleStore) -> None:
    store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 4))
    stats = store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 4, bump=1.0))
    assert (stats.added, stats.updated, stats.unchanged) == (0, 4, 0)
    assert store.read("BNB-USDT", "15m") == run(ms(2026, 11, 1), 4, bump=1.0)


def test_only_touched_months_are_rewritten(store: CandleStore) -> None:
    store.write("BNB-USDT", "15m", [candle(ms(2026, 10, 31, 23, 45)), candle(ms(2026, 11, 1))])
    october = store.month_path("BNB-USDT", "15m", 2026, 10)
    before = october.stat().st_mtime_ns
    stats = store.write("BNB-USDT", "15m", [candle(ms(2026, 11, 1, 0, 15))])
    assert stats.files_written == 1
    assert october.stat().st_mtime_ns == before


# ------------------------------------------------------------------------- атомарность


def test_failed_replace_leaves_old_file_intact_and_no_temp_files(
    store: CandleStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 5))
    path = store.month_files("BNB-USDT", "15m")[0]
    before = path.read_bytes()

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("диск отвалился")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="диск отвалился"):
        store.write("BNB-USDT", "15m", run(ms(2026, 11, 1) + 5 * TF, 5))
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["2026-11.parquet"]
    monkeypatch.undo()
    assert len(store.read("BNB-USDT", "15m")) == 5


def test_first_write_failure_leaves_no_file(
    store: CandleStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("нет места")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="нет места"):
        store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 3))
    directory = store.root / "candles" / "BNB-USDT" / "15m"
    assert list(directory.iterdir()) == []


def test_temp_files_and_foreign_names_are_ignored(store: CandleStore) -> None:
    store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 3))
    directory = store.root / "candles" / "BNB-USDT" / "15m"
    (directory / "2026-11.parquet.tmp-123").write_bytes(b"garbage")
    (directory / "notes.txt").write_text("не свечи", encoding="utf-8")
    (directory / "2026-13.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in store.month_files("BNB-USDT", "15m")] == ["2026-11.parquet"]
    assert len(store.read("BNB-USDT", "15m")) == 3


# ------------------------------------------------------------------------ проверки


@pytest.mark.parametrize("symbol", ["bnb-usdt", "BNBUSDT", "../etc", "BNB-USDT/x", ""])
def test_bad_symbol_is_rejected_before_touching_disk(store: CandleStore, symbol: str) -> None:
    with pytest.raises(ValueError, match="символ"):
        store.write(symbol, "15m", run(ms(2026, 11, 1), 1))
    assert not store.root.exists()


@pytest.mark.parametrize("tf", ["15x", "m15", "0m", "../1m", "", "15"])
def test_bad_timeframe_is_rejected(store: CandleStore, tf: str) -> None:
    with pytest.raises(ValueError, match="таймфрейм"):
        store.read("BNB-USDT", tf)


def test_corrupted_file_raises_store_error(store: CandleStore) -> None:
    store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 3))
    path = store.month_files("BNB-USDT", "15m")[0]
    path.write_bytes(GARBAGE)
    with pytest.raises(StoreError, match="не удалось прочитать"):
        store.read("BNB-USDT", "15m")
    with pytest.raises(StoreError):
        store.write("BNB-USDT", "15m", run(ms(2026, 11, 1) + 10 * TF, 1))
    assert path.read_bytes() == GARBAGE  # повреждённое не затирается молча


def test_foreign_schema_raises_store_error(store: CandleStore) -> None:
    import pyarrow as pa

    path = store.month_path("BNB-USDT", "15m", 2026, 11)
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"ts_open": pa.array([1], type=pa.int64())}), path)
    with pytest.raises(StoreError, match="схема"):
        store.read("BNB-USDT", "15m")


# ------------------------------------------------------ чтение «как лежит» и реестр пропусков


def write_raw_month(store: CandleStore, tf: str, year: int, month: int, rows: list[Candle]) -> Path:
    """Пишет файл месяца напрямую, мимо слияния: так можно получить дубли и беспорядок."""
    import pyarrow as pa

    path = store.month_path("BNB-USDT", tf, year, month)
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
    return path


def test_read_raw_keeps_duplicates_and_order_as_in_the_file(store: CandleStore) -> None:
    first = ms(2026, 11, 1)
    rows = [candle(first + TF), candle(first), candle(first), candle(first + 2 * TF)]
    write_raw_month(store, "15m", 2026, 11, rows)
    (raw,) = store.read_raw("BNB-USDT", "15m")
    assert raw.month == (2026, 11)
    assert [c.ts_open for c in raw.rows] == [first + TF, first, first, first + 2 * TF]
    assert len(store.read("BNB-USDT", "15m")) == 3  # обычное чтение сливает дубли


def test_read_raw_of_missing_data_is_empty(store: CandleStore) -> None:
    assert store.read_raw("BNB-USDT", "15m") == []


def test_symbols_and_timeframes_are_listed_from_disk(store: CandleStore) -> None:
    store.write("ETH-USDT", "1m", run(ms(2026, 11, 1), 2, 60_000))
    store.write("BNB-USDT", "1h", run(ms(2026, 11, 1), 2, 3_600_000))
    store.write("BNB-USDT", "1m", run(ms(2026, 11, 1), 2, 60_000))
    store.write("BNB-USDT", "15m", run(ms(2026, 11, 1), 2))
    (store.root / "candles" / "BNB-USDT" / "4h").mkdir()  # пустой каталог не считается
    (store.root / "candles" / "notes").mkdir()
    assert store.symbols() == ["BNB-USDT", "ETH-USDT"]
    assert store.timeframes("BNB-USDT") == ["1m", "15m", "1h"]
    assert store.timeframes("SOL-USDT") == []
    assert store.timeframes("bad name") == []
    assert CandleStore(store.root / "нет").symbols() == []


def test_confirmed_missing_roundtrip_and_update(store: CandleStore) -> None:
    ts = ms(2026, 11, 1, 3)
    assert store.confirmed_missing("BNB-USDT", "15m") == {}
    assert store.add_confirmed_missing("BNB-USDT", "15m", [ConfirmedMissing(ts, 3, 111)]) == 1
    assert store.add_confirmed_missing("BNB-USDT", "15m", [ConfirmedMissing(ts, 6, 222)]) == 0
    assert store.confirmed_missing("BNB-USDT", "15m") == {ts: ConfirmedMissing(ts, 6, 222)}
    assert (store.root / "candles" / "BNB-USDT" / "15m" / CONFIRMED_FILE).is_file()
    assert store.month_files("BNB-USDT", "15m") == []  # реестр не считается файлом свечей


def test_writing_a_candle_removes_its_confirmed_missing_mark(store: CandleStore) -> None:
    hole, other = ms(2026, 11, 1, 3), ms(2026, 11, 1, 4)
    store.add_confirmed_missing(
        "BNB-USDT", "15m", [ConfirmedMissing(hole, 3, 1), ConfirmedMissing(other, 3, 1)]
    )
    store.write("BNB-USDT", "15m", [candle(hole)])
    assert set(store.confirmed_missing("BNB-USDT", "15m")) == {other}
    store.write("BNB-USDT", "15m", [candle(hole)])  # повторная запись ничего не меняет
    assert set(store.confirmed_missing("BNB-USDT", "15m")) == {other}


def test_corrupted_registry_raises_store_error(store: CandleStore) -> None:
    store.add_confirmed_missing("BNB-USDT", "15m", [ConfirmedMissing(ms(2026, 11, 1), 3, 1)])
    path = store.confirmed_path("BNB-USDT", "15m")
    path.write_text("{не json", encoding="utf-8")
    with pytest.raises(StoreError, match="реестр"):
        store.confirmed_missing("BNB-USDT", "15m")


def test_failed_registry_write_keeps_the_old_registry(
    store: CandleStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ts = ms(2026, 11, 1)
    store.add_confirmed_missing("BNB-USDT", "15m", [ConfirmedMissing(ts, 3, 1)])

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("нет места")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="нет места"):
        store.add_confirmed_missing("BNB-USDT", "15m", [ConfirmedMissing(ts + TF, 3, 1)])
    monkeypatch.undo()
    assert set(store.confirmed_missing("BNB-USDT", "15m")) == {ts}
    directory = store.confirmed_path("BNB-USDT", "15m").parent
    assert not [p for p in directory.iterdir() if ".tmp" in p.name]
