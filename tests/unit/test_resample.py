"""Тесты P01-T03 (поправка B): ресемпл 15m из 1m, сверка с нативным, читающий API, CLI."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_bingx import Behaviour, FakeBingX, bar_values

from squeezelab.app.cli import main
from squeezelab.data.bingx_rest import BingXPublicClient
from squeezelab.data.downloader import RateLimiter
from squeezelab.data.models import Candle
from squeezelab.data.reader import CandleReader, SourcedCandle, count_by_source
from squeezelab.data.resample import (
    EXCEPTION_SHARE,
    LayerComparison,
    aggregate,
    compare_layers,
    render_comparison,
    render_resample,
    render_resample_markdown,
    resample_range,
    verdict_text,
)
from squeezelab.data.store import NATIVE_LAYER, RESAMPLED_LAYER, CandleStore

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
SYMBOL = "BNB-USDT"
TF = 900_000
MIN = 60_000
NOW = int(datetime(2030, 1, 1, tzinfo=UTC).timestamp() * 1000)


def ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


START = ms(2026, 11, 1)


def minutes(start: int, count: int) -> list[Candle]:
    return [
        Candle(
            start + i * MIN,
            100.0 + i * 0.01,
            100.0 + i * 0.01 + 0.5 + (i % 7) * 0.01,
            100.0 + i * 0.01 - 0.4 - (i % 5) * 0.01,
            100.0 + i * 0.01 + 0.1,
            1.5 + (i % 3),
        )
        for i in range(count)
    ]


def build_15m(fine: list[Candle], start: int) -> list[Candle]:
    return [aggregate(fine[i * 15 : (i + 1) * 15], start + i * TF) for i in range(len(fine) // 15)]


@pytest.fixture
def stores(tmp_path: Path) -> tuple[CandleStore, CandleStore]:
    root = tmp_path / "data"
    return CandleStore(root), CandleStore(root, layer=RESAMPLED_LAYER)


def resample(stores: tuple[CandleStore, CandleStore], **kwargs: int | None):  # type: ignore[no-untyped-def]
    native, derived = stores
    return resample_range(native, derived, SYMBOL, "1m", "15m", now_ms=NOW, **kwargs)


# ----------------------------------------------------------------------- слои


def test_layers_are_separate_directories_with_explicit_source(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    native, derived = stores
    assert (native.layer, native.src) == (NATIVE_LAYER, "native")
    assert (derived.layer, derived.src) == (RESAMPLED_LAYER, "resampled")
    assert native.tf_dir(SYMBOL, "15m") != derived.tf_dir(SYMBOL, "15m")
    with pytest.raises(ValueError, match="слой"):
        CandleStore(native.root, layer="other")


# ---------------------------------------------------------------------- сборка


def test_resample_builds_complete_buckets_on_utc_boundaries(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    native, derived = stores
    fine = minutes(START, 240)
    native.write(SYMBOL, "1m", fine)
    report = resample(stores)
    assert (report.buckets, report.built, report.incomplete) == (16, 16, 0)
    assert (report.start_ms, report.end_ms) == (START, START + 16 * TF)
    assert derived.read(SYMBOL, "15m") == build_15m(fine, START)
    assert native.read(SYMBOL, "15m") == []  # нативный слой не затронут
    assert derived.timeframes(SYMBOL) == ["15m"]
    assert all(c.ts_open % TF == 0 for c in derived.read(SYMBOL, "15m"))


def test_aggregate_rule() -> None:
    parts = minutes(START, 15)
    bar = aggregate(parts, START)
    assert bar.o == parts[0].o
    assert bar.c == parts[-1].c
    assert bar.h == max(c.h for c in parts)
    assert bar.l == min(c.l for c in parts)
    assert bar.v == pytest.approx(sum(c.v for c in parts))


def test_partial_first_and_last_buckets_are_not_built(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    native, derived = stores
    # 1m с 00:07 по 03:52 (включительно): первая корзина 00:00 и последняя 03:45 неполные
    native.write(SYMBOL, "1m", minutes(START + 7 * MIN, 226))
    report = resample(stores)
    assert (report.start_ms, report.end_ms) == (START + TF, START + 15 * TF)
    assert report.built == 14
    ts = [c.ts_open for c in derived.read(SYMBOL, "15m")]
    assert ts[0] == START + TF
    assert ts[-1] == START + 14 * TF


def test_bucket_with_a_hole_is_skipped_and_registered_then_built_after_sync(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    native, derived = stores
    fine = minutes(START, 240)
    hole = fine[20]  # 00:20, корзина 00:15
    native.write(SYMBOL, "1m", [c for c in fine if c is not hole])
    report = resample(stores)
    assert (report.built, report.incomplete, report.registered) == (15, 1, 1)
    assert START + TF not in derived.timestamps(SYMBOL, "15m")
    assert set(derived.confirmed_missing(SYMBOL, "15m")) == {START + TF}
    assert any("1 корзин не собраны" in line for line in render_resample(report))
    native.write(SYMBOL, "1m", [hole])  # sync закрыл дыру
    again = resample(stores)
    assert (again.built, again.incomplete) == (16, 0)
    assert derived.confirmed_missing(SYMBOL, "15m") == {}  # пометка снялась сама
    assert derived.read(SYMBOL, "15m") == build_15m(fine, START)


def test_resample_is_idempotent(stores: tuple[CandleStore, CandleStore]) -> None:
    native, _ = stores
    native.write(SYMBOL, "1m", minutes(START, 240))
    first = resample(stores)
    second = resample(stores)
    assert first.stats.added == 16
    assert (second.stats.added, second.stats.unchanged, second.stats.files_written) == (0, 16, 0)


def test_resample_period_can_be_limited(stores: tuple[CandleStore, CandleStore]) -> None:
    native, derived = stores
    native.write(SYMBOL, "1m", minutes(START, 240))
    report = resample(stores, start_ms=START + TF + 5, end_ms=START + 5 * TF + 3)
    assert (report.start_ms, report.end_ms) == (START + 2 * TF, START + 5 * TF)
    assert [c.ts_open for c in derived.read(SYMBOL, "15m")] == [
        START + 2 * TF,
        START + 3 * TF,
        START + 4 * TF,
    ]


def test_resample_without_fine_data_does_nothing(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    report = resample(stores)
    assert report.buckets == 0
    assert "нечего собирать" in render_resample(report)[0]


def test_resample_rejects_wrong_arguments(stores: tuple[CandleStore, CandleStore]) -> None:
    native, derived = stores
    with pytest.raises(ValueError, match="не делится"):
        resample_range(native, derived, SYMBOL, "15m", "1m", now_ms=NOW)
    with pytest.raises(ValueError, match="не делится"):
        resample_range(native, derived, SYMBOL, "5m", "7m", now_ms=NOW)
    with pytest.raises(ValueError, match="слой"):
        resample_range(derived, native, SYMBOL, "1m", "15m", now_ms=NOW)


# ---------------------------------------------------------- сравнение и вердикт


def put_both(
    stores: tuple[CandleStore, CandleStore],
    bars: list[Candle],
    tweak: dict[int, Candle] | None = None,
) -> None:
    native, derived = stores
    native.write(SYMBOL, "15m", bars)
    derived.write(SYMBOL, "15m", [(tweak or {}).get(i, b) for i, b in enumerate(bars)])


def bars_15m(count: int) -> list[Candle]:
    return [
        Candle(START + i * TF, 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 5.0) for i in range(count)
    ]


def changed(bar: Candle, **diff: float) -> Candle:
    return Candle(
        bar.ts_open,
        diff.get("o", bar.o),
        diff.get("h", bar.h),
        diff.get("l", bar.l),
        diff.get("c", bar.c),
        diff.get("v", bar.v),
    )


def compare(stores: tuple[CandleStore, CandleStore]) -> LayerComparison:
    return compare_layers(stores[0], stores[1], SYMBOL, "15m")


def test_identical_layers_are_acceptable(stores: tuple[CandleStore, CandleStore]) -> None:
    put_both(stores, bars_15m(500))
    result = compare(stores)
    assert result.compared == 500
    assert result.mismatched_ohlc == 0
    assert result.verdict == "acceptable"
    assert "ресемпл допустим: OHLC совпадают на всех барах" in verdict_text(result)
    assert render_comparison(result)[-1].startswith("ВЕРДИКТ: ресемпл допустим")


def test_rare_exceptions_are_acceptable_but_listed(stores: tuple[CandleStore, CandleStore]) -> None:
    bars = bars_15m(2000)
    put_both(stores, bars, {7: changed(bars[7], c=bars[7].c + 0.01)})  # 1 из 2000 = 0,05 %
    result = compare(stores)
    assert result.mismatched_ohlc == 1
    assert result.ohlc_share < EXCEPTION_SHARE
    assert result.verdict == "acceptable_with_exceptions"
    assert [d.ts_open for d in result.details] == [START + 7 * TF]
    assert result.by_field == {"close": 1}


def test_many_exceptions_are_not_acceptable_and_tell_to_stop(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    bars = bars_15m(2000)
    tweaks = {i: changed(bars[i], h=bars[i].h + 1) for i in (1, 2, 3)}  # 3 из 2000 = 0,15 %
    put_both(stores, bars, tweaks)
    result = compare(stores)
    assert result.ohlc_share >= EXCEPTION_SHARE
    assert result.verdict == "not_acceptable"
    text = verdict_text(result)
    assert "НЕ ДОПУСТИМ" in text
    assert "остановиться и написать мастеру" in text


def test_exactly_one_tenth_of_a_percent_is_not_acceptable(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    bars = bars_15m(1000)
    put_both(stores, bars, {0: changed(bars[0], o=bars[0].o + 1)})  # 1 из 1000 = 0,1 %
    assert compare(stores).verdict == "not_acceptable"  # порог «меньше 0,1 %»


def test_volume_mismatch_is_reported_but_does_not_change_the_verdict(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    bars = bars_15m(300)
    put_both(stores, bars, {i: changed(bars[i], v=bars[i].v + 2) for i in range(0, 300, 3)})
    result = compare(stores)
    assert result.mismatched_ohlc == 0
    assert result.mismatched_volume == 100
    assert result.verdict == "acceptable"
    assert "объём в 100 барах" in render_comparison(result)[1]


def test_each_ohlc_field_is_detected(stores: tuple[CandleStore, CandleStore]) -> None:
    bars = bars_15m(40)
    tweaks = {
        1: changed(bars[1], o=bars[1].o + 0.5),
        2: changed(bars[2], h=bars[2].h + 0.5),
        3: changed(bars[3], l=bars[3].l - 0.5),
        4: changed(bars[4], c=bars[4].c + 0.5),
    }
    put_both(stores, bars, tweaks)
    result = compare(stores)
    assert result.by_field == {"open": 1, "high": 1, "low": 1, "close": 1}
    assert result.mismatched_ohlc == 4
    assert result.max_abs["open"] == pytest.approx(0.5)


def test_float_noise_is_tolerated(stores: tuple[CandleStore, CandleStore]) -> None:
    bars = bars_15m(50)
    put_both(stores, bars, {i: changed(bars[i], v=bars[i].v + 1e-12) for i in range(50)})
    assert compare(stores).verdict == "acceptable"


def test_comparison_counts_bars_present_in_only_one_layer(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    native, derived = stores
    bars = bars_15m(10)
    native.write(SYMBOL, "15m", bars)  # нативные 0..9
    derived.write(SYMBOL, "15m", [b for b in bars if b.ts_open not in {START + 4 * TF}])
    derived.write(SYMBOL, "15m", [Candle(START - TF, 1, 2, 0.5, 1.5, 1)])  # вне общего периода
    result = compare(stores)
    assert result.compared == 9
    assert result.native_only == 1  # дыра в ресемпле внутри общего периода
    assert result.resampled_only == 0


def test_no_overlap_means_nothing_to_verify(stores: tuple[CandleStore, CandleStore]) -> None:
    native, derived = stores
    native.write(SYMBOL, "15m", bars_15m(5))
    derived.write(SYMBOL, "15m", [Candle(START - 10 * TF, 1, 2, 0.5, 1.5, 1)])
    result = compare(stores)
    assert result.verdict == "no_overlap"
    assert "нет общего периода" in verdict_text(result)
    assert "общего периода нет" in render_comparison(result)[0]


def test_resample_then_compare_real_pipeline(stores: tuple[CandleStore, CandleStore]) -> None:
    native, _ = stores
    fine = minutes(START, 600)
    native.write(SYMBOL, "1m", fine)
    native.write(SYMBOL, "15m", build_15m(fine, START))
    resample(stores)
    result = compare(stores)
    assert (result.compared, result.mismatched_ohlc, result.mismatched_volume) == (40, 0, 0)
    assert result.verdict == "acceptable"


def test_markdown_report_has_verdict_and_mismatch_list(
    stores: tuple[CandleStore, CandleStore],
) -> None:
    bars = bars_15m(2000)
    put_both(stores, bars, {5: changed(bars[5], c=bars[5].c + 0.25)})
    comparison = compare(stores)
    fine_report = resample_range(stores[0], stores[1], SYMBOL, "1m", "15m", now_ms=NOW)
    text = render_resample_markdown([fine_report], [comparison], generated_ms=NOW, version="9")
    assert text.startswith("# Отчёт о ресемпле SqueezeLab")
    assert "**Вердикт:** ресемпл допустим с исключениями" in text
    assert "| close | 1 | 0.25 |" in text
    assert "### Список расхождений (первые 1)" in text
    assert "| 2026-11-01 01:15 | close: 106 → 106.25 |" in text


# ------------------------------------------------------------------ читающий API


def test_reader_prefers_native_and_labels_every_candle(tmp_path: Path) -> None:
    reader = CandleReader(tmp_path / "data")
    native_bars = bars_15m(6)[3:]  # нативные только с 3-й свечи
    reader.native.write(SYMBOL, "15m", native_bars)
    derived = [changed(b, c=b.c + 5) for b in bars_15m(6)]  # ресемпл на всех шести, значения другие
    reader.resampled.write(SYMBOL, "15m", derived)

    items = reader.read(SYMBOL, "15m")
    assert [i.src for i in items] == ["resampled"] * 3 + ["native"] * 3
    assert [i.ts_open for i in items] == [START + k * TF for k in range(6)]
    assert items[3].candle == native_bars[0]  # там, где есть нативная, берётся она
    assert items[0].candle == derived[0]
    assert count_by_source(items) == {"native": 3, "resampled": 3}


def test_reader_can_select_a_single_source(tmp_path: Path) -> None:
    reader = CandleReader(tmp_path / "data")
    reader.native.write(SYMBOL, "15m", bars_15m(4)[2:])
    reader.resampled.write(SYMBOL, "15m", bars_15m(4))
    assert [i.src for i in reader.read(SYMBOL, "15m", src="native")] == ["native"] * 2
    assert [i.src for i in reader.read(SYMBOL, "15m", src="resampled")] == ["resampled"] * 4
    assert reader.read("ETH-USDT", "15m") == []
    with pytest.raises(ValueError, match="auto, native или resampled"):
        reader.read(SYMBOL, "15m", src="both")  # type: ignore[arg-type]


def test_reader_range_and_type(tmp_path: Path) -> None:
    reader = CandleReader(tmp_path / "data")
    reader.native.write(SYMBOL, "15m", bars_15m(10))
    items = reader.read(SYMBOL, "15m", START + 2 * TF, START + 5 * TF)
    assert [i.ts_open for i in items] == [START + 2 * TF, START + 3 * TF, START + 4 * TF]
    assert isinstance(items[0], SourcedCandle)
    assert count_by_source([]) == {}


# ------------------------------------------------------------------------------ CLI


@pytest.fixture
def data_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SQUEEZELAB_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def run_cli(*args: str) -> int:
    return main([*args, "--config-dir", str(REPO_CONFIG)])


def seed(root: Path, minutes_count: int = 240, native_15m: bool = True) -> list[Candle]:
    native = CandleStore(root)
    fine = minutes(START, minutes_count)
    native.write(SYMBOL, "1m", fine)
    if native_15m:
        native.write(SYMBOL, "15m", build_15m(fine, START))
    return fine


def test_cli_resample_acceptable(capsys: pytest.CaptureFixture[str], data_env: Path) -> None:
    seed(data_env)
    assert run_cli("data", "resample") == 0
    out = capsys.readouterr().out
    assert "BNB-USDT: ресемпл 15m из 1m" in out
    assert "корзин: 16; собрано полных: 16; неполных (в 1m есть дыры): 0" in out
    assert "нативный 15m против ресемпла: общий период" in out
    assert "ВЕРДИКТ: ресемпл допустим: OHLC совпадают на всех барах" in out
    (report,) = (data_env / "reports").glob("resample_*.md")
    assert f"Отчёт: {report}" in out
    assert len(CandleStore(data_env, layer=RESAMPLED_LAYER).read(SYMBOL, "15m")) == 16


def test_cli_resample_stops_on_inconsistent_data(
    capsys: pytest.CaptureFixture[str], data_env: Path
) -> None:
    fine = seed(data_env, native_15m=False)
    native = CandleStore(data_env)
    bad = [changed(c, c=c.c + 1) for c in build_15m(fine, START)]  # у всех баров другой close
    native.write(SYMBOL, "15m", bad)
    assert run_cli("data", "resample") == 1
    out = capsys.readouterr().out
    assert "РЕСЕМПЛ НЕ ДОПУСТИМ" in out
    assert "остановиться и написать мастеру" in out
    assert list((data_env / "reports").glob("resample_*.md"))  # отчёт записан и при вердикте «нет»


def test_cli_resample_without_native_15m_has_nothing_to_compare(
    capsys: pytest.CaptureFixture[str], data_env: Path
) -> None:
    seed(data_env, native_15m=False)
    assert run_cli("data", "resample") == 0
    out = capsys.readouterr().out
    assert "общего периода нет" in out
    assert "проверить нечего" in out


def test_cli_resample_no_compare(capsys: pytest.CaptureFixture[str], data_env: Path) -> None:
    seed(data_env)
    assert run_cli("data", "resample", "--no-compare") == 0
    assert "ВЕРДИКТ" not in capsys.readouterr().out


def test_cli_resample_without_fine_data_fails(
    capsys: pytest.CaptureFixture[str], data_env: Path
) -> None:
    assert run_cli("data", "resample") == 1
    assert "нет сохранённых 1m" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        (["--to", "1m", "--from", "15m"], "должен делиться"),
        (["--to", "7x"], "7x"),
        (["--symbol", "bnb"], "символ"),
        (["--start", "не-дата"], "не-дата"),
    ],
)
def test_cli_resample_rejects_bad_arguments(
    capsys: pytest.CaptureFixture[str], data_env: Path, extra: list[str], fragment: str
) -> None:
    seed(data_env)
    assert run_cli("data", "resample", *extra) == 2
    assert fragment in capsys.readouterr().err


def test_cli_check_shows_both_layers(capsys: pytest.CaptureFixture[str], data_env: Path) -> None:
    seed(data_env)
    assert run_cli("data", "resample") == 0
    capsys.readouterr()
    assert run_cli("data", "check") == 0
    out = capsys.readouterr().out
    assert "BNB-USDT 15m: 16 свечей" in out
    assert "BNB-USDT 15m [ресемпл]: 16 свечей" in out
    (report,) = sorted((data_env / "reports").glob("quality_*.md"))
    assert "15m [ресемпл]" in report.read_text(encoding="utf-8")

    assert run_cli("data", "check", "--layer", "native") == 0
    assert "[ресемпл]" not in capsys.readouterr().out
    assert run_cli("data", "check", "--layer", "resampled") == 0
    only = capsys.readouterr().out
    assert "[ресемпл]" in only
    assert "BNB-USDT 1m" not in only


def test_cli_check_flags_resampled_gaps_as_warnings(
    capsys: pytest.CaptureFixture[str], data_env: Path
) -> None:
    fine = seed(data_env, native_15m=False)
    native = CandleStore(data_env)
    for path in native.month_files(SYMBOL, "1m"):
        path.unlink()
    native.write(SYMBOL, "1m", [c for c in fine if c.ts_open != START + 20 * MIN])
    assert run_cli("data", "resample", "--no-compare") == 0
    capsys.readouterr()
    assert (
        run_cli("data", "check", "--layer", "resampled") == 0
    )  # дыра в ресемпле только предупреждение
    out = capsys.readouterr().out
    assert "OK с предупреждениями" in out


# ----------------------------------------------------------------- data sync: код 3


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


def exchange_bars(first: int, count: int, skip: set[int]) -> list[Candle]:
    out = []
    for i in range(count):
        ts = first + i * TF
        if ts in skip:
            continue
        v = bar_values(ts, TF)
        out.append(
            Candle(
                ts,
                float(str(v["open"])),
                float(str(v["high"])),
                float(str(v["low"])),
                float(str(v["close"])),
                float(str(v["volume"])),
            )
        )
    return out


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M")


def test_sync_exit_code_three_while_a_young_gap_is_unconfirmed(
    capsys: pytest.CaptureFixture[str], data_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = int(time.time() * 1000)
    young = now - now % TF - 2 * 3_600_000
    first = young - 5 * TF
    CandleStore(data_env).write(SYMBOL, "15m", exchange_bars(first, 11, {young}))
    patch_exchange(monkeypatch, FakeBingX(Behaviour(holes=frozenset({young}))))
    args = ("data", "sync", "--tf", "15m", "--start", iso(first), "--end", iso(young + 6 * TF))
    assert run_cli(*args) == 3
    out = capsys.readouterr().out
    assert "ОСТАЛИСЬ ПРОПУСКИ: 1" in out


def test_sync_exit_code_zero_when_the_gap_is_confirmed_missing(
    capsys: pytest.CaptureFixture[str], data_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = int(time.time() * 1000)
    old = now - now % TF - 3 * 86_400_000
    first = old - 5 * TF
    CandleStore(data_env).write(SYMBOL, "15m", exchange_bars(first, 11, {old}))
    patch_exchange(monkeypatch, FakeBingX(Behaviour(holes=frozenset({old}))))
    args = ("data", "sync", "--tf", "15m", "--start", iso(first), "--end", iso(old + 6 * TF))
    assert run_cli(*args) == 0
    assert "Пропусков не осталось" in capsys.readouterr().out
    assert run_cli(*args) == 0  # повторный запуск: подтверждённое не запрашивается


def test_sync_uses_configured_timeframes_by_default(
    capsys: pytest.CaptureFixture[str], data_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CandleStore(data_env)
    now = int(time.time() * 1000)
    for tf, tf_ms in (
        ("15m", TF),
        ("1m", MIN),
        ("1h", 3_600_000),
        ("4h", 14_400_000),
        ("1d", 86_400_000),
    ):
        end = now - now % tf_ms - tf_ms  # последняя закрытая свеча
        first = end - 5 * tf_ms
        rows = []
        for i in range(6):
            ts = first + i * tf_ms
            v = bar_values(ts, tf_ms)
            rows.append(
                Candle(
                    ts,
                    float(str(v["open"])),
                    float(str(v["high"])),
                    float(str(v["low"])),
                    float(str(v["close"])),
                    float(str(v["volume"])),
                )
            )
        store.write(SYMBOL, tf, rows)
    patch_exchange(monkeypatch, FakeBingX())
    code = run_cli("data", "sync")
    out = capsys.readouterr().out
    assert "Пары: BNB-USDT; таймфреймы: 15m, 1m, 1h, 4h, 1d" in out
    for tf in ("15m", "1m", "1h", "4h", "1d"):
        assert f"BNB-USDT {tf}: докачка пропусков" in out
    assert code in (0, 3)  # 3 только если у фальшивой биржи нет самой свежей закрытой свечи
