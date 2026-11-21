"""Тесты P01-T02 (часть 1): пробные запросы klines. Биржа имитируется фальшивым сервером."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
import pytest
from fake_bingx import Behaviour, FakeBingX

from squeezelab.app.cli import main
from squeezelab.data.bingx_rest import BingXPublicClient
from squeezelab.data.models import tf_to_ms
from squeezelab.data.probe import (
    COVERAGE_DAYS,
    DEFAULT_ANCHOR,
    DEFAULT_VOLUME_BAR,
    DEPTH_INTERVALS,
    ProbeCase,
    build_cases,
    build_paging_cases,
    coverage_samples,
    derive_coverage_findings,
    derive_depth_findings,
    derive_findings,
    derive_paging_findings,
    render_coverage,
    render_depth,
    render_results,
    run_coverage_probe,
    run_depth_probe,
    run_probe,
    to_ms,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
ANCHOR = to_ms(DEFAULT_ANCHOR)
VOLUME_BAR = to_ms(DEFAULT_VOLUME_BAR)
HISTORY_START = to_ms(DEFAULT_ANCHOR.replace(year=2020))  # «листинг» фальшивой биржи


@dataclass(frozen=True)
class Semantics:
    """Как ведёт себя фальшивая биржа (варианты, которые мы хотим уметь различать)."""

    start_inclusive: bool = True
    end_inclusive: bool = True
    align_start_down: bool = False  # start округляется вниз до начала свечи, в которую он попал
    truncate: Literal["earliest", "latest"] = "latest"  # что остаётся при окне шире limit
    start_only: Literal["earliest", "latest"] = "earliest"  # что отдаётся при одном startTime
    order: Literal["desc", "asc"] = "desc"


def fake_exchange(sem: Semantics, now_ms: int | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.params
        tf = tf_to_ms(q["interval"])
        limit = int(q["limit"])
        start = int(q["startTime"]) if "startTime" in q else None
        end = int(q["endTime"]) if "endTime" in q else None
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        newest = now - now % tf  # самая новая (формирующаяся) свеча

        if end is None:
            last = newest
        else:
            last = min(end - end % tf, newest)
            if not sem.end_inclusive and end % tf == 0:
                last -= tf
        if start is None:
            first = HISTORY_START
        elif sem.align_start_down:
            first = start - start % tf
        else:
            first = -(-start // tf) * tf  # первая граница >= start
            if not sem.start_inclusive and first == start:
                first += tf
        first = max(first, HISTORY_START)

        count = (last - first) // tf + 1 if last >= first else 0
        if count > limit:
            keep_latest = (
                sem.start_only == "latest"
                if (start is not None and end is None)
                else sem.truncate == "latest"
                if start is not None
                else True
            )
            if keep_latest:
                first = last - (limit - 1) * tf
            else:
                last = first + (limit - 1) * tf
        opens = list(range(first, last + 1, tf)) if count > 0 else []
        if sem.order == "desc":
            opens.reverse()
        data = [
            {
                "open": "1",
                "close": "1.5",
                "high": "2",
                "low": "0.5",
                "volume": "167.58" if t_open == VOLUME_BAR else "1",
                "time": t_open,
            }
            for t_open in opens
        ]
        return httpx.Response(200, json={"code": 0, "msg": "", "data": data})

    return httpx.MockTransport(handler)


def run_all(sem: Semantics) -> tuple[list[str], list[str]]:
    now_ms = int(time.time() * 1000)
    with BingXPublicClient(transport=fake_exchange(sem, now_ms), sleep=lambda _s: None) as client:
        results = run_probe(
            client, "BNB-USDT", build_cases(ANCHOR, VOLUME_BAR), sleep=lambda _s: None
        )
    return render_results(results), derive_findings(results, ANCHOR, now_ms, VOLUME_BAR)


# ------------------------------------------------------------------------------ наборы


def test_cases_are_aligned_and_use_valid_limits() -> None:
    cases = build_cases(ANCHOR, VOLUME_BAR)
    assert [c.key for c in cases] == ["A", "B", "C", "D", "E", "F", "G", "H", "I1", "I2", "J", "K"]
    assert all(1 <= c.limit <= 1440 for c in cases)
    a = next(c for c in cases if c.key == "A")
    assert a.start == ANCHOR
    assert a.end == ANCHOR + 4 * 900_000
    j = next(c for c in cases if c.key == "J")
    assert j.pick_raw_ts == VOLUME_BAR


def test_default_anchor_is_old_enough_for_wide_window() -> None:
    wide = ANCHOR + 1500 * 900_000
    assert wide < int(time.time() * 1000)


# ------------------------------------------------------------------------------ выводы


def test_findings_for_inclusive_exchange() -> None:
    lines, findings = run_all(Semantics())
    text = "\n".join(findings)
    assert "startTime включительно по времени открытия: ДА" in text
    assert "endTime включительно по времени открытия: ДА" in text
    assert "всего 5 из 5" in text
    assert "start=T-1 мс: предыдущая свеча не попала, нужная вернулась" in text
    assert "отбор по времени открытия >= start (C)" in text
    assert "САМЫЕ РАННИЕ свечи от start (D)" in text
    assert "остались САМЫЕ НОВЫЕ, ранние потеряны" in text
    assert "получено 1440 из 1501" in text
    assert "(G:" in text
    assert "без времён: самая новая свеча" in text
    assert "UTC UTC" not in text
    assert "1m окно 00:00..00:09: получено 10" in text
    assert "пересечений 0, пропусков на стыке и внутри 0" in text
    assert "порядок свечей во всех ответах: по убыванию ts_open" in text
    assert any('"volume": "167.58"' in line for line in lines)


def test_findings_detect_exclusive_end_and_aligned_start() -> None:
    _, findings = run_all(Semantics(end_inclusive=False, align_start_down=True))
    text = "\n".join(findings)
    assert "endTime включительно по времени открытия: НЕТ" in text
    assert "всего 4 из 5" in text
    assert "округляет startTime вниз до начала свечи" in text


def test_findings_detect_exclusive_start() -> None:
    _, findings = run_all(Semantics(start_inclusive=False))
    assert "startTime включительно по времени открытия: НЕТ" in "\n".join(findings)


def test_findings_detect_other_truncation_and_start_only() -> None:
    _, findings = run_all(Semantics(truncate="earliest", start_only="latest", order="asc"))
    text = "\n".join(findings)
    assert "остались САМЫЕ РАННИЕ, новые потеряны" in text
    assert "САМЫЕ НОВЫЕ свечи, start лишь нижняя граница (D)" in text
    assert "порядок свечей во всех ответах: по возрастанию ts_open" in text


def test_forming_candle_is_flagged() -> None:
    now_ms = int(time.time() * 1000)
    # Фальшивая биржа отдаёт формирующуюся свечу при запросе без времён (как и настоящая).
    with BingXPublicClient(
        transport=fake_exchange(Semantics(), now_ms), sleep=lambda _s: None
    ) as client:
        results = run_probe(
            client,
            "BNB-USDT",
            [ProbeCase("F", "без времён", "15m", 3)],
            sleep=lambda _s: None,
        )
    findings = derive_findings(results, ANCHOR, now_ms, VOLUME_BAR)
    assert any("ещё ФОРМИРУЕТСЯ (отбрасывать)" in line for line in findings)


def test_api_error_in_one_case_does_not_stop_the_rest() -> None:
    base = fake_exchange(Semantics())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("limit") == "5":
            return httpx.Response(200, json={"code": 109400, "msg": "bad params"})
        return base.handle_request(request)

    with BingXPublicClient(transport=httpx.MockTransport(handler), sleep=lambda _s: None) as client:
        results = run_probe(
            client, "BNB-USDT", build_cases(ANCHOR, VOLUME_BAR), sleep=lambda _s: None
        )
    failed = [r.case.key for r in results if r.candles is None]
    assert failed == ["D", "E"]
    lines = render_results(results)
    assert any("ОШИБКА" in line and "109400" in line for line in lines)
    findings = derive_findings(results, ANCHOR, int(time.time() * 1000), VOLUME_BAR)
    assert any("запросы с ошибкой: D, E" in line for line in findings)
    assert any("startTime включительно" in line for line in findings)


def test_pauses_between_requests() -> None:
    pauses: list[float] = []
    with BingXPublicClient(transport=fake_exchange(Semantics()), sleep=lambda _s: None) as client:
        cases = build_cases(ANCHOR, VOLUME_BAR)
        run_probe(client, "BNB-USDT", cases, sleep=pauses.append)
    assert len(pauses) == len(cases) - 1


# --------------------------------------------------------------------------------- CLI


def _patch_client(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    original = BingXPublicClient.__init__

    def patched(self: BingXPublicClient, base_url: str, **kwargs: object) -> None:
        original(self, base_url, transport=transport, sleep=lambda _s: None)

    monkeypatch.setattr(BingXPublicClient, "__init__", patched)
    monkeypatch.setattr(time, "sleep", lambda _s: None)


def test_cli_probe_prints_report(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(monkeypatch, fake_exchange(Semantics()))
    code = main(["bingx-probe-klines", "--config-dir", str(REPO_CONFIG)])
    out = capsys.readouterr().out
    assert code == 0
    assert "символ BNB-USDT, запросов: 12" in out
    assert "[J] свеча для сверки volume с TradingView" in out
    assert '"volume": "167.58"' in out
    assert "Выводы по этим запросам:" in out
    assert "startTime включительно по времени открытия: ДА" in out


def test_cli_probe_rejects_unaligned_or_recent_anchor(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(monkeypatch, fake_exchange(Semantics()))
    base = ["bingx-probe-klines", "--config-dir", str(REPO_CONFIG)]
    assert main([*base, "--anchor", "2026-11-01T00:07"]) == 2
    assert "границе 15 минут" in capsys.readouterr().err
    recent = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H:00")
    assert main([*base, "--anchor", recent]) == 2
    assert "слишком близко" in capsys.readouterr().err
    assert main([*base, "--anchor", "не-дата"]) == 2


def test_cli_probe_fails_when_every_request_fails(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = httpx.MockTransport(
        lambda _r: httpx.Response(200, json={"code": 100400, "msg": "x"})
    )
    _patch_client(monkeypatch, transport)
    code = main(["bingx-probe-klines", "--config-dir", str(REPO_CONFIG)])
    assert code == 1
    assert "запросы с ошибкой" in capsys.readouterr().out


# ---------------------------------------------- поведение настоящей биржи (проба 2026-11-20)

LISTING = to_ms(datetime(2022, 8, 1, tzinfo=UTC))  # «листинг» в фальшивой бирже


def observed_exchange(now_ms: int | None = None) -> FakeBingX:
    return FakeBingX(Behaviour(history_start=LISTING), now_ms or int(time.time() * 1000))


def run_set(exchange: FakeBingX, cases: list[ProbeCase]) -> list:  # type: ignore[type-arg]
    with BingXPublicClient(transport=exchange.transport, sleep=lambda _s: None) as client:
        return run_probe(client, "BNB-USDT", cases, sleep=lambda _s: None)


def test_basic_findings_reproduce_the_real_probe_output() -> None:
    """На модели наблюдённого поведения выводы те же, что напечатал настоящий BingX."""
    exchange = observed_exchange()
    results = run_set(exchange, build_cases(ANCHOR, VOLUME_BAR))
    text = "\n".join(derive_findings(results, ANCHOR, exchange.now_ms, VOLUME_BAR))
    assert "startTime включительно по времени открытия: ДА" in text
    assert "endTime включительно по времени открытия: НЕТ" in text
    assert "всего 4 из 5" in text
    assert "отбор по времени открытия >= start (C)" in text
    assert "САМЫЕ РАННИЕ свечи от start (D)" in text
    assert "свеча с временем end ВХОДИТ" in text  # E: endTime без startTime включительно
    assert "ещё ФОРМИРУЕТСЯ (отбрасывать)" in text
    assert "получено 1000 из 1500 в окне" in text
    assert "остались САМЫЕ НОВЫЕ, ранние потеряны" in text
    assert "у биржи есть свой предел ответа" in text
    assert "1m окно 00:00..00:09: получено 9" in text
    assert "пересечений 0, пропусков на стыке и внутри 1" in text
    assert "порядок свечей во всех ответах: по убыванию ts_open" in text
    assert "start=2019-01-01" not in text  # K пуст: вывода по нему нет, как в настоящей пробе


def test_paging_set_has_expected_cases() -> None:
    cases = build_paging_cases(ANCHOR)
    keys = [c.key for c in cases]
    assert keys[:12] == [
        "L1",
        "L2",
        "L3",
        "W1000",
        "W1001",
        "W1440",
        "W1500",
        "E2",
        "E3",
        "X1",
        "M1",
        "M2",
    ]
    assert keys[12:] == ["R2021", "R2022", "R2023", "R2024", "R2025"]
    assert all(1 <= c.limit <= 1440 for c in cases)
    assert max(c.end for c in cases if c.end) < int(time.time() * 1000)  # окна целиком в прошлом


def test_paging_findings_on_observed_behaviour() -> None:
    exchange = observed_exchange()
    results = run_set(exchange, build_paging_cases(ANCHOR))
    text = "\n".join(derive_paging_findings(results, ANCHOR, exchange.now_ms))
    assert "15m без времён, limit 1440: получено 1000" in text
    assert "окно 1000 баров, limit 1000: получено 1000" in text
    assert "окно 1001 бар, limit 1440: получено 1000" in text
    assert "1m окно 1440 баров, limit 1440: получено 1440" in text
    assert "ПУСТОЙ ответ" not in text  # в фальшивой бирже 2023 год уже не пуст
    assert "наибольший ответ при limit=1440: 15m: 1000 свечей; 1m: 1440 свечей" in text
    assert "потеряны ранние свечи (W1001)" in text
    assert "endTime на 1 мс раньше границы: самая новая свеча 2026-11-02 01:00 UTC (E2)" in text
    assert "endTime на 1 мс позже границы: самая новая свеча 2026-11-02 01:15 UTC (E3)" in text
    assert "endTime округляется ВВЕРХ до границы свечи" in text
    assert "свеча на границе ВХОДИТ; получено 5 (X1)" in text
    assert "2021: пусто" in text
    assert "2022: пусто" in text
    assert "2023: 2023-01-01 00:00 UTC" in text


def test_paging_findings_when_cap_is_1440() -> None:
    exchange = FakeBingX(Behaviour(cap=1440, history_start=LISTING))
    results = run_set(exchange, build_paging_cases(ANCHOR))
    text = "\n".join(derive_paging_findings(results, ANCHOR, exchange.now_ms))
    assert "наибольший ответ при limit=1440: 15m: 1440 свечей; 1m: 1440 свечей" in text
    assert "окно 1440 баров, limit 1440: получено 1440" in text
    assert "потеряны" not in text


def test_cli_probe_paging_set(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(monkeypatch, observed_exchange().transport)
    code = main(["bingx-probe-klines", "--set", "paging", "--config-dir", str(REPO_CONFIG)])
    out = capsys.readouterr().out
    assert code == 0
    assert "запросов: 17" in out
    assert "[W1001]" in out
    assert "наибольший ответ при limit=1440: 15m: 1000 свечей; 1m: 1440 свечей" in out


def test_cli_probe_paging_needs_older_anchor(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(monkeypatch, observed_exchange().transport)
    anchor = (datetime.now(UTC) - timedelta(days=18)).strftime("%Y-%m-%dT%H:00")
    assert main(["bingx-probe-klines", "--set", "paging", "--anchor", anchor]) == 2
    assert "слишком близко" in capsys.readouterr().err


# ------------------------------------------------------------------- набор depth


def run_depth(listing: int, **kwargs: object) -> tuple[list, FakeBingX]:  # type: ignore[type-arg]
    exchange = FakeBingX(Behaviour(history_start=listing), **kwargs)  # type: ignore[arg-type]
    with BingXPublicClient(transport=exchange.transport, sleep=lambda _s: None) as client:
        results = run_depth_probe(client, "BNB-USDT", ANCHOR, sleep=lambda _s: None)
    return results, exchange


def test_depth_probe_finds_every_interval_and_measures_caps() -> None:
    listing = to_ms(datetime(2025, 3, 9, tzinfo=UTC))
    results, exchange = run_depth(listing)
    assert [r.interval for r in results] == list(DEPTH_INTERVALS)
    for r in results:
        assert r.history is not None
        assert r.history.first_ts == listing
        assert r.history.reason == "found"
    caps = {r.interval: r.newest_count for r in results}
    assert caps["15m"] == 1000
    assert caps["1m"] == 1440
    text = "\n".join(derive_depth_findings(results, exchange.now_ms))
    assert "15m: самая ранняя свеча 2025-03-09 00:00 UTC" in text
    assert "ДОСТАТОЧНО" in text  # 2025-03-09 .. 2026-11: больше 565 суток
    assert "начало 1m относительно 15m: +0.0 суток" in text
    assert "1d: самая ранняя свеча 2025-03-09 00:00 UTC" in text
    assert "15m: 1000; 1m: 1440" in text


def test_depth_findings_say_when_history_is_too_short() -> None:
    listing = to_ms(datetime(2026, 4, 1, tzinfo=UTC))
    results, exchange = run_depth(listing)
    text = "\n".join(derive_depth_findings(results, exchange.now_ms))
    assert "НЕДОСТАТОЧНО: доступно" in text
    assert "ДОСТАТОЧНО" not in text.replace("НЕДОСТАТОЧНО", "")


def test_depth_report_lists_steps_and_first_candle() -> None:
    listing = to_ms(datetime(2025, 3, 9, tzinfo=UTC))
    results, _ = run_depth(listing)
    lines = render_depth(results)
    assert "[15m] начало истории (бисекция по startTime, limit=1)" in lines
    assert any("самая ранняя свеча: 2025-03-09 00:00 UTC" in line for line in lines)
    assert any("без времён, limit=1440: получено 1000" in line for line in lines)


def test_depth_probe_survives_an_error_in_one_interval() -> None:
    listing = to_ms(datetime(2025, 3, 9, tzinfo=UTC))
    exchange = FakeBingX(Behaviour(history_start=listing), fail_after=5)
    with BingXPublicClient(transport=exchange.transport, sleep=lambda _s: None) as client:
        results = run_depth_probe(
            client, "BNB-USDT", ANCHOR, intervals=("15m", "1d"), sleep=lambda _s: None
        )
    assert results[0].history is None
    assert results[0].error is not None
    assert "ОШИБКА" in "\n".join(render_depth(results))
    assert "15m: ошибка запроса" in "\n".join(derive_depth_findings(results, exchange.now_ms))


def test_cli_probe_depth_set(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = to_ms(datetime(2025, 3, 9, tzinfo=UTC))
    _patch_client(monkeypatch, FakeBingX(Behaviour(history_start=listing)).transport)
    code = main(["bingx-probe-klines", "--set", "depth", "--config-dir", str(REPO_CONFIG)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Поиск начала истории" in out
    assert "[1d] начало истории" in out
    assert "цель по глубине 15m (565 суток: 12 мес. + 200 суток прогрева): ДОСТАТОЧНО" in out


def test_cli_probe_depth_fails_when_every_request_fails(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(monkeypatch, FakeBingX(fail_after=0).transport)
    code = main(["bingx-probe-klines", "--set", "depth", "--config-dir", str(REPO_CONFIG)])
    assert code == 1
    assert "ошибка запроса" in capsys.readouterr().out


# ---------------------------------------------------------------------- набор coverage

DAY = 86_400_000
NOW_FOR_COVERAGE = to_ms(datetime(2026, 11, 21, 17, 38, tzinfo=UTC))


def run_coverage(exchange: FakeBingX):  # type: ignore[no-untyped-def]
    with BingXPublicClient(transport=exchange.transport, sleep=lambda _s: None) as client:
        return run_coverage_probe(client, "BNB-USDT", exchange.now_ms, sleep=lambda _s: None)


def test_coverage_samples_are_full_past_days_and_include_default_start() -> None:
    samples = coverage_samples(NOW_FOR_COVERAGE)
    assert (
        len(samples) == 59 * 7 + 2
    )  # 59 месяцев (2022-01 … 2026-11), 7 образцов в месяц, 2 добавочных
    assert all(day + DAY <= NOW_FOR_COVERAGE - NOW_FOR_COVERAGE % DAY for _, day in samples)
    assert ("15m", to_ms(datetime(2026, 6, 25, tzinfo=UTC))) in samples
    assert ("1m", to_ms(datetime(2026, 6, 25, tzinfo=UTC))) in samples
    assert ("1d", to_ms(datetime(2026, 11, 1, tzinfo=UTC))) in samples
    assert ("15m", to_ms(datetime(2026, 11, 15, tzinfo=UTC))) in samples
    assert not any(day >= to_ms(datetime(2026, 11, 21, tzinfo=UTC)) for _, day in samples)
    assert set(COVERAGE_DAYS) == {"15m", "1m", "1h", "4h", "1d"}


def test_coverage_samples_skip_days_that_have_not_ended() -> None:
    now = to_ms(datetime(2026, 11, 15, 12, tzinfo=UTC))  # сутки 15-го ещё идут
    samples = coverage_samples(now)
    assert ("15m", to_ms(datetime(2026, 11, 15, tzinfo=UTC))) not in samples
    assert ("15m", to_ms(datetime(2026, 11, 1, tzinfo=UTC))) in samples


def test_coverage_finds_where_data_begins_and_where_it_is_solid() -> None:
    listing = to_ms(datetime(2025, 3, 9, tzinfo=UTC))
    exchange = FakeBingX(Behaviour(history_start=listing), now_ms=NOW_FOR_COVERAGE)
    cells = run_coverage(exchange)
    text = "\n".join(derive_coverage_findings(cells, exchange.now_ms))
    assert "15m: первые свечи в образцах 2025-03-15; сплошное покрытие с 2025-03-15" in text
    assert "1d: первые свечи в образцах 2025-04-01" in text
    assert "пустых ПОСЛЕ первых данных" not in text
    by_key = {(c.interval, c.day_ms): c for c in cells}
    full_day = by_key[("15m", to_ms(datetime(2026, 6, 25, tzinfo=UTC)))]
    assert (full_day.got, full_day.expected) == (96, 96)
    one_minute_day = by_key[("1m", to_ms(datetime(2026, 6, 25, tzinfo=UTC)))]
    assert (one_minute_day.got, one_minute_day.expected) == (1440, 1440)
    before = by_key[("15m", to_ms(datetime(2025, 3, 1, tzinfo=UTC)))]
    assert before.got == 0


def test_coverage_reports_partial_days_and_gaps_after_first_data() -> None:
    tf = 900_000
    partial_day = to_ms(datetime(2025, 6, 1, tzinfo=UTC))
    empty_day = to_ms(datetime(2025, 7, 15, tzinfo=UTC))
    holes = {partial_day + i * tf for i in range(10, 96)}  # от суток остаётся 10 свечей
    holes |= {empty_day + i * tf for i in range(96)}  # сутки целиком пусты
    exchange = FakeBingX(
        Behaviour(history_start=to_ms(datetime(2025, 3, 9, tzinfo=UTC)), holes=frozenset(holes)),
        now_ms=NOW_FOR_COVERAGE,
    )
    text = "\n".join(derive_coverage_findings(run_coverage(exchange), exchange.now_ms))
    assert "частичных 1: 2025-06-01 10/96" in text
    assert "пустых ПОСЛЕ первых данных 1: 2025-07-15" in text
    # «сплошное покрытие» начинается только после последнего дефекта
    assert "сплошное покрытие с 2025-08-01" in text


def test_coverage_table_has_one_row_per_month_plus_header_and_extras() -> None:
    exchange = FakeBingX(
        Behaviour(history_start=to_ms(datetime(2025, 3, 9, tzinfo=UTC))), now_ms=NOW_FOR_COVERAGE
    )
    lines = render_coverage(run_coverage(exchange))
    assert len(lines) == 1 + 59 + 2
    assert lines[0].startswith("месяц")
    assert "2025-03" in lines[1 + 38]  # 2022-01 + 38 месяцев
    assert "0/96" in lines[1 + 38]
    assert "96/96" in lines[-3 - 1 + 1] or any("96/96" in line for line in lines[-10:])
    assert lines[-2].startswith("2026-06-25 (начало периода по умолчанию) 15m: 96/96")
    assert lines[-1].startswith("2026-06-25 (начало периода по умолчанию) 1m: 1440/1440")


def test_cli_probe_coverage_set(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    exchange = FakeBingX(Behaviour(history_start=to_ms(datetime(2025, 3, 9, tzinfo=UTC))))
    _patch_client(monkeypatch, exchange.transport)
    code = main(["bingx-probe-klines", "--set", "coverage", "--config-dir", str(REPO_CONFIG)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Карта наличия данных" in out
    assert "месяц" in out
    assert "15m: первые свечи в образцах 2025-03-15" in out


def test_cli_probe_coverage_fails_when_every_request_fails(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(monkeypatch, FakeBingX(fail_after=0).transport)
    code = main(["bingx-probe-klines", "--set", "coverage", "--config-dir", str(REPO_CONFIG)])
    assert code == 1
    assert "запросов с ошибкой" in capsys.readouterr().out
