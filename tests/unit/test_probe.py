"""Тесты P01-T02 (часть 1): пробные запросы klines. Биржа имитируется фальшивым сервером."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
import pytest

from squeezelab.app.cli import main
from squeezelab.data.bingx_rest import BingXPublicClient
from squeezelab.data.models import tf_to_ms
from squeezelab.data.probe import (
    DEFAULT_ANCHOR,
    DEFAULT_VOLUME_BAR,
    ProbeCase,
    build_cases,
    derive_findings,
    render_results,
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
    assert "оставлены самые поздние" in text
    assert "получено 1440 при 1500" in text
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
    assert "оставлены самые ранние" in text
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
