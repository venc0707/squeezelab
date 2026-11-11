"""Тесты P00-T04: модель Candle, парсинг klines, ретраи публичного клиента, bingx-smoke."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from squeezelab.app.cli import main
from squeezelab.data.bingx_rest import (
    KLINES_PATH,
    BingXAPIError,
    BingXHTTPError,
    BingXParseError,
    BingXPublicClient,
    BingXRequestError,
    parse_klines_data,
    unwrap_envelope,
)
from squeezelab.data.models import (
    Candle,
    detect_order,
    format_ts,
    is_candle_closed,
    is_contiguous,
    sort_candles,
    tf_to_ms,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
SAMPLES = [FIXTURES / "bingx_klines_sample_docs.json"]
if (FIXTURES / "bingx_klines_real.json").is_file():
    SAMPLES.append(FIXTURES / "bingx_klines_real.json")

TF = tf_to_ms("15m")


def _sample_payload() -> object:
    return json.loads(SAMPLES[0].read_text(encoding="utf-8"))


def _candle(ts: int) -> Candle:
    return Candle(ts, 1.0, 2.0, 0.5, 1.5, 10.0)


# --------------------------------------------------------------------- модель


def test_tf_to_ms() -> None:
    assert tf_to_ms("15m") == 900_000
    assert tf_to_ms("1h") == 3_600_000
    assert tf_to_ms("1d") == 86_400_000


@pytest.mark.parametrize("bad", ["", "m", "15", "0m", "15x", "-5m", "1.5h"])
def test_tf_to_ms_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError, match="неизвестный таймфрейм"):
        tf_to_ms(bad)


def test_format_ts() -> None:
    assert format_ts(1794304800000) == "2026-11-10 10:00:00 UTC"


def test_detect_order() -> None:
    assert detect_order([]) == "empty"
    assert detect_order([_candle(1)]) == "single"
    assert detect_order([_candle(1), _candle(2), _candle(3)]) == "asc"
    assert detect_order([_candle(3), _candle(2), _candle(1)]) == "desc"
    assert detect_order([_candle(1), _candle(3), _candle(2)]) == "mixed"
    assert detect_order([_candle(1), _candle(1)]) == "mixed"  # дубль — не порядок


def test_sort_candles_returns_new_ascending_list() -> None:
    original = [_candle(3), _candle(1), _candle(2)]
    result = sort_candles(original)
    assert [c.ts_open for c in result] == [1, 2, 3]
    assert [c.ts_open for c in original] == [3, 1, 2]


def test_is_candle_closed_boundaries() -> None:
    candle = _candle(1_000_000)
    end = 1_000_000 + TF
    assert not is_candle_closed(candle, TF, end - 1)
    assert is_candle_closed(candle, TF, end)
    assert is_candle_closed(candle, TF, end + 1)


def test_is_contiguous() -> None:
    assert is_contiguous([_candle(0), _candle(TF), _candle(2 * TF)], TF)
    assert not is_contiguous([_candle(0), _candle(2 * TF)], TF)
    assert not is_contiguous([_candle(0), _candle(0)], TF)


def test_candle_is_frozen() -> None:
    with pytest.raises(AttributeError):
        _candle(1).ts_open = 2  # type: ignore[misc]


# -------------------------------------------------------------------- парсинг


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_parse_saved_sample(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candles = parse_klines_data(unwrap_envelope(payload))
    assert candles
    for candle in candles:
        assert isinstance(candle.ts_open, int)
        assert candle.h >= max(candle.o, candle.c)
        assert candle.l <= min(candle.o, candle.c)
        assert candle.v >= 0


def test_parse_synthetic_sample_values() -> None:
    candles = parse_klines_data(unwrap_envelope(_sample_payload()))
    assert len(candles) == 5
    assert detect_order(candles) == "desc"  # образец отражает наблюдавшийся порядок
    newest = candles[0]
    assert newest == Candle(1794308400000, 613.75, 614.5, 613.4, 614.1, 1122.33)
    assert is_contiguous(candles, TF)
    assert [c.ts_open for c in sort_candles(candles)] == sorted(c.ts_open for c in candles)


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"a": 1}, "должен быть списком"),
        ([1], "ожидался объект"),
        ([{"time": 1, "open": "1"}], "нет полей"),
        (
            [{"time": 1, "open": "x", "high": "1", "low": "1", "close": "1", "volume": "1"}],
            "не число",
        ),
        (
            [{"time": 1, "open": "nan", "high": "1", "low": "1", "close": "1", "volume": "1"}],
            "не конечное",
        ),
        (
            [{"time": 1.5, "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"}],
            "целым числом",
        ),
        (
            [{"time": -1, "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"}],
            "целым числом",
        ),
    ],
)
def test_parse_rejects_unexpected_shapes(data: object, fragment: str) -> None:
    with pytest.raises(BingXParseError, match=fragment):
        parse_klines_data(data)


def test_unwrap_envelope_errors() -> None:
    with pytest.raises(BingXParseError, match="code"):
        unwrap_envelope([1, 2])
    with pytest.raises(BingXAPIError) as exc:
        unwrap_envelope({"code": 100400, "msg": "bad symbol"})
    assert exc.value.code == 100400
    with pytest.raises(BingXParseError, match="нет поля data"):
        unwrap_envelope({"code": 0})


# ---------------------------------------------------------------------- клиент


def _client(
    handler: httpx.MockTransport, sleeps: list[float], retries: int = 3
) -> BingXPublicClient:
    return BingXPublicClient(
        "https://example.invalid",
        max_retries=retries,
        backoff_base=0.5,
        transport=handler,
        sleep=sleeps.append,
    )


def test_fetch_klines_request_shape_and_result() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json=_sample_payload(), headers={"X-RateLimit-Remaining": "499", "Server": "x"}
        )

    with _client(httpx.MockTransport(handler), []) as client:
        result = client.fetch_klines("BNB-USDT", "15m", 5)
    request = seen[0]
    assert request.method == "GET"
    assert request.url.path == KLINES_PATH
    assert dict(request.url.params) == {"symbol": "BNB-USDT", "interval": "15m", "limit": "5"}
    assert "squeezelab/" in request.headers["user-agent"]
    assert len(result.candles) == 5
    assert result.limit_headers == {"x-ratelimit-remaining": "499"}
    assert json.loads(result.raw_text)["code"] == 0


def test_time_range_params() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"code": 0, "data": []})

    with _client(httpx.MockTransport(handler), []) as client:
        client.get_klines("BNB-USDT", "15m", 10, start_time=100, end_time=200)
    assert seen[0].url.params["startTime"] == "100"
    assert seen[0].url.params["endTime"] == "200"


def test_retries_5xx_with_growing_pause_then_succeeds() -> None:
    statuses = iter([500, 503, 429, 200])

    def handler(_request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, json=_sample_payload())
        return httpx.Response(status, text="busy")

    sleeps: list[float] = []
    with _client(httpx.MockTransport(handler), sleeps) as client:
        assert len(client.get_klines("BNB-USDT", "15m")) == 5
    assert sleeps == [0.5, 1.0, 2.0]


def test_retry_after_header_is_respected_and_capped() -> None:
    answers = iter(
        [
            httpx.Response(429, headers={"Retry-After": "3"}, text="slow"),
            httpx.Response(429, headers={"Retry-After": "9999"}, text="slow"),
            httpx.Response(200, json=_sample_payload()),
        ]
    )
    sleeps: list[float] = []
    with _client(httpx.MockTransport(lambda _r: next(answers)), sleeps) as client:
        client.get_klines("BNB-USDT", "15m")
    assert sleeps == [3.0, 60.0]


def test_gives_up_after_max_retries_on_5xx() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    sleeps: list[float] = []
    client = _client(httpx.MockTransport(handler), sleeps, retries=2)
    with client, pytest.raises(BingXHTTPError) as exc:
        client.get_klines("BNB-USDT", "15m")
    assert exc.value.status == 502
    assert calls == 3
    assert len(sleeps) == 2


def test_client_errors_are_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad request")

    with _client(httpx.MockTransport(handler), []) as client, pytest.raises(BingXHTTPError):
        client.get_klines("BNB-USDT", "15m")
    assert calls == 1


def test_network_errors_are_retried_then_reported() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("dns failure")

    sleeps: list[float] = []
    client = _client(httpx.MockTransport(handler), sleeps, retries=2)
    with client, pytest.raises(BingXRequestError, match="ConnectError"):
        client.get_klines("BNB-USDT", "15m")
    assert calls == 3
    assert sleeps == [0.5, 1.0]


def test_api_level_error_is_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"code": 100400, "msg": "symbol not exist"})

    with _client(httpx.MockTransport(handler), []) as client, pytest.raises(BingXAPIError):
        client.get_klines("NOPE-USDT", "15m")
    assert calls == 1


def test_non_json_body_is_parse_error() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, text="not json"))
    with _client(transport, []) as client, pytest.raises(BingXParseError, match="не JSON"):
        client.get_klines("BNB-USDT", "15m")


def test_argument_validation() -> None:
    with _client(httpx.MockTransport(lambda _r: httpx.Response(200)), []) as client:
        with pytest.raises(ValueError, match="интервал"):
            client.get_klines("BNB-USDT", "7m")
        with pytest.raises(ValueError, match="limit"):
            client.get_klines("BNB-USDT", "15m", 0)


# ------------------------------------------------------------------ bingx-smoke


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: httpx.MockTransport) -> None:
    original = BingXPublicClient.__init__

    def patched(self: BingXPublicClient, base_url: str, **kwargs: object) -> None:
        original(self, base_url, transport=handler, sleep=lambda _s: None)

    monkeypatch.setattr(BingXPublicClient, "__init__", patched)


def test_smoke_prints_report_and_saves_raw(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(
        monkeypatch, httpx.MockTransport(lambda _r: httpx.Response(200, json=_sample_payload()))
    )
    raw = tmp_path / "out" / "raw.json"
    code = main(["bingx-smoke", "--config-dir", str(REPO_CONFIG), "--save-raw", str(raw)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Получено свечей: 5" in out
    assert "по убыванию ts_open" in out
    assert "закрыта: да" in out  # образец в прошлом относительно текущего времени
    assert "Без пропусков и дублей: да" in out
    assert "Символ: BNB-USDT, интервал: 15m" in out
    assert json.loads(raw.read_text(encoding="utf-8"))["code"] == 0


def test_smoke_flags_unclosed_newest_candle(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    now_ms = int(time.time() * 1000)
    forming_open = now_ms - now_ms % TF  # текущая, ещё не закрытая 15-минутка
    payload = {
        "code": 0,
        "data": [
            {
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "volume": "3",
                "time": forming_open,
            },
            {
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "volume": "3",
                "time": forming_open - TF,
            },
        ],
    }
    _patch_client(monkeypatch, httpx.MockTransport(lambda _r: httpx.Response(200, json=payload)))
    code = main(["bingx-smoke", "--config-dir", str(REPO_CONFIG)])
    assert code == 0
    assert "НЕТ, ещё формируется" in capsys.readouterr().out


def test_smoke_reports_api_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_client(
        monkeypatch,
        httpx.MockTransport(lambda _r: httpx.Response(200, json={"code": 100400, "msg": "bad"})),
    )
    code = main(["bingx-smoke", "--config-dir", str(REPO_CONFIG)])
    assert code == 1
    assert "code=100400" in capsys.readouterr().err
