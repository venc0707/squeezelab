"""Тесты P00-T05: подпись запросов, read-only клиент, bingx-auth-check (сеть не используется)."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import httpx
import pytest
from pydantic import SecretStr

from squeezelab.app.cli import main
from squeezelab.data.bingx_rest import (
    API_PERMISSIONS_PATH,
    BALANCE_PATH,
    LIVE_BASE_URL,
    VST_BASE_URL,
    BingXAPIError,
    BingXHTTPError,
    BingXParseError,
    BingXReadOnlyClient,
    BingXRequestError,
    describe_shape,
    extract_permissions,
    permission_fields,
    permissions_differ_from_reference,
    risky_permissions,
    summarize_balance,
)
from squeezelab.data.bingx_signing import API_KEY_HEADER, build_signed_query, sign

KEY = "api-key-ABCDEFGH1234"
SECRET = "test-secret-zyxwvu9876"
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
TS = 1_700_000_000_000


# ------------------------------------------------------------------- подпись


def test_known_answer_vectors_from_openssl() -> None:
    # Эталоны получены независимо: printf '<строка>' | openssl dgst -sha256 -hmac test-secret
    assert (
        sign("recvWindow=5000&timestamp=1700000000000", "test-secret")
        == "e80444d3300edcb80b05d266439eb51c0f9551b00a09836c26b05dea9af0eba3"
    )
    assert (
        sign("recvWindow=5000&symbol=BNB-USDT&timestamp=1700000000000", "test-secret")
        == "898c788fa7463d0ce19f78a835b658967055efacf9ff1b9cdd6159f86cdf4884"
    )


def test_build_signed_query_balance_request() -> None:
    query = build_signed_query({}, "test-secret", timestamp_ms=TS)
    assert query == (
        "recvWindow=5000&timestamp=1700000000000"
        "&signature=e80444d3300edcb80b05d266439eb51c0f9551b00a09836c26b05dea9af0eba3"
    )


def test_params_are_sorted_ascii_and_signature_last() -> None:
    query = build_signed_query({"symbol": "BNB-USDT"}, "test-secret", timestamp_ms=TS)
    assert query.startswith("recvWindow=5000&symbol=BNB-USDT&timestamp=1700000000000&signature=")
    assert query.endswith("898c788fa7463d0ce19f78a835b658967055efacf9ff1b9cdd6159f86cdf4884")


def test_recv_window_can_be_omitted() -> None:
    query = build_signed_query({}, "s", timestamp_ms=TS, recv_window_ms=None)
    assert query.startswith("timestamp=1700000000000&signature=")
    assert "recvWindow" not in query


def test_signature_depends_on_secret_and_timestamp() -> None:
    a = build_signed_query({}, "one", timestamp_ms=TS)
    assert a != build_signed_query({}, "two", timestamp_ms=TS)
    assert a != build_signed_query({}, "one", timestamp_ms=TS + 1)


@pytest.mark.parametrize("bad", ["a b", "a&b", "a=b", "a?b", "a#b", "", "a\nb"])
def test_forbidden_value_characters(bad: str) -> None:
    with pytest.raises(ValueError, match="недопустимые символы"):
        build_signed_query({"x": bad}, "s", timestamp_ms=TS)


def test_reserved_names_and_bad_types() -> None:
    for name in ("timestamp", "recvWindow", "signature"):
        with pytest.raises(ValueError, match="автоматически"):
            build_signed_query({name: "1"}, "s", timestamp_ms=TS)
    with pytest.raises(TypeError):
        build_signed_query({"x": 1.5}, "s", timestamp_ms=TS)  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        build_signed_query({"x": True}, "s", timestamp_ms=TS)


# --------------------------------------------------------------- read-only клиент


def _client(handler: httpx.MockTransport, base_url: str = VST_BASE_URL) -> BingXReadOnlyClient:
    return BingXReadOnlyClient(
        SecretStr(KEY),
        SecretStr(SECRET),
        base_url,
        transport=handler,
        now_ms=lambda: TS,
    )


BALANCE_LIST = {
    "code": 0,
    "msg": "",
    "data": [
        {"asset": "VST", "balance": "100000", "equity": "100000", "availableMargin": "99000"},
        {"asset": "USDT", "balance": "0", "equity": "0", "availableMargin": "0"},
    ],
}


def test_signed_get_request_shape_and_valid_signature() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=BALANCE_LIST)

    with _client(httpx.MockTransport(handler)) as client:
        summary = client.fetch_balance()
    request = seen[0]
    assert request.method == "GET"
    assert request.url.path == BALANCE_PATH
    assert request.headers[API_KEY_HEADER] == KEY
    pairs = parse_qsl(urlsplit(str(request.url)).query)
    assert pairs[-1][0] == "signature"
    signed_part = "&".join(f"{k}={v}" for k, v in pairs[:-1])
    expected = hmac.new(SECRET.encode(), signed_part.encode(), hashlib.sha256).hexdigest()
    assert pairs[-1][1] == expected
    assert dict(pairs)["timestamp"] == str(TS)
    assert KEY not in str(request.url)
    assert SECRET not in str(request.url)
    assert summary.assets == ["VST", "USDT"]
    assert summary.shown_asset == "USDT"  # приоритет у USDT, затем VST
    assert summary.fields["availableMargin"] == "0"


def test_each_call_uses_fresh_timestamp() -> None:
    stamps = iter([1000, 2000])
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.query.decode())
        return httpx.Response(200, json=BALANCE_LIST)

    client = BingXReadOnlyClient(
        SecretStr(KEY),
        SecretStr(SECRET),
        VST_BASE_URL,
        transport=httpx.MockTransport(handler),
        now_ms=lambda: next(stamps),
    )
    with client:
        client.fetch_balance()
        client.fetch_balance()
    assert "timestamp=1000" in queries[0]
    assert "timestamp=2000" in queries[1]


def test_only_allowlisted_paths() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"code": 0, "data": {}})

    with _client(httpx.MockTransport(handler)) as client:
        for path in ("/openApi/swap/v2/trade/order", "/openApi/swap/v3/trade/order", "/x"):
            with pytest.raises(ValueError, match="не входит в список"):
                client.signed_get(path)
    assert not called


def test_api_error_is_reported_without_secrets() -> None:
    body = {"code": 100413, "msg": f"Incorrect apiKey {KEY} / {SECRET}"}
    client = _client(httpx.MockTransport(lambda _r: httpx.Response(200, json=body)))
    with client, pytest.raises(BingXAPIError) as exc:
        client.fetch_balance()
    assert exc.value.code == 100413
    assert KEY not in str(exc.value)
    assert SECRET not in str(exc.value)


def test_http_error_masks_body() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(401, text=f"bad key {KEY}"))
    with _client(transport) as client, pytest.raises(BingXHTTPError) as exc:
        client.fetch_balance()
    assert exc.value.status == 401
    assert KEY not in str(exc.value)


def test_network_error_is_not_retried_and_does_not_leak() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError(f"failed {request.url}")

    with _client(httpx.MockTransport(handler)) as client, pytest.raises(BingXRequestError) as exc:
        client.fetch_balance()
    assert calls == 1
    assert "ConnectError" in str(exc.value)
    assert "signature" not in str(exc.value)
    assert SECRET not in str(exc.value)


def test_non_json_response() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, text="<html>"))
    with _client(transport) as client, pytest.raises(BingXParseError, match="не JSON"):
        client.fetch_balance()


# ------------------------------------------------------------- разбор ответов


def test_summarize_balance_shapes() -> None:
    single = summarize_balance({"balance": {"asset": "USDT", "balance": "5", "equity": "6"}})
    assert single.assets == ["USDT"]
    assert single.fields == {"balance": "5", "equity": "6"}
    flat = summarize_balance({"asset": "VST", "availableMargin": "7"})
    assert flat.shown_asset == "VST"
    assert flat.fields == {"availableMargin": "7"}
    unknown_first = summarize_balance([{"asset": "BTC", "balance": "1"}])
    assert unknown_first.shown_asset == "BTC"


@pytest.mark.parametrize("bad", ["text", 5, None, {"x": 1}, []])
def test_summarize_balance_rejects_unknown_shapes(bad: object) -> None:
    with pytest.raises(BingXParseError):
        summarize_balance(bad)


def test_permission_helpers() -> None:
    data = {
        "ipRestrict": True,
        "permitsEnableWithdraw": False,
        "permitsEnableSpotAndMarginTrade": True,
        "permitsEnableFutures": True,
        "permitsEnableReading": True,
        "apiKey": KEY,
    }
    assert risky_permissions(data) == ["permitsEnableSpotAndMarginTrade"]
    assert "apiKey" not in permission_fields(data)
    assert permission_fields(data)["ipRestrict"] is True
    assert risky_permissions({"permitsEnableWithdraw": True}) == ["permitsEnableWithdraw"]
    assert risky_permissions("text") == []
    assert permission_fields(None) == {}


# Реальная форма ответа apiPermissions на VST (проверено запросом 2026-11-13)
REAL_PERMISSIONS = {"permissions": [1], "ipAddresses": [], "note": "my note", "apiKey": KEY}


def test_permission_fields_real_shape() -> None:
    fields = permission_fields(REAL_PERMISSIONS)
    assert fields == {"permissions": [1], "ipAddresses": []}
    assert "note" not in fields
    assert "apiKey" not in fields
    assert permission_fields({"ipAddresses": ["1.2.3.4", "5.6.7.8"]}) == {"ipAddresses": "list[2]"}


def test_reference_comparison() -> None:
    assert permissions_differ_from_reference(REAL_PERMISSIONS) is None
    assert permissions_differ_from_reference({"permissions": [1, 1]}) is None
    message = permissions_differ_from_reference({"permissions": [1, 3]})
    assert message is not None
    assert "[1, 3]" in message
    assert "вручную" in message
    assert permissions_differ_from_reference({"permissions": []}) is not None
    assert permissions_differ_from_reference({"other": 1}) is None
    assert permissions_differ_from_reference("text") is None


# ---------------------------------------------------------------- команда CLI


def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: str = "vst") -> Path:
    for name in ("BINGX_API_KEY", "BINGX_API_SECRET", "BINGX_ENV", "TELEGRAM_BOT_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        f"BINGX_API_KEY={KEY}\nBINGX_API_SECRET={SECRET}\nBINGX_ENV={env}\n", encoding="utf-8"
    )
    return path


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler: httpx.MockTransport) -> list[str]:
    hosts: list[str] = []
    original = BingXReadOnlyClient.__init__

    def patched(
        self: BingXReadOnlyClient, key: SecretStr, secret: SecretStr, base_url: str
    ) -> None:
        hosts.append(base_url)
        original(self, key, secret, base_url, transport=handler, now_ms=lambda: TS)

    monkeypatch.setattr(BingXReadOnlyClient, "__init__", patched)
    return hosts


def _router(permissions: object) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == BALANCE_PATH:
            return httpx.Response(200, json=BALANCE_LIST)
        assert request.url.path == API_PERMISSIONS_PATH
        return httpx.Response(200, json={"code": 0, "data": permissions})

    return httpx.MockTransport(handler)


def test_cli_ok_on_vst(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)
    hosts = _patch_transport(monkeypatch, _router({"permitsEnableReading": True}))
    code = main(["bingx-auth-check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    captured = capsys.readouterr()
    assert code == 0
    assert hosts == [VST_BASE_URL]
    assert "Окружение: vst" in captured.out
    assert "подпись принята" in captured.out
    assert "availableMargin=0" in captured.out  # на VST суммы показываются
    assert KEY not in captured.out + captured.err
    assert SECRET not in captured.out + captured.err
    assert "api-…1234" in captured.out  # ключ показан только в маске


def test_cli_live_hides_amounts_and_uses_live_host(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch, "live")
    hosts = _patch_transport(monkeypatch, _router({}))
    code = main(["bingx-auth-check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    out = capsys.readouterr().out
    assert code == 0
    assert hosts == [LIVE_BASE_URL]
    assert "Суммы скрыты" in out
    assert "availableMargin" not in out
    assert "только запросы на чтение" in out
    main(
        [
            "bingx-auth-check",
            "--show-balance",
            "--config-dir",
            str(REPO_CONFIG),
            "--env-file",
            str(env),
        ]
    )
    assert "availableMargin=0" in capsys.readouterr().out


def test_cli_warns_about_risky_permissions(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)
    _patch_transport(
        monkeypatch, _router({"permitsEnableFutures": True, "permitsEnableWithdraw": True})
    )
    code = main(["bingx-auth-check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    err = capsys.readouterr().err
    assert code == 0
    assert "ВНИМАНИЕ" in err
    assert "permitsEnableWithdraw" in err


def test_cli_permissions_endpoint_failure_is_not_fatal(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == BALANCE_PATH:
            return httpx.Response(200, json=BALANCE_LIST)
        return httpx.Response(404, text="not found")

    _patch_transport(monkeypatch, httpx.MockTransport(handler))
    code = main(["bingx-auth-check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    assert code == 0
    assert "проверить не удалось" in capsys.readouterr().out


def test_cli_failure_gives_reason_and_hints(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)
    body = {"code": 100001, "msg": "signature verification failed"}
    _patch_transport(monkeypatch, httpx.MockTransport(lambda _r: httpx.Response(200, json=body)))
    code = main(["bingx-auth-check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    err = capsys.readouterr().err
    assert code == 1
    assert "code=100001" in err
    assert "Что проверить" in err
    assert KEY not in err
    assert SECRET not in err


def test_cli_without_credentials(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("BINGX_API_KEY", "BINGX_API_SECRET", "BINGX_ENV"):
        monkeypatch.delenv(name, raising=False)
    empty = tmp_path / ".env"
    empty.write_text("", encoding="utf-8")
    code = main(["bingx-auth-check", "--config-dir", str(REPO_CONFIG), "--env-file", str(empty)])
    assert code == 2
    assert "BINGX_API_KEY" in capsys.readouterr().err


# ------------------------------------- права ключа: ответ без поля code (как на VST)


def test_extract_permissions_accepts_three_shapes() -> None:
    flags = {"permitsEnableReading": True}
    assert extract_permissions({"code": 0, "data": flags}) == flags
    assert extract_permissions({"data": flags, "success": True}) == flags
    assert extract_permissions(flags) == flags
    assert extract_permissions("text") == "text"


def test_extract_permissions_reports_api_error() -> None:
    with pytest.raises(BingXAPIError):
        extract_permissions({"code": 100001, "msg": "bad"})


def test_describe_shape_never_shows_values() -> None:
    payload = {"apiKey": KEY, "flags": {"a": True, "b": 1}, "items": [1, 2], "n": None}
    shape = describe_shape(payload)
    assert KEY not in shape
    assert "apiKey: str" in shape
    assert "a: bool" in shape
    assert "items: list[2]" in shape
    assert describe_shape({"x": {"y": {"z": 1}}}, depth=2) == "{x: {y: dict}}"


def test_signed_get_json_returns_payload_as_is() -> None:
    body = {"success": True, "data": {"permitsEnableReading": True}}
    with _client(httpx.MockTransport(lambda _r: httpx.Response(200, json=body))) as client:
        assert client.fetch_api_permissions() == body


def test_cli_handles_permissions_response_without_code(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)
    body = {"apiKey": KEY, "permitsEnableReading": True, "permitsEnableFutures": False}
    _patch_transport(monkeypatch, _router_raw(body))
    code = main(["bingx-auth-check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    captured = capsys.readouterr()
    assert code == 0
    assert "проверить не удалось" not in captured.out
    assert "'permitsEnableReading': True" in captured.out
    assert "Форма ответа (только имена и типы)" in captured.out
    assert "apiKey: str" in captured.out
    assert KEY not in captured.out + captured.err


def _router_raw(permissions_payload: object) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == BALANCE_PATH:
            return httpx.Response(200, json=BALANCE_LIST)
        return httpx.Response(200, json=permissions_payload)

    return httpx.MockTransport(handler)


def test_cli_real_permissions_shape_reports_ip_and_no_warning(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)
    _patch_transport(monkeypatch, _router_raw(REAL_PERMISSIONS))
    code = main(["bingx-auth-check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    captured = capsys.readouterr()
    assert code == 0
    assert "{'permissions': [1], 'ipAddresses': []}" in captured.out
    assert "Привязка к IP (ipAddresses): нет адресов" in captured.out
    assert "ВНИМАНИЕ" not in captured.err
    assert "my note" not in captured.out
    assert KEY not in captured.out + captured.err


def test_cli_warns_when_permissions_differ_from_reference(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path, monkeypatch)
    payload = {"permissions": [1, 3], "ipAddresses": ["203.0.113.7"]}
    _patch_transport(monkeypatch, _router_raw(payload))
    code = main(["bingx-auth-check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    captured = capsys.readouterr()
    assert code == 0
    assert "ВНИМАНИЕ" in captured.err
    assert "[1, 3]" in captured.err
    assert "Привязка к IP (ipAddresses): есть" in captured.out
    assert "203.0.113.7" not in captured.out + captured.err
