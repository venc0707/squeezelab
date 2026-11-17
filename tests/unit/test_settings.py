"""Тесты P00-T02 и P01-T01: схемы конфигов, режимы, два хэша, секреты по областям, логи."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
from pydantic import SecretStr

from squeezelab.app.cli import main
from squeezelab.app.logging_setup import RedactingFormatter, redact
from squeezelab.app.settings import (
    ConfigError,
    Mode,
    Secrets,
    app_hash,
    load_bingx_secrets,
    load_project_config,
    load_secrets,
    load_telegram_secrets,
    strategy_hash,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"

PINNED_STRATEGY_HASH = "3a4474d65a98ddc6921bbe848a5b8ec54dab5fa34cbcd96e7773102cf60a2c66"
PINNED_APP_HASH = "ba9ad58477eeeea59d6766731003c6f57e2c90524b442066841f71bc555419fe"


@pytest.fixture
def cfg_dir(tmp_path: Path) -> Path:
    target = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, target)
    return target


def _edit(cfg_dir: Path, name: str, old: str, new: str) -> None:
    path = cfg_dir / name
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} не найден в {name}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _fill_decisions(cfg_dir: Path, *, priorities: bool = False) -> None:
    """Закрывает открытые решения risk и exits (только для тестов режимов).

    priorities=True дополнительно задаёт priority пресетов (нужен paper_vst и live).
    """
    (cfg_dir / "risk.yaml").write_text(
        "risk_per_trade_pct: 0.5\nmax_leverage: 5\ndaily_loss_limit_pct: 2\n"
        "weekly_loss_limit_pct: 5\nstop_buffer_atr: 0.1\nmin_stop_atr: 0.5\nmax_stop_atr: 3\n"
        "position_mode: one_way\nmargin_mode: isolated\n",
        encoding="utf-8",
    )
    _edit(cfg_dir, "exits.yaml", "breakeven_buffer_bps: null", "breakeven_buffer_bps: 5")
    _edit(cfg_dir, "exits.yaml", "tp_order_type: null", "tp_order_type: limit")
    if priorities:
        _edit(
            cfg_dir,
            "presets.yaml",
            "max_len: 250\n    priority: null",
            "max_len: 250\n    priority: 1",
        )
        _edit(
            cfg_dir,
            "presets.yaml",
            "max_len: 300\n    priority: null",
            "max_len: 300\n    priority: 2",
        )


# ------------------------------------------------------------------ корректный конфиг


def test_repo_config_is_valid() -> None:
    cfg = load_project_config(REPO_CONFIG)
    assert cfg.app.mode is Mode.SIGNALS_ONLY
    assert set(cfg.presets.presets) == {"lvl6", "lvl3"}


def test_presets_match_agreed_decisions() -> None:
    presets = load_project_config(REPO_CONFIG).presets.presets
    lvl6, lvl3 = presets["lvl6"], presets["lvl3"]
    assert (lvl6.level, lvl6.max_slope, lvl6.max_len) == (6, 25, 250)
    assert (lvl3.level, lvl3.max_slope, lvl3.max_len) == (3, 22, 300)
    for p in (lvl6, lvl3):
        assert (p.tf, p.length, p.source) == ("15m", 500, "close")
        assert (p.min_dist, p.min_in, p.gap_tol, p.min_touches) == (5, 1, 2, 3)
        assert (p.min_depth, p.max_depth, p.min_grow, p.tol) == (2, 25, 2, 1)
        assert p.squeeze_upper
        assert p.squeeze_lower
        assert p.priority is None


def test_exits_and_symbols_match_agreed_decisions() -> None:
    cfg = load_project_config(REPO_CONFIG)
    assert cfg.exits.ladder.r_multiples == [1, 2, 3, 5, 10]
    assert cfg.exits.ladder.shares_pct == [25, 25, 25, 15, 10]
    assert cfg.exits.breakeven_after == "TP1"
    assert cfg.exits.breakeven_buffer_bps is None
    assert cfg.exits.tp_order_type is None
    sym = cfg.symbols.symbols[0]
    assert (sym.symbol, sym.tf, str(sym.data_start), sym.warmup_bars) == (
        "BNB-USDT",
        "15m",
        "2026-07-01",
        500,
    )


def test_open_and_closed_risk_decisions() -> None:
    risk = load_project_config(REPO_CONFIG).risk
    assert risk.position_mode == "one_way"  # D-04, решение пользователя 2026-11-13
    assert risk.margin_mode == "isolated"  # D-05, решение пользователя 2026-11-13
    assert set(risk.unresolved()) == {
        "risk_per_trade_pct",
        "max_leverage",
        "daily_loss_limit_pct",
        "weekly_loss_limit_pct",
        "stop_buffer_atr",
        "min_stop_atr",
        "max_stop_atr",
    }


# ------------------------------------------------------------------ ошибки валидации


def test_shares_not_summing_to_100(cfg_dir: Path) -> None:
    _edit(cfg_dir, "exits.yaml", "[25, 25, 25, 15, 10]", "[25, 25, 25, 15, 15]")
    with pytest.raises(ConfigError, match="сумма долей равна 105"):
        load_project_config(cfg_dir)


def test_r_must_strictly_increase(cfg_dir: Path) -> None:
    _edit(cfg_dir, "exits.yaml", "[1, 2, 3, 5, 10]", "[1, 2, 2, 5, 10]")
    with pytest.raises(ConfigError, match="строго возрастать"):
        load_project_config(cfg_dir)


def test_detector_ranges(cfg_dir: Path) -> None:
    _edit(cfg_dir, "presets.yaml", "level: 6", "level: 51")
    with pytest.raises(ConfigError, match=r"lvl6\.level"):
        load_project_config(cfg_dir)


def test_max_depth_over_100_and_min_touches_below_3(cfg_dir: Path) -> None:
    _edit(cfg_dir, "presets.yaml", "max_depth: 25", "max_depth: 101")
    _edit(cfg_dir, "presets.yaml", "min_touches: 3", "min_touches: 2")
    with pytest.raises(ConfigError) as exc:
        load_project_config(cfg_dir)
    assert "max_depth" in str(exc.value)
    assert "min_touches" in str(exc.value)


def test_unknown_key_is_rejected(cfg_dir: Path) -> None:
    _edit(
        cfg_dir, "risk.yaml", "margin_mode: isolated", "margin_mode: isolated\nmargin_modee: null"
    )
    with pytest.raises(ConfigError, match="margin_modee"):
        load_project_config(cfg_dir)


def test_missing_file(cfg_dir: Path) -> None:
    (cfg_dir / "exits.yaml").unlink()
    with pytest.raises(ConfigError, match="файл не найден"):
        load_project_config(cfg_dir)


def test_duplicate_priority(cfg_dir: Path) -> None:
    text = (
        (cfg_dir / "presets.yaml")
        .read_text(encoding="utf-8")
        .replace("priority: null", "priority: 1")
    )
    (cfg_dir / "presets.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="priority 1 задан у двух пресетов"):
        load_project_config(cfg_dir)


def test_warmup_shorter_than_length(cfg_dir: Path) -> None:
    _edit(cfg_dir, "symbols.yaml", "warmup_bars: 500", "warmup_bars: 300")
    with pytest.raises(ConfigError, match="warmup_bars=300"):
        load_project_config(cfg_dir)


# ------------------------------------------------------- режимы и открытые решения


def test_non_signals_mode_rejected_while_null(cfg_dir: Path) -> None:
    _edit(cfg_dir, "app.yaml", "mode: signals_only", "mode: backtest")
    with pytest.raises(ConfigError, match=r"backtest запрещён.*risk\.max_leverage"):
        load_project_config(cfg_dir)


@pytest.mark.parametrize("mode", ["backtest", "paper_vst", "live"])
def test_every_non_signals_mode_blocked_by_null(cfg_dir: Path, mode: str) -> None:
    _edit(cfg_dir, "app.yaml", "mode: signals_only", f"mode: {mode}")
    with pytest.raises(ConfigError, match="запрещён"):
        load_project_config(cfg_dir)


def test_non_signals_mode_allowed_when_decisions_closed(cfg_dir: Path) -> None:
    _fill_decisions(cfg_dir, priorities=True)
    _edit(cfg_dir, "app.yaml", "mode: signals_only", "mode: paper_vst")
    assert load_project_config(cfg_dir).app.mode is Mode.PAPER_VST


@pytest.mark.parametrize("mode", ["paper_vst", "live"])
def test_null_priority_blocks_paper_and_live(cfg_dir: Path, mode: str) -> None:
    _fill_decisions(cfg_dir)  # risk и exits закрыты, priority остаётся null
    _edit(cfg_dir, "app.yaml", "mode: signals_only", f"mode: {mode}")
    with pytest.raises(ConfigError, match=r"presets\.lvl6\.priority.*presets\.lvl3\.priority"):
        load_project_config(cfg_dir)


def test_null_priority_does_not_block_backtest(cfg_dir: Path) -> None:
    _fill_decisions(cfg_dir)
    _edit(cfg_dir, "app.yaml", "mode: signals_only", "mode: backtest")
    assert load_project_config(cfg_dir).app.mode is Mode.BACKTEST


def test_backtest_still_blocked_by_risk_and_exits_nulls(cfg_dir: Path) -> None:
    # Исключение касается только priority: прочие открытые решения блокируют бэктест, как раньше.
    _edit(cfg_dir, "app.yaml", "mode: signals_only", "mode: backtest")
    with pytest.raises(ConfigError, match=r"exits\.breakeven_buffer_bps") as exc:
        load_project_config(cfg_dir)
    assert "priority" not in str(exc.value)


def test_old_breakeven_buffer_key_is_rejected(cfg_dir: Path) -> None:
    _edit(cfg_dir, "exits.yaml", "breakeven_buffer_bps:", "breakeven_buffer:")
    with pytest.raises(ConfigError, match="breakeven_buffer"):
        load_project_config(cfg_dir)


def test_breakeven_buffer_bps_must_not_be_negative(cfg_dir: Path) -> None:
    _edit(cfg_dir, "exits.yaml", "breakeven_buffer_bps: null", "breakeven_buffer_bps: -1")
    with pytest.raises(ConfigError, match="breakeven_buffer_bps"):
        load_project_config(cfg_dir)


def test_one_remaining_null_still_blocks(cfg_dir: Path) -> None:
    _fill_decisions(cfg_dir, priorities=True)
    _edit(cfg_dir, "exits.yaml", "tp_order_type: limit", "tp_order_type: null")
    _edit(cfg_dir, "app.yaml", "mode: signals_only", "mode: live")
    with pytest.raises(ConfigError, match=r"exits\.tp_order_type"):
        load_project_config(cfg_dir)


def test_min_stop_must_be_below_max_stop(cfg_dir: Path) -> None:
    _edit(cfg_dir, "risk.yaml", "min_stop_atr: null", "min_stop_atr: 3")
    _edit(cfg_dir, "risk.yaml", "max_stop_atr: null", "max_stop_atr: 2")
    with pytest.raises(ConfigError, match="min_stop_atr"):
        load_project_config(cfg_dir)


# ------------------------------------------------------- recorder в app.yaml


def test_recorder_defaults_match_repo_config() -> None:
    recorder = load_project_config(REPO_CONFIG).app.recorder
    assert recorder.symbols == ["BNB-USDT"]
    assert (recorder.oi_interval_s, recorder.funding_interval_s) == (60, 60)


def test_recorder_section_is_optional(cfg_dir: Path) -> None:
    (cfg_dir / "app.yaml").write_text(
        "mode: signals_only\npaths:\n  data_dir: data\n  logs_dir: logs\n", encoding="utf-8"
    )
    recorder = load_project_config(cfg_dir).app.recorder
    assert recorder.symbols == ["BNB-USDT"]
    assert recorder.oi_interval_s == 60


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("symbols: [BNB-USDT]", "symbols: []", "recorder.symbols"),
        ("symbols: [BNB-USDT]", "symbols: [bnbusdt]", "ожидается вид"),
        ("symbols: [BNB-USDT]", "symbols: [BNB-USDT, BNB-USDT]", "дважды"),
        ("oi_interval_s: 60", "oi_interval_s: 1", "recorder.oi_interval_s"),
        ("funding_interval_s: 60", "funding_interval_s: 100000", "recorder.funding_interval_s"),
    ],
)
def test_recorder_validation(cfg_dir: Path, old: str, new: str, fragment: str) -> None:
    _edit(cfg_dir, "app.yaml", old, new)
    with pytest.raises(ConfigError, match=fragment):
        load_project_config(cfg_dir)


def test_data_dir_may_be_omitted(cfg_dir: Path) -> None:
    (cfg_dir / "app.yaml").write_text("paths:\n  logs_dir: logs\n", encoding="utf-8")
    assert load_project_config(cfg_dir).app.paths.data_dir is None


# ---------------------------------------------------------------- хэш: две части


def test_hashes_are_stable_between_loads() -> None:
    first = load_project_config(REPO_CONFIG)
    second = load_project_config(REPO_CONFIG)
    for func in (strategy_hash, app_hash):
        assert func(first) == func(second)
        assert len(func(first)) == 64
    assert strategy_hash(first) != app_hash(first)


def test_repo_hashes_are_pinned() -> None:
    """Эталон: меняется только при осознанной правке presets/exits/risk/symbols или app.yaml.

    Те же значения должны напечатать `config check` на ПК и на сервере.
    """
    cfg = load_project_config(REPO_CONFIG)
    assert strategy_hash(cfg) == PINNED_STRATEGY_HASH
    assert app_hash(cfg) == PINNED_APP_HASH


def test_strategy_hash_ignores_comments_and_number_style(cfg_dir: Path) -> None:
    before = strategy_hash(load_project_config(cfg_dir))
    _edit(cfg_dir, "presets.yaml", "level: 6", "level: 6.0  # тот же уровень")
    (cfg_dir / "risk.yaml").write_text(
        "# другой комментарий\n" + (cfg_dir / "risk.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert strategy_hash(load_project_config(cfg_dir)) == before


@pytest.mark.parametrize(
    ("name", "old", "new"),
    [
        ("presets.yaml", "max_slope: 25", "max_slope: 26"),
        ("exits.yaml", "breakeven_after: TP1", "breakeven_after: TP2"),
        ("risk.yaml", "max_leverage: null", "max_leverage: 5"),
        ("symbols.yaml", "warmup_bars: 500", "warmup_bars: 600"),
    ],
)
def test_strategy_hash_changes_with_strategy_files_only(
    cfg_dir: Path, name: str, old: str, new: str
) -> None:
    cfg = load_project_config(cfg_dir)
    strategy_before, app_before = strategy_hash(cfg), app_hash(cfg)
    _edit(cfg_dir, name, old, new)
    changed = load_project_config(cfg_dir)
    assert strategy_hash(changed) != strategy_before
    assert app_hash(changed) == app_before


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("log_level: INFO", "log_level: DEBUG"),
        ("data_dir: data", "data_dir: /srv/data/squeezelab"),
        ("symbols: [BNB-USDT]", "symbols: [BNB-USDT, ETH-USDT]"),
        ("oi_interval_s: 60", "oi_interval_s: 120"),
    ],
)
def test_app_hash_changes_with_app_yaml_only(cfg_dir: Path, old: str, new: str) -> None:
    cfg = load_project_config(cfg_dir)
    strategy_before, app_before = strategy_hash(cfg), app_hash(cfg)
    _edit(cfg_dir, "app.yaml", old, new)
    changed = load_project_config(cfg_dir)
    assert app_hash(changed) != app_before
    assert strategy_hash(changed) == strategy_before


def test_hashes_do_not_depend_on_yaml_key_order(cfg_dir: Path) -> None:
    cfg = load_project_config(cfg_dir)
    strategy_before, app_before = strategy_hash(cfg), app_hash(cfg)
    (cfg_dir / "app.yaml").write_text(
        "recorder:\n  funding_interval_s: 60\n  oi_interval_s: 60\n  symbols: [BNB-USDT]\n"
        "log_level: INFO\npaths:\n  logs_dir: logs\n  data_dir: data\ntimezone: UTC\n"
        "mode: signals_only\n",
        encoding="utf-8",
    )
    reordered = load_project_config(cfg_dir)
    assert strategy_hash(reordered) == strategy_before
    assert app_hash(reordered) == app_before


def test_mode_change_is_app_hash_only(cfg_dir: Path) -> None:
    _fill_decisions(cfg_dir, priorities=True)
    before = load_project_config(cfg_dir)
    _edit(cfg_dir, "app.yaml", "mode: signals_only", "mode: paper_vst")
    after = load_project_config(cfg_dir)
    assert strategy_hash(after) == strategy_hash(before)
    assert app_hash(after) != app_hash(before)


# ---------------------------------------------------------------------- секреты

_SECRET_ENV_NAMES = (
    "BINGX_API_KEY",
    "BINGX_API_SECRET",
    "BINGX_ENV",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_IDS",
    "TELEGRAM_PROXY_URL",
)


def _clear_secret_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_secrets_masked_in_repr_and_str(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "BINGX_API_KEY=key-abcdef123456\nBINGX_API_SECRET=sec-zyxwvu987654\n"
        "TELEGRAM_BOT_TOKEN=123456:AAH-token-value\nTELEGRAM_CHAT_IDS=111, -1002223334445\n",
        encoding="utf-8",
    )
    secrets = load_secrets(env)
    shown = repr(secrets) + str(secrets)
    for raw in ("key-abcdef123456", "sec-zyxwvu987654", "123456:AAH-token-value"):
        assert raw not in shown
    assert secrets.telegram_chat_ids == [111, -1002223334445]
    assert secrets.has_bingx_credentials
    assert secrets.has_telegram


def test_blank_values_mean_not_set(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("BINGX_API_KEY=\nTELEGRAM_BOT_TOKEN=  \nTELEGRAM_CHAT_IDS=\n", encoding="utf-8")
    secrets = load_secrets(env)
    assert not secrets.has_bingx_credentials
    assert not secrets.has_telegram
    assert secrets.telegram_chat_ids == []
    assert secrets.bingx_env == "vst"  # безопасное значение по умолчанию


def test_missing_env_file_is_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_secret_env(monkeypatch)
    assert load_secrets(tmp_path / "нет-такого.env").presence() == {
        "BINGX_API_KEY": False,
        "BINGX_API_SECRET": False,
        "TELEGRAM_BOT_TOKEN": False,
        "TELEGRAM_CHAT_IDS": False,
        "TELEGRAM_PROXY_URL": False,
    }


def test_environment_overrides_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("BINGX_ENV=vst\n", encoding="utf-8")
    monkeypatch.setenv("BINGX_ENV", "live")
    assert load_secrets(env).bingx_env == "live"


def test_bad_chat_ids_and_env_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINGX_ENV", raising=False)
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_CHAT_IDS=abc\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="TELEGRAM_CHAT_IDS"):
        load_secrets(env)
    env.write_text("BINGX_ENV=prod\nBINGX_API_SECRET=very-secret-value\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_secrets(env)
    assert "BINGX_ENV" in str(exc.value)
    assert "very-secret-value" not in str(exc.value)


def test_telegram_scope_never_loads_bingx_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_secret_env(monkeypatch)
    env = tmp_path / ".env"
    # BINGX_ENV=prod в полной области был бы ошибкой; рекордеру он не мешает и не читается.
    env.write_text(
        "BINGX_API_KEY=key-abcdef123456\nBINGX_API_SECRET=sec-zyxwvu987654\nBINGX_ENV=prod\n"
        "TELEGRAM_BOT_TOKEN=123456:AAH-token-value\nTELEGRAM_CHAT_IDS=111\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BINGX_API_KEY", "key-from-process-env")
    secrets = load_telegram_secrets(env)
    assert secrets.has_telegram
    assert not hasattr(secrets, "bingx_api_key")
    assert not hasattr(secrets, "bingx_api_secret")
    assert secrets.secret_values() == ["123456:AAH-token-value"]
    assert set(secrets.presence()) == {
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_IDS",
        "TELEGRAM_PROXY_URL",
    }
    shown = repr(secrets) + str(secrets)
    assert "key-abcdef123456" not in shown
    assert "key-from-process-env" not in shown
    with pytest.raises(ConfigError, match="BINGX_ENV"):
        load_secrets(env)


def test_bingx_scope_never_loads_telegram_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_secret_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        "BINGX_API_KEY=key-abcdef123456\nBINGX_API_SECRET=sec-zyxwvu987654\n"
        "TELEGRAM_BOT_TOKEN=123456:AAH-token-value\nTELEGRAM_CHAT_IDS=abc\n",
        encoding="utf-8",
    )
    secrets = load_bingx_secrets(env)  # кривой TELEGRAM_CHAT_IDS команде BingX не мешает
    assert secrets.has_bingx_credentials
    assert not hasattr(secrets, "telegram_bot_token")
    assert sorted(secrets.secret_values()) == ["key-abcdef123456", "sec-zyxwvu987654"]


def test_telegram_scope_works_with_no_secrets_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_secret_env(monkeypatch)
    secrets = load_telegram_secrets(tmp_path / "нет-такого.env")
    assert not secrets.has_telegram
    assert secrets.telegram_proxy_url is None
    assert secrets.secret_values() == []


def test_proxy_url_is_a_secret_and_is_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_secret_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_PROXY_URL=http://user:pa55word@127.0.0.1:10809\n", encoding="utf-8")
    secrets = load_telegram_secrets(env)
    assert secrets.telegram_proxy_url is not None
    assert "pa55word" not in repr(secrets) + str(secrets)
    assert secrets.secret_values() == ["http://user:pa55word@127.0.0.1:10809"]
    assert secrets.presence()["TELEGRAM_PROXY_URL"] is True


def test_blank_proxy_url_means_not_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_secret_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_PROXY_URL=   \n", encoding="utf-8")
    assert load_telegram_secrets(env).telegram_proxy_url is None


@pytest.mark.parametrize(
    "bad",
    [
        "user:pa55word@127.0.0.2:10809",  # нет схемы
        "socks5://user:pa55word@127.0.0.2:1080",  # схема не поддерживается
        "http://user:pa55word@",  # нет хоста
        "http://user:pa55word@127.0.0.2:port",  # порт не число
        "ftp://user:pa55word@h:1",
    ],
)
def test_bad_proxy_url_is_rejected_without_echoing_the_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    _clear_secret_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(f"TELEGRAM_PROXY_URL={bad}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="TELEGRAM_PROXY_URL") as exc:
        load_telegram_secrets(env)
    assert "pa55word" not in str(exc.value)
    assert "127.0.0.2" not in str(exc.value)


def test_secret_values_helper() -> None:
    secrets = Secrets(
        _env_file=None,
        bingx_api_key=SecretStr("k" * 8),
        bingx_api_secret=SecretStr("s" * 8),
    )
    assert sorted(secrets.secret_values()) == ["k" * 8, "s" * 8]
    with_proxy = Secrets(
        _env_file=None,
        telegram_bot_token=SecretStr("t" * 8),
        telegram_proxy_url=SecretStr("http://127.0.0.1:10809"),
    )
    assert sorted(with_proxy.secret_values()) == ["http://127.0.0.1:10809", "t" * 8]


# --------------------------------------------------------------------- логи


def test_redact_masks_all_secrets() -> None:
    text = "POST https://api.telegram.org/bot123456:AAH-token/sendMessage key=key-abcdef"
    out = redact(text, ["123456:AAH-token", "key-abcdef"])
    assert "AAH-token" not in out
    assert "key-abcdef" not in out
    assert out.count("***") == 2


def test_redact_ignores_too_short_values() -> None:
    assert redact("a b c", ["a"]) == "a b c"


def test_formatter_masks_message_and_traceback() -> None:
    formatter = RedactingFormatter("%(levelname)s %(message)s", ["topsecret-value"])
    try:
        raise RuntimeError("fail for topsecret-value")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            "t", logging.ERROR, __file__, 1, "token=%s", ("topsecret-value",), sys.exc_info()
        )
    out = formatter.format(record)
    assert "topsecret-value" not in out
    assert "***" in out


# ---------------------------------------------------------------------- CLI


def test_cli_config_check_ok(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = main(
        ["config", "check", "--config-dir", str(REPO_CONFIG), "--env-file", str(tmp_path / "x")]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Конфиг: OK" in out
    cfg = load_project_config(REPO_CONFIG)
    assert f"Хэш стратегии (strategy_hash, sha256): {strategy_hash(cfg)}" in out
    assert f"Хэш приложения (app_hash, sha256, только диагностика): {app_hash(cfg)}" in out
    assert "Каталог данных: data (источник: paths.data_dir в app.yaml)" in out
    assert "Рекордер: пары BNB-USDT" in out
    assert "Открытые решения (null): 11" in out


def test_cli_config_check_shows_data_dir_from_environment(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SQUEEZELAB_DATA_DIR", "/srv/data/squeezelab")
    code = main(
        ["config", "check", "--config-dir", str(REPO_CONFIG), "--env-file", str(tmp_path / "x")]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "переменная SQUEEZELAB_DATA_DIR" in out
    # Путь данных не входит ни в один хэш: на сервере и на ПК хэши совпадают.
    cfg = load_project_config(REPO_CONFIG)
    assert strategy_hash(cfg) in out
    assert app_hash(cfg) in out


def test_cli_config_check_fails_on_broken_config(
    capsys: pytest.CaptureFixture[str], cfg_dir: Path, tmp_path: Path
) -> None:
    _edit(cfg_dir, "exits.yaml", "[25, 25, 25, 15, 10]", "[25, 25, 25, 15, 15]")
    code = main(
        ["config", "check", "--config-dir", str(cfg_dir), "--env-file", str(tmp_path / "x")]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "ОШИБКА" in captured.err
    assert "сумма долей" in captured.err


def test_cli_config_check_never_prints_secret_values(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_secret_env(monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        "BINGX_API_KEY=key-abcdef123456\nTELEGRAM_BOT_TOKEN=999:token-secret\n"
        "TELEGRAM_PROXY_URL=http://user:pa55word@127.0.0.1:10809\n",
        encoding="utf-8",
    )
    code = main(["config", "check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    out = capsys.readouterr().out
    assert code == 0
    assert "key-abcdef123456" not in out
    assert "token-secret" not in out
    assert "pa55word" not in out
    assert "BINGX_API_KEY: задан" in out
    assert "TELEGRAM_PROXY_URL: задан" in out
