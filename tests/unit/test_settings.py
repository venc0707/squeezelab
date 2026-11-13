"""Тесты P00-T02: схемы конфигов, режимы, хэш, секреты, маскирование в логах."""

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
    config_hash,
    load_project_config,
    load_secrets,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


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


def _fill_decisions(cfg_dir: Path) -> None:
    """Закрывает все открытые решения (только для тестов режимов)."""
    (cfg_dir / "risk.yaml").write_text(
        "risk_per_trade_pct: 0.5\nmax_leverage: 5\ndaily_loss_limit_pct: 2\n"
        "weekly_loss_limit_pct: 5\nstop_buffer_atr: 0.1\nmin_stop_atr: 0.5\nmax_stop_atr: 3\n"
        "position_mode: one_way\nmargin_mode: isolated\n",
        encoding="utf-8",
    )
    _edit(cfg_dir, "exits.yaml", "breakeven_buffer: null", "breakeven_buffer: 0.05")
    _edit(cfg_dir, "exits.yaml", "tp_order_type: null", "tp_order_type: limit")


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
    assert cfg.exits.breakeven_buffer is None
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
    _fill_decisions(cfg_dir)
    _edit(cfg_dir, "app.yaml", "mode: signals_only", "mode: paper_vst")
    assert load_project_config(cfg_dir).app.mode is Mode.PAPER_VST


def test_one_remaining_null_still_blocks(cfg_dir: Path) -> None:
    _fill_decisions(cfg_dir)
    _edit(cfg_dir, "exits.yaml", "tp_order_type: limit", "tp_order_type: null")
    _edit(cfg_dir, "app.yaml", "mode: signals_only", "mode: live")
    with pytest.raises(ConfigError, match=r"exits\.tp_order_type"):
        load_project_config(cfg_dir)


def test_min_stop_must_be_below_max_stop(cfg_dir: Path) -> None:
    _edit(cfg_dir, "risk.yaml", "min_stop_atr: null", "min_stop_atr: 3")
    _edit(cfg_dir, "risk.yaml", "max_stop_atr: null", "max_stop_atr: 2")
    with pytest.raises(ConfigError, match="min_stop_atr"):
        load_project_config(cfg_dir)


# ------------------------------------------------------------------------- хэш


def test_hash_is_stable_between_loads() -> None:
    first = config_hash(load_project_config(REPO_CONFIG))
    second = config_hash(load_project_config(REPO_CONFIG))
    assert first == second
    assert len(first) == 64


def test_hash_ignores_comments_and_number_style(cfg_dir: Path) -> None:
    before = config_hash(load_project_config(cfg_dir))
    _edit(cfg_dir, "presets.yaml", "level: 6", "level: 6.0  # тот же уровень")
    (cfg_dir / "risk.yaml").write_text(
        "# другой комментарий\n" + (cfg_dir / "risk.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert config_hash(load_project_config(cfg_dir)) == before


def test_hash_changes_with_any_value(cfg_dir: Path) -> None:
    before = config_hash(load_project_config(cfg_dir))
    _edit(cfg_dir, "presets.yaml", "max_slope: 25", "max_slope: 26")
    assert config_hash(load_project_config(cfg_dir)) != before


def test_hash_does_not_depend_on_yaml_key_order(cfg_dir: Path) -> None:
    before = config_hash(load_project_config(cfg_dir))
    (cfg_dir / "app.yaml").write_text(
        "log_level: INFO\npaths:\n  logs_dir: logs\n  data_dir: data\ntimezone: UTC\n"
        "mode: signals_only\n",
        encoding="utf-8",
    )
    assert config_hash(load_project_config(cfg_dir)) == before


# ---------------------------------------------------------------------- секреты


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
    for name in ("BINGX_API_KEY", "BINGX_API_SECRET", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_IDS"):
        monkeypatch.delenv(name, raising=False)
    assert load_secrets(tmp_path / "нет-такого.env").presence() == {
        "BINGX_API_KEY": False,
        "BINGX_API_SECRET": False,
        "TELEGRAM_BOT_TOKEN": False,
        "TELEGRAM_CHAT_IDS": False,
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


def test_secret_values_helper() -> None:
    secrets = Secrets(
        _env_file=None,
        bingx_api_key=SecretStr("k" * 8),
        bingx_api_secret=SecretStr("s" * 8),
    )
    assert sorted(secrets.secret_values()) == ["k" * 8, "s" * 8]


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
    assert config_hash(load_project_config(REPO_CONFIG)) in out
    assert "Открытые решения (null): 11" in out


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
    for name in ("BINGX_API_KEY", "BINGX_API_SECRET", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_IDS"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "BINGX_API_KEY=key-abcdef123456\nTELEGRAM_BOT_TOKEN=999:token-secret\n", encoding="utf-8"
    )
    code = main(["config", "check", "--config-dir", str(REPO_CONFIG), "--env-file", str(env)])
    out = capsys.readouterr().out
    assert code == 0
    assert "key-abcdef123456" not in out
    assert "token-secret" not in out
    assert "BINGX_API_KEY: задан" in out
