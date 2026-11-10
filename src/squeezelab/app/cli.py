"""Командная строка SqueezeLab.

Коды выхода: 0 — успех, 1 — сбой операции (сеть, API), 2 — ошибка конфигурации или аргументов.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from squeezelab import __version__
from squeezelab.app.logging_setup import configure_logging
from squeezelab.app.settings import (
    ConfigError,
    ProjectConfig,
    config_hash,
    load_project_config,
    load_secrets,
    resolve_config_dir,
)
from squeezelab.notify.telegram import TelegramClient, TelegramError, build_test_message

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="каталог с YAML-конфигами (по умолчанию $SQUEEZELAB_CONFIG_DIR или ./config)",
    )
    common.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="файл с секретами (по умолчанию ./.env)",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="squeezelab",
        description="SqueezeLab: торговый бот для BingX USDT-M perpetual.",
    )
    parser.add_argument("--version", action="version", version=f"squeezelab {__version__}")
    common = _common_options()
    sub = parser.add_subparsers(dest="command", metavar="КОМАНДА")

    config = sub.add_parser("config", help="работа с конфигами")
    config_sub = config.add_subparsers(dest="config_command", metavar="ДЕЙСТВИЕ", required=True)
    config_sub.add_parser(
        "check", parents=[common], help="проверить конфиги и напечатать хэш"
    ).set_defaults(handler=cmd_config_check)

    notify = sub.add_parser(
        "notify-test",
        parents=[common],
        help="отправить тестовое сообщение в Telegram (только в чаты из белого списка)",
    )
    notify.add_argument(
        "--dry-run", action="store_true", help="показать сообщение и адресатов, ничего не отправляя"
    )
    notify.set_defaults(handler=cmd_notify_test)

    return parser


def _load_config(config_dir: Path | None) -> ProjectConfig:
    return load_project_config(resolve_config_dir(config_dir))


def cmd_config_check(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        secrets = load_secrets(args.env_file)
    except ConfigError as exc:
        print(f"Конфиг: ОШИБКА\n{exc}", file=sys.stderr)
        return EXIT_CONFIG

    print("Конфиг: OK")
    print(f"Каталог: {resolve_config_dir(args.config_dir)}")
    print(f"Режим: {cfg.app.mode.value}")
    print(f"Хэш (sha256): {config_hash(cfg)}")
    print(f"Пресеты: {', '.join(cfg.presets.presets)}")
    print(
        "Пары: " + ", ".join(f"{s.symbol} {s.tf} (с {s.data_start})" for s in cfg.symbols.symbols)
    )
    open_items = cfg.unresolved_decisions()
    print(f"Открытые решения (null): {len(open_items)}")
    for item in open_items:
        print(f"  - {item}")

    env_path = Path(args.env_file)
    print(f"Файл секретов {env_path}: {'найден' if env_path.is_file() else 'не найден'}")
    print(f"BINGX_ENV: {secrets.bingx_env}")
    for name, present in secrets.presence().items():
        print(f"  {name}: {'задан' if present else 'не задан'}")
    return EXIT_OK


def cmd_notify_test(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        secrets = load_secrets(args.env_file)
    except ConfigError as exc:
        print(f"Конфиг: ОШИБКА\n{exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level, secrets.secret_values())

    if secrets.telegram_bot_token is None or not secrets.telegram_chat_ids:
        print(
            "TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_IDS должны быть заданы в .env "
            "(инструкция: docs/runbook.md, раздел Telegram)",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    text = build_test_message(cfg.app.mode.value, config_hash(cfg))
    if args.dry_run:
        print("Сухой прогон, ничего не отправлено. Текст сообщения:\n")
        print(text)
        print(f"\nАдресаты (белый список): {', '.join(map(str, secrets.telegram_chat_ids))}")
        return EXIT_OK

    failed = False
    with TelegramClient(secrets.telegram_bot_token, secrets.telegram_chat_ids) as client:
        for chat_id in secrets.telegram_chat_ids:
            try:
                client.send_message(chat_id, text)
            except TelegramError as exc:
                failed = True
                print(f"chat_id {chat_id}: ОШИБКА: {exc}", file=sys.stderr)
            else:
                print(f"chat_id {chat_id}: отправлено")
    return EXIT_FAILURE if failed else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    # На Windows консоль может быть не в UTF-8: сообщения на русском не должны падать.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_OK
    result: int = handler(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
