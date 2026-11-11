"""Командная строка SqueezeLab.

Коды выхода: 0 — успех, 1 — сбой операции (сеть, API), 2 — ошибка конфигурации или аргументов.
"""

from __future__ import annotations

import argparse
import sys
import time
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
from squeezelab.data.bingx_rest import (
    LIVE_BASE_URL,
    BingXError,
    BingXPublicClient,
)
from squeezelab.data.models import (
    detect_order,
    format_ts,
    is_candle_closed,
    is_contiguous,
    tf_to_ms,
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

    smoke = sub.add_parser(
        "bingx-smoke",
        parents=[common],
        help="публичный запрос свечей BingX (без ключей)",
    )
    smoke.add_argument("--symbol", default=None, help="по умолчанию первая пара из symbols.yaml")
    smoke.add_argument("--interval", default=None, help="по умолчанию её таймфрейм")
    smoke.add_argument("--limit", type=int, default=20, help="сколько свечей запросить (20)")
    smoke.add_argument("--base-url", default=LIVE_BASE_URL, help=f"по умолчанию {LIVE_BASE_URL}")
    smoke.add_argument(
        "--save-raw", type=Path, default=None, help="сохранить сырой JSON-ответ в этот файл"
    )
    smoke.set_defaults(handler=cmd_bingx_smoke)

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


def cmd_bingx_smoke(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
    except ConfigError as exc:
        print(f"Конфиг: ОШИБКА\n{exc}", file=sys.stderr)
        return EXIT_CONFIG
    first = cfg.symbols.symbols[0]
    symbol = args.symbol or first.symbol
    interval = args.interval or first.tf
    try:
        tf_ms = tf_to_ms(interval)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level)

    try:
        with BingXPublicClient(args.base_url) as client:
            result = client.fetch_klines(symbol, interval, args.limit)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_CONFIG
    except BingXError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    candles = result.candles
    now_ms = int(time.time() * 1000)
    print(f"Запрос: GET {args.base_url}/openApi/swap/v3/quote/klines")
    print(f"Символ: {symbol}, интервал: {interval}, limit: {args.limit}")
    print(f"Получено свечей: {len(candles)}")
    if not candles:
        print("Свечей нет: проверь символ и интервал.")
        return EXIT_FAILURE
    order_names = {
        "single": "одна свеча",
        "asc": "по возрастанию ts_open (старые первыми)",
        "desc": "по убыванию ts_open (новые первыми)",
        "mixed": "смешанный",
    }
    print(f"Порядок в ответе: {order_names[detect_order(candles)]}")
    print(f"Первая в ответе:  {candles[0].ts_open} ({format_ts(candles[0].ts_open)})")
    print(f"Последняя в ответе: {candles[-1].ts_open} ({format_ts(candles[-1].ts_open)})")
    newest = max(candles, key=lambda candle: candle.ts_open)
    closed = is_candle_closed(newest, tf_ms, now_ms)
    print(
        f"Самая новая свеча: {newest.ts_open} ({format_ts(newest.ts_open)}), "
        f"закрыта: {'да' if closed else 'НЕТ, ещё формируется (отбрасывать)'}"
    )
    print(f"Без пропусков и дублей: {'да' if is_contiguous(candles, tf_ms) else 'НЕТ'}")
    print(f"Локальное время сейчас: {format_ts(now_ms)} (сверка по часам этой машины)")
    if result.limit_headers:
        print("Заголовки лимитов:")
        for name, value in sorted(result.limit_headers.items()):
            print(f"  {name}: {value}")
    else:
        print("Заголовки лимитов: в ответе не обнаружено")
    if args.save_raw is not None:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        args.save_raw.write_text(result.raw_text, encoding="utf-8")
        print(f"Сырой ответ сохранён: {args.save_raw}")
    return EXIT_OK


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
