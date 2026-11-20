"""Командная строка SqueezeLab.

Коды выхода: 0 — успех, 1 — сбой операции (сеть, API), 2 — ошибка конфигурации или аргументов.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from squeezelab import __version__
from squeezelab.app.logging_setup import configure_logging
from squeezelab.app.settings import (
    DATA_DIR_ENV,
    ConfigError,
    ProjectConfig,
    app_hash,
    load_bingx_secrets,
    load_project_config,
    load_secrets,
    load_telegram_secrets,
    resolve_config_dir,
    resolve_data_dir,
    strategy_hash,
)
from squeezelab.data.bingx_rest import (
    LIVE_BASE_URL,
    MAX_KLINES_LIMIT,
    VALID_INTERVALS,
    VST_BASE_URL,
    BingXError,
    BingXPublicClient,
    BingXReadOnlyClient,
    describe_shape,
    extract_permissions,
    permission_fields,
)
from squeezelab.data.downloader import (
    DEFAULT_START,
    WINDOW_BARS,
    RateLimiter,
    download_tf,
    render_report,
)
from squeezelab.data.models import (
    detect_order,
    format_ts,
    is_candle_closed,
    is_contiguous,
    tf_to_ms,
)
from squeezelab.data.probe import (
    DEFAULT_ANCHOR,
    DEFAULT_VOLUME_BAR,
    build_cases,
    build_paging_cases,
    derive_findings,
    derive_paging_findings,
    render_results,
    run_probe,
    to_ms,
)
from squeezelab.data.store import CandleStore, StoreError
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

    data = sub.add_parser("data", help="данные: загрузка и проверка свечей")
    data_sub = data.add_subparsers(dest="data_command", metavar="ДЕЙСТВИЕ", required=True)
    download = data_sub.add_parser(
        "download",
        parents=[common],
        help="скачать закрытые свечи BingX в Parquet (повторный запуск докачивает)",
    )
    download.add_argument(
        "--symbol",
        action="append",
        default=None,
        help="пара, можно несколько раз; по умолчанию первая из symbols.yaml",
    )
    download.add_argument(
        "--tf",
        action="append",
        default=None,
        help="таймфрейм, можно несколько раз; по умолчанию 15m и 1m",
    )
    download.add_argument(
        "--start",
        default=DEFAULT_START.strftime("%Y-%m-%dT%H:%M"),
        help="начало периода, UTC (ГГГГ-ММ-ДД или ГГГГ-ММ-ДДTЧЧ:ММ); по умолчанию "
        + DEFAULT_START.strftime("%Y-%m-%d %H:%M"),
    )
    download.add_argument(
        "--end",
        default=None,
        help="конец периода, UTC, исключительно; по умолчанию последняя закрытая свеча",
    )
    download.add_argument("--base-url", default=LIVE_BASE_URL, help=f"по умолчанию {LIVE_BASE_URL}")
    download.add_argument(
        "--window-bars",
        type=int,
        default=WINDOW_BARS,
        help=f"баров в окне запроса (1..1440); по умолчанию {WINDOW_BARS} (предел ответа биржи)",
    )
    download.set_defaults(handler=cmd_data_download)

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

    probe = sub.add_parser(
        "bingx-probe-klines",
        parents=[common],
        help="пробные публичные запросы свечей: границы startTime/endTime, порядок, усечение",
    )
    probe.add_argument("--symbol", default=None, help="по умолчанию первая пара из symbols.yaml")
    probe.add_argument("--base-url", default=LIVE_BASE_URL, help=f"по умолчанию {LIVE_BASE_URL}")
    probe.add_argument(
        "--anchor",
        default=DEFAULT_ANCHOR.strftime("%Y-%m-%dT%H:%M"),
        help="граница свечи в прошлом (UTC, ГГГГ-ММ-ДДTЧЧ:ММ), от которой строятся окна",
    )
    probe.add_argument(
        "--volume-bar",
        default=DEFAULT_VOLUME_BAR.strftime("%Y-%m-%dT%H:%M"),
        help="начало 15м свечи (UTC) для сверки volume с TradingView",
    )
    probe.add_argument(
        "--set",
        dest="probe_set",
        choices=("basic", "paging"),
        default="basic",
        help="basic: границы и порядок; paging: предел ответа, endTime, 1m, начало истории",
    )
    probe.set_defaults(handler=cmd_bingx_probe_klines)

    auth = sub.add_parser(
        "bingx-auth-check",
        parents=[common],
        help="один подписанный запрос только на чтение (баланс) в окружении BINGX_ENV",
    )
    auth.add_argument(
        "--show-balance",
        action="store_true",
        help="показать суммы и для боевого окружения (для vst суммы показываются всегда)",
    )
    auth.set_defaults(handler=cmd_bingx_auth_check)

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
    print(f"Хэш стратегии (strategy_hash, sha256): {strategy_hash(cfg)}")
    print(f"Хэш приложения (app_hash, sha256, только диагностика): {app_hash(cfg)}")
    data_dir = resolve_data_dir(cfg.app)
    source_names = {
        "env": f"переменная {DATA_DIR_ENV}",
        "app.yaml": "paths.data_dir в app.yaml",
        "default": "по умолчанию ./data",
    }
    print(f"Каталог данных: {data_dir.path} (источник: {source_names[data_dir.source]})")
    print(f"Рекордер: пары {', '.join(cfg.app.recorder.symbols)}")
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
        secrets = load_telegram_secrets(args.env_file)
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

    text = build_test_message(cfg.app.mode.value, strategy_hash(cfg), app_hash(cfg))
    if args.dry_run:
        print("Сухой прогон, ничего не отправлено. Текст сообщения:\n")
        print(text)
        print(f"\nАдресаты (белый список): {', '.join(map(str, secrets.telegram_chat_ids))}")
        print(f"Прокси Telegram: {'задан' if secrets.telegram_proxy_url else 'не задан'}")
        return EXIT_OK

    failed = False
    with TelegramClient(
        secrets.telegram_bot_token,
        secrets.telegram_chat_ids,
        proxy=secrets.telegram_proxy_url,
    ) as client:
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
        # Завершающий перевод строки нужен, чтобы хук end-of-file-fixer не менял файл при коммите.
        raw_text = result.raw_text if result.raw_text.endswith("\n") else result.raw_text + "\n"
        args.save_raw.write_text(raw_text, encoding="utf-8")
        print(f"Сырой ответ сохранён: {args.save_raw}")
    return EXIT_OK


def _parse_utc_minute(text: str) -> int:
    """'2026-11-01T00:00' (UTC) -> миллисекунды."""
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return to_ms(moment.astimezone(UTC))


def cmd_bingx_probe_klines(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        anchor_ms = _parse_utc_minute(args.anchor)
        volume_bar_ms = _parse_utc_minute(args.volume_bar)
    except (ConfigError, ValueError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    symbol = args.symbol or cfg.symbols.symbols[0].symbol
    configure_logging(cfg.app.log_level)
    now_ms = int(time.time() * 1000)
    if anchor_ms % 900_000 or volume_bar_ms % 900_000:
        print("ОШИБКА: --anchor и --volume-bar должны лежать на границе 15 минут", file=sys.stderr)
        return EXIT_CONFIG
    needed_bars = 1800 if args.probe_set == "paging" else 1600
    if anchor_ms + needed_bars * 900_000 > now_ms:
        print(
            "ОШИБКА: --anchor слишком близко к текущему времени "
            f"(нужно >= {needed_bars * 15 // 60 // 24 + 1} суток назад)",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    cases = (
        build_paging_cases(anchor_ms)
        if args.probe_set == "paging"
        else build_cases(anchor_ms, volume_bar_ms)
    )
    print(f"Пробные запросы klines: {args.base_url}, символ {symbol}, запросов: {len(cases)}")
    print(f"Опорная граница: {format_ts(anchor_ms)}; свеча для volume: {format_ts(volume_bar_ms)}")
    print(f"Локальное время сейчас: {format_ts(now_ms)} (по часам этой машины)\n")
    with BingXPublicClient(args.base_url) as client:
        results = run_probe(client, symbol, cases, sleep=time.sleep)
    for line in render_results(results):
        print(line)
    print("\nВыводы по этим запросам:")
    findings = (
        derive_paging_findings(results, anchor_ms, now_ms)
        if args.probe_set == "paging"
        else derive_findings(results, anchor_ms, now_ms, volume_bar_ms)
    )
    for line in findings:
        print(f"  - {line}")
    failed = [r for r in results if r.candles is None]
    return EXIT_FAILURE if failed and len(failed) == len(results) else EXIT_OK


def cmd_data_download(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        start_ms = _parse_utc_minute(args.start)
        end_ms = _parse_utc_minute(args.end) if args.end else None
    except (ConfigError, ValueError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    symbols: list[str] = args.symbol or [cfg.symbols.symbols[0].symbol]
    tfs: list[str] = args.tf or ["15m", "1m"]
    window_bars: int = args.window_bars
    data_dir = resolve_data_dir(cfg.app)
    store = CandleStore(data_dir.path)
    try:
        for tf in tfs:
            if tf not in VALID_INTERVALS:
                raise ValueError(f"интервал {tf!r} не из списка {sorted(VALID_INTERVALS)}")
            for symbol in symbols:
                store.tf_dir(symbol, tf)  # проверка вида пары и таймфрейма
        if not 1 <= window_bars <= MAX_KLINES_LIMIT:
            raise ValueError(f"--window-bars: ожидается число от 1 до {MAX_KLINES_LIMIT}")
    except ValueError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level)

    now_ms = int(time.time() * 1000)
    end_text = format_ts(end_ms) if end_ms is not None else "последняя закрытая свеча"
    print(f"Каталог данных: {data_dir.path} (источник: {data_dir.source})")
    print(f"Пары: {', '.join(symbols)}; таймфреймы: {', '.join(tfs)}")
    print(f"Период: с {format_ts(start_ms)} до {end_text}; окно {window_bars} баров\n")

    exit_code = EXIT_OK
    gaps = 0
    with BingXPublicClient(args.base_url) as client:
        limiter = RateLimiter()
        for symbol in symbols:
            for tf in tfs:
                try:
                    report = download_tf(
                        client,
                        store,
                        symbol,
                        tf,
                        start_ms,
                        end_ms if end_ms is not None else now_ms,
                        now_ms=now_ms,
                        limiter=limiter,
                        progress=print,
                        window_bars=window_bars,
                    )
                except (BingXError, StoreError) as exc:
                    print(
                        f"{symbol} {tf}: ОШИБКА: {exc}\n"
                        "  накопленное сохранено; повторный запуск продолжит с этого места",
                        file=sys.stderr,
                    )
                    exit_code = EXIT_FAILURE
                    continue
                for line in render_report(report):
                    print(line)
                gaps += len(report.missing)
    if gaps:
        print(
            f"\nПРЕДУПРЕЖДЕНИЕ: пропусков всего {gaps}. Повторный запуск попробует их докачать; "
            "пометку «подтверждённо отсутствует» добавит проверка качества (P01-T03)."
        )
    return exit_code


def _mask_key(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}" if len(key) >= 12 else "***"


_AUTH_HINTS = (
    "Что проверить:\n"
    "  1. BINGX_API_KEY и BINGX_API_SECRET в .env без кавычек и пробелов;\n"
    "  2. BINGX_ENV совпадает с окружением, где создан ключ (vst или live);\n"
    "  3. если включена привязка к IP, текущий внешний IP этой машины есть в списке;\n"
    "  4. время на компьютере синхронизировано (окно подписи 5000 мс);\n"
    "  5. у ключа включено чтение."
)


def cmd_bingx_auth_check(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        secrets = load_bingx_secrets(args.env_file)
    except ConfigError as exc:
        print(f"Конфиг: ОШИБКА\n{exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level, secrets.secret_values())

    if secrets.bingx_api_key is None or secrets.bingx_api_secret is None:
        print(
            "BINGX_API_KEY и BINGX_API_SECRET должны быть заданы в .env "
            "(инструкция: docs/runbook.md, раздел BingX)",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    base_url = VST_BASE_URL if secrets.bingx_env == "vst" else LIVE_BASE_URL
    print(f"Окружение: {secrets.bingx_env} ({base_url})")
    print(f"Ключ: {_mask_key(secrets.bingx_api_key.get_secret_value())}")
    if secrets.bingx_env == "live":
        print("Боевой контур: выполняются только запросы на чтение.")

    show_amounts = secrets.bingx_env == "vst" or args.show_balance
    with BingXReadOnlyClient(secrets.bingx_api_key, secrets.bingx_api_secret, base_url) as client:
        try:
            balance = client.fetch_balance()
        except BingXError as exc:
            print(f"Подписанный запрос: ОШИБКА: {exc}", file=sys.stderr)
            print(_AUTH_HINTS, file=sys.stderr)
            return EXIT_FAILURE
        print("Подписанный запрос (баланс): OK, подпись принята")
        print(f"Активов в ответе: {len(balance.assets)} ({', '.join(balance.assets)})")
        if balance.fields and show_amounts:
            details = ", ".join(f"{k}={v}" for k, v in balance.fields.items())
            print(f"Актив {balance.shown_asset}: {details}")
        elif balance.fields:
            print("Суммы скрыты (добавь --show-balance, чтобы показать).")

        try:
            payload = client.fetch_api_permissions()
            permissions = extract_permissions(payload)
        except BingXError as exc:
            print(f"Права ключа: проверить не удалось ({exc}); эндпоинт не подтверждён.")
        else:
            fields = permission_fields(permissions)
            print(f"Права ключа (поля из ответа): {fields or 'полей не найдено'}")
            if "ipAddresses" in fields:
                ips = fields["ipAddresses"]
                count = len(ips) if isinstance(ips, list) else ips
                print(f"ipAddresses: в списке {count} адресов")
            print(f"Форма ответа (только имена и типы): {describe_shape(payload)}")
            print(
                "Примечание: значения кодов permissions не сверены с документацией, команда "
                "выводов о правах ключа не делает. Проверь права вручную в API Management."
            )
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
